import torch
import torch.nn as nn
import math
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

from myMixConv import *
from myghostnet import GhostBottleneck


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.swish(x)

        return x

class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class BoxNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
        super(BoxNet, self).__init__()
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        # 每一个有效特征层对应的Batchnor不同
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        # 9
        # 4 中心，宽高
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        # 对每个特征层循环
        for feat, bn_list in zip(inputs, self.bn_list):
            # 每个特征层需要进行num_layer次卷积+标准化+激活函数
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)
        # 进行一个堆叠
        feats = torch.cat(feats, dim=1)

        return feats


class ClassNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(ClassNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        # 每一个有效特征层对应的BatchNorm2d不同
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        # num_anchors = 9
        # num_anchors num_classes
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        # 对每个特征层循环
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                # 每个特征层需要进行num_layer次卷积+标准化+激活函数
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)
        # 进行一个堆叠
        feats = torch.cat(feats, dim=1)
        # 取sigmoid表示概率
        feats = feats.sigmoid()

        return feats

def testCreationAndOut():
    #denseTest
    inputs = torch.randn(4, 3, 512, 512)
    print("input size ", inputs.size())

    upsample = nn.Upsample(scale_factor=2, mode='nearest')
    upsampleOutput = upsample(inputs)
    print("up sample size ", upsampleOutput.size())

    downsample = ConvMixer(3,1)
    downsampleOutput = downsample(inputs)
    print("down sample size ", downsampleOutput.size())

    sameConv2d = nn.Conv2d(3, 3, 1, 1)
    sameConvOutput = sameConv2d(inputs)
    print("same Conv Output size ", sameConvOutput.size())

    divideTwoConv2d = nn.Conv2d(3, 3, 2, 2)
    divideTwooutput = divideTwoConv2d(inputs)
    print("divide two output size ", divideTwooutput.size())

    # 简单的注意力机制，用于确定更关注p7_in还是p6_in
    swish = Swish()
    p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    p6_in = inputs
    p7_in = sameConvOutput

    epsilon = 1e-4
    p6_w1_relu = nn.ReLU()
    p6_w1 = p6_w1_relu(p6_w1)
    weight = p6_w1 / (torch.sum(p6_w1, dim=0) + epsilon)
    p6_attention_weight_out= swish(weight[0] * p6_in + weight[1] * p7_in)
    print("attention_weight_out size ", p6_attention_weight_out.size())

    #down channel
    down_channel = nn.Sequential(
        Conv2dStaticSamePadding(32, 16, 1),
        nn.BatchNorm2d(16, momentum=0.01, eps=1e-3),
    )
    inputsDownChan = torch.randn(4, 32, 512, 512)
    outDownChan = down_channel(inputsDownChan)
    print("down channel size ", outDownChan.size())

class MyEfficientNet(nn.Module):
    def __init__(self):
        super(MyEfficientNet, self).__init__()
        #in_chs, mid_chs, out_chs
        self.p1Div2 = GhostBottleneck(3, 16, 8, stride=2)
        self.p2Div2 = GhostBottleneck(8, 32, 16, stride=2)
        self.p3Div2 = GhostBottleneck(16, 24, 32, stride=2)
        self.p4Div2 = GhostBottleneck(32, 8, 64, stride=2)
        self.p5Div2 = GhostBottleneck(64, 32, 128, stride=2)
        self.p6Div2 = GhostBottleneck(128, 64, 256, stride=2)

        self.p2_same_dim = nn.Sequential(Conv2dStaticSamePadding(16, 64, 1),
                                          nn.BatchNorm2d(64, momentum=0.01, eps=1e-3),)
        self.p3_same_dim = nn.Sequential(Conv2dStaticSamePadding(32, 64, 1),
                                          nn.BatchNorm2d(64, momentum=0.01, eps=1e-3),)
        self.p4_same_dim = nn.Sequential(Conv2dStaticSamePadding(64, 64, 1),
                                          nn.BatchNorm2d(64, momentum=0.01, eps=1e-3),)
        self.p5_same_dim = nn.Sequential(Conv2dStaticSamePadding(128, 64, 1),
                                          nn.BatchNorm2d(64, momentum=0.01, eps=1e-3),)
        self.p6_same_dim = nn.Sequential(Conv2dStaticSamePadding(256, 64, 1),
                                          nn.BatchNorm2d(64, momentum=0.01, eps=1e-3),)

        self.downSample = ConvMixer(64,3,64)
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest')

        self.regressor      = BoxNet(in_channels=64, num_anchors=9,
                                    num_layers=3)

        self.classifier     = ClassNet(in_channels=64, num_anchors=9,
                                    num_classes=3, num_layers=3)


    def calAttention(self, x1, x2):
        swish = Swish()
        x1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        epsilon = 1e-4
        p6_w1_relu = nn.ReLU()
        p6_w1 = p6_w1_relu(x1_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + epsilon)
        attention_weight_out = swish(weight[0] * x1 + weight[1] * x2)
        return attention_weight_out

    def forward(self, x):
        p0 = x
        print("p0 size ", p0.size())
        p1 = self.p1Div2(p0)
        print("p1 size ", p1.size())
        p2 = self.p2Div2(p1)
        print("p2 size ", p2.size())
        p3 = self.p3Div2(p2)
        print("p3 size ", p3.size())
        p4 = self.p4Div2(p3)
        print("p4 size ", p4.size())
        p5 = self.p5Div2(p4)
        print("p5 size ", p5.size())
        p6 = self.p6Div2(p5)
        print("p6 size ", p6.size())

        p2_1 = self.p2_same_dim(p2)
        print("p2_1 size ", p2_1.size())
        p3_1 = self.p3_same_dim(p3)
        print("p3_1 size ", p3_1.size())
        p4_1 = self.p4_same_dim(p4)
        print("p4_1 size ", p4_1.size())
        p5_1 = self.p5_same_dim(p5)
        print("p5_1 size ", p5_1.size())
        p6_1 = self.p6_same_dim(p6)
        print("p6_1 size ", p6_1.size())

        p2_1_down = self.downSample(p2_1)
        print("p2_1_down ", p2_1_down.size())
        p3_1_down = self.downSample(p3_1)
        print("p3_1_down ", p3_1_down.size())
        p4_1_down = self.downSample(p4_1)
        print("p4_1_down ", p4_1_down.size())
        p5_1_down = self.downSample(p5_1)
        print("p5_1_down ", p5_1_down.size())

        p3_attenOut = self.calAttention(p2_1_down, p3_1)
        print("p3_attenOut ", p3_attenOut.size())
        p4_attenOut = self.calAttention(p3_1_down, p4_1)
        print("p4_attenOut ", p4_attenOut.size())
        p5_attenOut = self.calAttention(p4_1_down, p5_1)
        print("p5_attenOut ", p5_attenOut.size())
        p6_attenOut = self.calAttention(p5_1_down, p6_1)
        print("p6_attenOut ", p6_attenOut.size())

        p6_1_up = self.upSample(p6_attenOut)
        print("p6_1_up ", p6_1_up.size())
        p5_1_up = self.upSample(p5_attenOut)
        print("p5_1_up ", p5_1_up.size())
        p4_1_up = self.upSample(p4_attenOut)
        print("p4_1_up ", p4_1_up.size())
        p3_1_up = self.upSample(p3_attenOut)
        print("p3_1_up ", p3_1_up.size())

        p2_Out = self.calAttention(p3_1_up, p2_1)
        print("p2_Out ", p2_Out.size())
        p3_Out = self.calAttention(p4_1_up, p3_attenOut)
        print("p3_Out ", p3_Out.size())
        p4_Out = self.calAttention(p5_1_up, p4_attenOut)
        print("p4_Out ", p4_Out.size())
        p5_Out = self.calAttention(p6_1_up, p5_attenOut)
        print("p5_Out ", p5_Out.size())
        p6_Out = p6_attenOut
        print("p6_Out ", p6_Out.size())

        features = (p2_Out, p3_Out, p4_Out, p5_Out, p6_Out)
        regression = self.regressor(features)
        print("regression:", regression.size())
        classification = self.classifier(features)
        print("classification: ", classification.size())

        return p2_Out, p3_Out, p4_Out, p5_Out, p6_Out


if __name__ == "__main__":
    print("Test simple efficient det starts.")
    inputs = torch.randn(4, 3, 512, 512)
    print("input size ", inputs.size())

    myEfficientNet = MyEfficientNet()
    allOut = myEfficientNet(inputs)










