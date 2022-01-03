import torch
from torch import nn
from torchvision import models
from attention import *
from bam import *
from cbam import *

class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


class ResNet101(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet101(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class Spatial_path(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        print("before avg pool ")
        x = self.avgpool(input)

        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(self.bn(x))
        #x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from spatial path) + 1024(from context path) + 2048(from context path)
        # resnet18  1024 = 256(from spatial path) + 256(from context path) + 512(from context path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class BiSeNet(nn.Module):
    def __init__(self, num_classes, context_path, output_aux=True):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()
        self.output_aux = output_aux

        # build context path
        self.backbone_model = {
            'resnet18': ResNet18(pretrained=True),
            'resnet101': ResNet101(pretrained=True)
        }

        self.context_path = self.backbone_model[context_path]

        # build attention refinement module  for resnet 101
        if context_path == 'resnet101':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 3328)

        elif context_path == 'resnet18':
            # build attention refinement module  for resnet 18
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 1024)
        else:
            print('Error: unspport context_path network \n')

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

        self.init_weight()

        self.mul_lr = []
        self.mul_lr.append(self.saptial_path)
        self.mul_lr.append(self.attention_refinement_module1)
        self.mul_lr.append(self.attention_refinement_module2)
        self.mul_lr.append(self.supervision1)
        self.mul_lr.append(self.supervision2)
        self.mul_lr.append(self.feature_fusion_module)
        self.mul_lr.append(self.conv)

    def init_weight(self):
        for name, m in self.named_modules():
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)

        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, size=sx.size()[-2:], mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, size=sx.size()[-2:], mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)

        if self.training == True:
            cx1_sup = self.supervision1(cx1)
            cx2_sup = self.supervision2(cx2)
            cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=input.size()[-2:], mode='bilinear')
            cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=input.size()[-2:], mode='bilinear')

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
        result = self.conv(result)

        if self.training == True:
            if self.output_aux:
                return result, cx1_sup, cx2_sup
            else:
                return result
        return result


class MyBiSeNet(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        # build spatial path
        self.spatial_path = Spatial_path()
        # build context path
        self.context_path = ResNet101(pretrained=True)

        self.bam = BAM(1024)
        self.cbam = CBAM(2048)

        self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
        self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)

        self.att = Att(3072)
        self.feature_fusion_module = FeatureFusionModule(num_classes, 3328)

        # build final convolution
        self.finalConv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)

    def forward(self, input):
        # output of spatial path
        print("forward of input ", input.shape)
        sx = self.spatial_path(input)
        print("spatial path sx shape ", sx.shape)

        # output of context path
        cx1, cx2, tail = self.context_path(input)
        print("resnet output cx1 ", cx1.shape)
        print("resnet cx2 ", cx2.shape)
        print("resnet tail ", tail.shape)


        cx11 = self.bam(cx1)
        print("cx11 ", cx11.shape)
        cx21 = self.cbam(cx2)
        print("cx21 ", cx21.shape)

        cx12 = self.attention_refinement_module1(cx11)
        print("cx12 ", cx12.shape)

        cx22 = torch.mul(cx21, tail)
        print("cx22 ", cx22.shape)

        # upsampling
        cx13 = torch.nn.functional.interpolate(cx12, size=sx.size()[-2:], mode='bilinear')
        print("cx13 ", cx13.shape)
        cx23 = torch.nn.functional.interpolate(cx22, size=sx.size()[-2:], mode='bilinear')
        print("cx23 ", cx23.shape)

        cx = torch.cat((cx13, cx23), dim=1)
        print("cx ", cx.shape)

        cxAtt = self.att(cx)
        print("cxAtt ", cxAtt.shape)

        res1 = self.feature_fusion_module(sx, cxAtt)
        print("res1 ", res1.shape)

        # upsampling
        res2 = torch.nn.functional.interpolate(res1, scale_factor=8, mode='bilinear')
        print("res2 ", res2.shape)
        #final convolution
        result = self.finalConv(res2)
        print("result ", result.shape)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = BiSeNet(32, 'resnet18')
    # model = nn.DataParallel(model)

    model = model  #.cuda()
    inputBatchX = torch.rand(4, 3, 512, 512)
    record = model.parameters()

    print(model.parameters())
    outputX = model(inputBatchX)
    #print("outputX shape ", outputX[0].shape)

    # convBlock = ConvBlock(in_channels=3, out_channels=64)
    # inputBatchX = torch.rand(4, 3, 512, 512)
    # print("output convBlock shape ", convBlock(inputBatchX).shape)
    #
    # spatialPath = Spatial_path()   #out channel 256, size /8
    # print("output spatial path " , spatialPath(inputBatchX).shape)
    #
    # attentionRefine = AttentionRefinementModule(3, 3)  #same
    # print("output attentionRefine ", attentionRefine(inputBatchX).shape)
    #
    # inputBatchX = torch.rand(4, 16, 512, 512)
    # inputBatchY = torch.rand(4, 64, 512, 512)
    # featureFusion = FeatureFusionModule(3, 80)     #outChannel,  input(channel1+channel2), same shape
    # featureFusionOut = featureFusion(inputBatchX, inputBatchY)
    # print("featureFusionOut ", featureFusionOut.shape)

    inputBatchX = torch.rand(4, 3, 512, 512)
    modelMyBiSeNet = MyBiSeNet(num_classes = 2)

    outputBiSetNet = modelMyBiSeNet(inputBatchX)






