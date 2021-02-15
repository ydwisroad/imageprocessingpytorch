import torch
import torch.nn as nn
import torchvision
import numpy as np

import sys
import os
sys.path.append(os.path.abspath("../../"))
from toolbox.blocks.attention import PAM_CAM_Layer

def bilinear_kernel(in_channels, out_channels, kernel_size):
    """Define a bilinear kernel according to in channels and out channels.
    Returns:
        return a bilinear filter tensor
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
    return torch.from_numpy(weight)

class DilationEncode(nn.Module):
    def __init__(self, in_plans, out_plans, kernel_size=1,  stride=1, padding=0,
                 dilation=1, dropout_prob=0.001):
        super(DilationEncode, self).__init__()
        self.lastEncoder = nn.Sequential(
            nn.Conv2d(in_plans, out_plans, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.Dropout2d(p=dropout_prob),
            nn.BatchNorm2d(out_plans),
            nn.LeakyReLU(0.1)
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.lastEncoder(x)

        return x

class CompDecoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, output_padding=0, dropout_prob=0.001, bias=False):
        super(CompDecoder, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//2, kernel_size=1, stride=1, padding=padding, bias=bias),
                                nn.Dropout2d(p=dropout_prob),
                                nn.BatchNorm2d(in_planes//2),
                                nn.LeakyReLU(0.1),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//2, in_planes//2, kernel_size=1, stride=1, padding=padding, bias=bias),
                                   nn.Dropout2d(p=dropout_prob),
                                nn.BatchNorm2d(in_planes//2),
                                nn.LeakyReLU(0.1),)

        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//2, in_planes//2, kernel_size=2,
                                     stride=2, padding=0, dilation = 1, output_padding=0, bias=bias),
                                nn.BatchNorm2d(in_planes//2),
                                nn.LeakyReLU(0.1),)

        self.conv3 = nn.Sequential(nn.Conv2d(in_planes//2, out_planes, kernel_size=1, stride=1, padding=padding, bias=bias),
                                nn.Dropout2d(p=dropout_prob),
                                nn.BatchNorm2d(out_planes),
                                nn.LeakyReLU(0.1),)

    def forward(self, x):
        #print("CompDecoder input ", x.size())
        x = self.conv1(x)
        #print("CompDecoder conv1 ", x.size())
        x = self.conv2(x)
        #print("CompDecoder conv2 ", x.size())
        x = self.tp_conv(x)
        #print("CompDecoder tp_conv ", x.size())
        x = self.conv3(x)
        #print("CompDecoder conv3 ", x.size())
        return x


class CompDecoderSameSize(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, output_padding=0, dropout_prob=0.001, bias=False):
        super(CompDecoderSameSize, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//2, kernel_size=1, stride=1, padding=padding, bias=bias),
                                nn.Dropout2d(p=dropout_prob),
                                nn.BatchNorm2d(in_planes//2),
                                nn.LeakyReLU(0.1),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//2, in_planes//2, kernel_size=1, stride=1, padding=padding, bias=bias),
                                   nn.Dropout2d(p=dropout_prob),
                                nn.BatchNorm2d(in_planes//2),
                                nn.LeakyReLU(0.1),)

        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//2, in_planes//2, kernel_size=1,
                                     stride=1, padding=0, output_padding=0, bias=bias),
                                nn.BatchNorm2d(in_planes//2),
                                nn.LeakyReLU(0.1),)

        self.conv3 = nn.Sequential(nn.Conv2d(in_planes//2, out_planes, kernel_size=1, stride=1, padding=padding, bias=bias),
                                   nn.Dropout2d(p=dropout_prob),
                                nn.BatchNorm2d(out_planes),
                                nn.LeakyReLU(0.1),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.tp_conv(x)
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.LeakyReLU(0.1),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.LeakyReLU(0.1),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.LeakyReLU(0.1),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)
        return x

class ResAttentionNet(nn.Module):
    def __init__(self, n_classes=2):
        super(ResAttentionNet, self).__init__()
        resnet34base = torchvision.models.resnet34(pretrained=False)
        self.in_block = nn.Sequential(
            resnet34base.conv1,
            resnet34base.bn1,
            resnet34base.relu,
            resnet34base.maxpool
        )   #1, 64, 112, 112

        self.encoder1 = resnet34base.layer1
        self.encoder2 = resnet34base.layer2
        self.encoder3 = resnet34base.layer3
        self.encoder4 = resnet34base.layer4

        self.encoder5 = DilationEncode(in_plans=512, out_plans=1024, kernel_size=2,  stride=2, padding=0,
                 dilation=1, dropout_prob=0.003)

        self.pam_attention_1_1= PAM_CAM_Layer(32, True)
        self.cam_attention_1_1= PAM_CAM_Layer(32, False)

        self.pam_attention_1_2= PAM_CAM_Layer(64, True)
        self.cam_attention_1_2= PAM_CAM_Layer(64, False)

        self.pam_attention_1_3= PAM_CAM_Layer(128, True)
        self.cam_attention_1_3= PAM_CAM_Layer(128, False)

        self.pam_attention_1_4= PAM_CAM_Layer(256, True)
        self.cam_attention_1_4= PAM_CAM_Layer(256, False)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        self.comDecoder5 = CompDecoder(in_planes=1024, out_planes=512, kernel_size=1,  stride=1, padding=0)
        self.comDecoder4 = CompDecoder(in_planes=512, out_planes=256, kernel_size=1, stride=1, padding=0)
        self.comDecoder3 = CompDecoder(in_planes=256, out_planes=128, kernel_size=1, stride=1, padding=0)
        self.comDecoder2 = CompDecoder(in_planes=128, out_planes=64, kernel_size=1, stride=1, padding=0)
        self.comDecoder1 = CompDecoderSameSize(in_planes=64, out_planes=64, kernel_size=1, stride=1, padding=0)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.Dropout2d(p=0.001),
                                      nn.BatchNorm2d(32),
                                      nn.LeakyReLU(0.1),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.Dropout2d(p=0.001),
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(0.1),)
        self.conv3 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1),
                                nn.Dropout2d(p=0.001),
                                nn.BatchNorm2d(16),
                                nn.LeakyReLU(0.1),)
        self.tp_conv2 = nn.ConvTranspose2d(16, n_classes, 2, 2, 0)
        #self.tp_conv2.weight.data = bilinear_kernel(16, n_classes, 2)

    def forward(self, x):
        # Initial block
        #print("x original block ", x.size())
        x = self.in_block(x)

        lamda = 0.2
        #print("1.after in block ", x.size())
        # Encoder blocks
        e1 = self.encoder1(x)
        #print("2.after encoder1 ", e1.size())
        attn_pam1 = self.pam_attention_1_1(e1)
        #print("2.after e1 attention attn_pam1 ", attn_pam1.size())
        attn_cam1 = self.cam_attention_1_1(e1)
        #print("2.after e1 attention attn_cam1 ", attn_cam1.size())
        #fusionattn1 = torch.cat((attn_pam1,attn_cam1),1)

        fusionattn1 = attn_pam1 * (1-lamda) + lamda * attn_cam1
        #print("fusion attn1" , fusionattn1.size())

        e2 = self.encoder2(e1)
        #print("3.after encoder2 ", e2.size())
        attn_pam2 = self.pam_attention_1_2(e2)
        #print("3.after e1 attention attn_pam2 ", attn_pam2.size())
        attn_cam2 = self.cam_attention_1_2(e2)
        #print("3.after e1 attention attn_cam2 ", attn_cam2.size())
        #fusionattn2 = torch.cat((attn_pam2, attn_cam2), 1)
        fusionattn2 = attn_pam2 * (1 - lamda) + lamda * attn_cam2
        #print("fusion attn2", fusionattn2.size())

        e3 = self.encoder3(e2)
        #print("4.after encoder3 ", e3.size())
        attn_pam3 = self.pam_attention_1_3(e3)
        #print("4.after e1 attention attn_pam3 ", attn_pam3.size())
        attn_cam3 = self.cam_attention_1_3(e3)
        #print("4.after e1 attention attn_cam3 ", attn_cam3.size())
        #fusionattn3 = torch.cat((attn_pam3, attn_cam3), 1)
        fusionattn3 = attn_pam3 * (1 - lamda) + lamda * attn_cam3
        #print("fusion attn3", fusionattn3.size())

        e4 = self.encoder4(e3)
        #print("5.after encoder4 ", e4.size())
        attn_pam4 = self.pam_attention_1_4(e4)
        #print("5.after e1 attention attn_pam4 ", attn_pam4.size())
        attn_cam4 = self.cam_attention_1_4(e4)
        #print("5.after e1 attention attn_cam4 ", attn_cam4.size())
        #fusionattn4 = torch.cat((attn_pam4, attn_cam4), 1)
        fusionattn4 = attn_pam4 * (1 - lamda) + lamda * attn_cam4
        #print("fusion attn4", fusionattn4.size())

        e5 = self.encoder5(e4)
        #print("e5 size", e5.size())

        dcomp5 = self.comDecoder5(e5)
        #print("dcomp5 size", dcomp5.size())

        # Decoder blocks
        d4 = e3 + self.decoder4(dcomp5 + fusionattn4)
        #d4 = e3 + self.decoder4(e4 + fusionattn4)
        #print("d4 got  ", d4.size())
        #dcomp4 = self.comDecoder4(e4 + fusionattn4)
        #print("dcomp4 size", dcomp4.size())

        d3 = e2 + self.decoder3(d4 + fusionattn3)
        #print("d3 got  ", d3.size())
        #dcomp3 = self.comDecoder3(dcomp4 + fusionattn3)
        #print("dcomp3 size", dcomp3.size())

        d2 = e1 + self.decoder2(d3 + fusionattn2)
        #print("d2 got  ", d2.size())
        #dcomp2 = self.comDecoder2(dcomp3 + fusionattn2)
        #print("dcomp2 size", dcomp2.size())

        d1 = x + self.decoder1(d2 + fusionattn1)
        #print("d1 got  ", d1.size())
        #dcomp1 = self.comDecoder1(dcomp2 + fusionattn1)
        #print("dcomp1 size", dcomp1.size())

        y = self.tp_conv1(d1)
        #print("y after topConv1  ", y.size())
        y = self.conv2(y)
        #print("y after conv2  ", y.size())
        y = self.conv3(y)
        #print("y after conv3  ", y.size())
        y = self.tp_conv2(y)
        #print("y after tp_conv2  ", y.size())

        return y

if __name__ == '__main__':

    inputs = torch.randn(4, 3, 448, 448)
    resnet34Model = ResAttentionNet(n_classes=2)

    out = resnet34Model(inputs)
    #print(out.size())

    input2 = torch.randn(4, 512,14,14)
    dilationEncoder = DilationEncode(in_plans=512, out_plans=1024, kernel_size=2,  stride=2, padding=0,
                 dilation=1, dropout_prob=0.003)
    output2 = dilationEncoder(input2)
    #print(output2.size())

    input3 = torch.randn(4, 1024, 7, 7)
    compDecoder = CompDecoder(in_planes=1024, out_planes=512, kernel_size=1,  stride=1, padding=0)
    output3 = compDecoder(input3)
    #print(output3.size())
