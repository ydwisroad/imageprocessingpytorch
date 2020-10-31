import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision import models

class ConvBatchNormReluRModule(nn.Module):
    #Conv => BatchNorm => Relu
    def __init__(self, in_channel, out_channel):
        super(ConvBatchNormReluRModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

#max Pooling => ConvBatchNormRelu => ConvBatchNormRelu
class DownConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownConv, self).__init__()
        self.downConv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBatchNormReluRModule(in_channel, out_channel),
            ConvBatchNormReluRModule(out_channel, out_channel)
        )

    def forward(self, x):
        x = self.downConv(x)
        return x

class UpConcatConv(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(UpConcatConv, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)

        self.conv1 = ConvBatchNormReluRModule(in_channel * 2, out_channel)
        self.conv2 = ConvBatchNormReluRModule(out_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if x1.size(2) != x2.size(2) or x1.size(3) != x2.size(3):
            x1 = F.interpolate(x1, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=True)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class MyUnet(nn.Module):
    def __init__(self, n_classes = 2, input_channel=3):
        super(MyUnet, self).__init__()
        self.pre_conv1 = ConvBatchNormReluRModule(input_channel, 64)  # 1
        self.pre_conv2 = ConvBatchNormReluRModule(64, 64)  # 1

        self.down11 = DownConv(64, 128)  # 1/2
        self.down21 = DownConv(128, 256)  # 1/4
        self.down31 = DownConv(256, 512)  # 1/8
        self.down41 = DownConv(512, 512)  # 1/16

        self.up1 = UpConcatConv(512, 256)
        self.up2 = UpConcatConv(256, 128)
        self.up3 = UpConcatConv(128, 64)
        self.up4 = UpConcatConv(64, 64)
        self.end_conv = nn.Conv2d(64, n_classes, 1)


    def forward(self, x):
        x1 = self.pre_conv1(x)
        x1 = self.pre_conv2(x1)   # 4, 64, 448, 448

        x2 = self.down11(x1)
        x3 = self.down21(x2)
        x4 = self.down31(x3)
        x5 = self.down41(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.end_conv(x)

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_dim,out_dim,act_fn,stride=1):
        super(ConvBlock, self).__init__()
        self.convBlock = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_dim),
                act_fn
            )

    def forward(self, x):
        x = self.convBlock(x)
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, in_dim,out_dim,act_fn):
        super(ConvTransBlock, self).__init__()
        self.convTransBlock = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )

    def forward(self, x):
        x = self.convTransBlock(x)
        return x

class ConvBlockThree(nn.Module):
    def __init__(self, in_dim,out_dim,act_fn,stride=1):
        super(ConvBlockThree, self).__init__()
        self.convBlockThree = nn.Sequential(
            ConvBlock(in_dim, out_dim, act_fn),
            ConvBlock(out_dim, out_dim, act_fn),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        x = self.convBlockThree(x)
        return x

class ConvResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(ConvResidualBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = ConvBlock(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = ConvBlockThree(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = ConvBlock(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)

        return conv_3


class MyFusionNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, out_clamp=None):
        super(MyFusionNet, self).__init__()
        self.out_clamp = out_clamp
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        act_fn = nn.ReLU()
        act_fn_2 = nn.ELU(inplace=True)

        # encoder
        self.down_1 = ConvResidualBlock(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = ConvBlock(self.out_dim, self.out_dim, act_fn, 2)
        self.down_2 = ConvResidualBlock(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = ConvBlock(self.out_dim * 2, self.out_dim * 2, act_fn, 2)
        self.down_3 = ConvResidualBlock(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = ConvBlock(self.out_dim * 4, self.out_dim * 4, act_fn, 2)
        self.down_4 = ConvResidualBlock(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = ConvBlock(self.out_dim * 8, self.out_dim * 8, act_fn, 2)

        # bridge
        self.bridge = ConvResidualBlock(self.out_dim * 8, self.out_dim * 16, act_fn)

        # decoder
        self.deconv_1 = ConvTransBlock(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = ConvResidualBlock(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = ConvTransBlock(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = ConvResidualBlock(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = ConvTransBlock(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = ConvResidualBlock(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = ConvTransBlock(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = ConvResidualBlock(self.out_dim, self.out_dim, act_fn_2)

        # output
        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.up_4(skip_4)

        out = self.out(up_4)

        return out



if __name__ == "__main__":
    print("Unet started")

    inputX = torch.randn(4, 3, 448, 448)
    cbReluModule = ConvBatchNormReluRModule(3, 64)
    outputX = cbReluModule(inputX)
    print(" outputX size " , outputX.size())

    downConv = DownConv(3, 64)
    outputDownConv = downConv(inputX)
    print(" outputDownConv size ", outputDownConv.size())

    inputX1 = torch.randn(4, 3, 224, 224)
    inputX2 = torch.randn(4, 3, 448, 448)

    upconcatConv = UpConcatConv(3, 64)
    outputX1X2 = upconcatConv(inputX1, inputX2)
    print(" outputX1X2 size ", outputX1X2.size())

    myUnet = MyUnet(2, 3)
    outputMyUnet = myUnet(inputX)
    print(" outputMyUnet size ", outputMyUnet.size())

    act_fn = nn.ReLU()
    act_fn_2 = nn.ELU(inplace=True)
    convResidBlock = ConvResidualBlock(3, 64,act_fn_2)
    outConvResBlock = convResidBlock(inputX)
    print(" outConvResBlock size ", outConvResBlock.size())

    convTransBlock = ConvTransBlock(3, 64, act_fn_2)
    outConvTransBlock = convTransBlock(inputX)
    print(" outConvTransBlock size ", outConvTransBlock.size())

    fusionNet = MyFusionNet(3, 64, 64)
    #outFusionNet = fusionNet(inputX)
    #print(" outFusionNet size ", outFusionNet.size())







