# change from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down_conv, self).__init__()
        self.go = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.go(x)


class up_concate_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_concate_conv, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = double_conv(in_ch * 2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if x1.size(2) != x2.size(2) or x1.size(3) != x2.size(3):
            x1 = F.interpolate(x1, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=True)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class unet(nn.Module):
    def __init__(self, n_classes, input_channel=3):
        super(unet, self).__init__()
        self.pre_conv = double_conv(input_channel, 64)  # 1
        self.down1 = down_conv(64, 128)  # 1/2
        self.down2 = down_conv(128, 256)  # 1/4
        self.down3 = down_conv(256, 512)  # 1/8
        self.down4 = down_conv(512, 512)  # 1/16

        self.up1 = up_concate_conv(512, 256)
        self.up2 = up_concate_conv(256, 128)
        self.up3 = up_concate_conv(128, 64)
        self.up4 = up_concate_conv(64, 64)
        self.end_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.pre_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.end_conv(x)
        return x


if __name__ == '__main__':
    inputs = torch.randn((4, 3, 360, 480)).cuda()
    model = unet(n_classes=12).cuda()
    out = model(inputs)
    print(out.size())
