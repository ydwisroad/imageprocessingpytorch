import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision import models

pretrained_VGGnet = models.vgg16_bn(pretrained=False)

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

class MyFCN(nn.Module):
    def __init__(self, n_classes):
        super(MyFCN, self).__init__()
        self.stage1 = pretrained_VGGnet.features[:7]
        self.stage2 = pretrained_VGGnet.features[7:14]
        self.stage3 = pretrained_VGGnet.features[14:24]
        self.stage4 = pretrained_VGGnet.features[24:34]
        self.stage5 = pretrained_VGGnet.features[34:]

        #inChannel, outChannel, kernelSize
        self.score1 = nn.Conv2d(512, n_classes, 1)
        self.score2 = nn.Conv2d(512, n_classes, 1)
        self.score3 = nn.Conv2d(128, n_classes, 1)

        self.conv_trans1 = nn.Conv2d(512,256,1)
        self.conv_trans2 = nn.Conv2d(256, n_classes, 1)
        #inChannels, outChannels, kernelSize, stride, padding
        self.upsample8x = nn.ConvTranspose2d(n_classes, n_classes, 16,8,4, bias=False)
        self.upsample8x.weight.data = bilinear_kernel(n_classes, n_classes, 16)

        self.upsample2x_1 = nn.ConvTranspose2d(512, 512, 4, 2 , 1, bias=False)
        self.upsample2x_1.weight.data = bilinear_kernel(512, 512, 4)
        self.upsample2x_2 = nn.ConvTranspose2d(256, 256, 4, 2 , 1, bias=False)
        self.upsample2x_2.weight.data = bilinear_kernel(256, 256, 4)

    def forward(self, x):
        print("size changes ")
        print("original size ", x.size())
        s1 = self.stage1(x)
        print("s1 size ", s1.size())
        s2 = self.stage2(s1)
        print("s2 size ", s2.size())
        s3 = self.stage3(s2)
        print("s3 size ", s3.size())
        s4 = self.stage4(s3)
        print("s4 size ", s4.size())
        s5 = self.stage5(s4)
        print("s5 size ", s5.size())

        score1 = self.score1(s5)
        print("score1 first size ", score1.size())
        s5 = self.upsample2x_1(s5)
        print("s5 first size ", s5.size())
        add1 = s4 + s5
        print("add1 first size ", add1.size())

        score2 = self.score2(add1)
        print("score2 first size ", score2.size())
        add1  =self.conv_trans1(add1)
        print("add1 second size ", add1.size())
        add1 = self.upsample2x_2(add1)
        print("add1 third size ", add1.size())
        add2 = add1 + s3
        print("add2 first size ", add2.size())

        add2 = self.conv_trans2(add2)
        print("add2 second size ", add2.size())
        score3 = self.upsample8x(add2)
        print("score3 size ", add2.size())

        return score3

if __name__ == "__main__":
    print("VGG Net Features")
    print(pretrained_VGGnet.features)
    print(pretrained_VGGnet.features[4:8])
    inputX = torch.randn(4, 3, 448, 448)

    stage1 = pretrained_VGGnet.features[:7]
    outStage1 = stage1(inputX)
    print(outStage1.size())   #This is correct

    myFCN = MyFCN(2)
    myFCN.eval()
    output = myFCN(inputX)