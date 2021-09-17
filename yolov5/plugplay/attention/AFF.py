# source: https://mp.weixin.qq.com/s/La6rbQpnZzjWH3psB2gD6Q
# code: https://github.com/YimianDai/open-aff
# https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py
# AFF
import torch
import torch.nn as nn


class DAF(nn.Module):
    '''
    直接相加 DirectAddFuse
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, sudoOutput=64, r=4):
        super(AFF, self).__init__()
        #print("AFF channels:", channels)
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("AFF x0 size ", x[0].shape)
        #print("AFF x1 size ", x[1].shape)
        #if (x[0].shape[0] < 2):
            #return torch.cat(x, 1)
        #print("residual size ", residual.shape)
        #x[0] + x[1]
        xa = x[0] + x[1]
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        #
        xo = 2 * x[0] * wei + 2 * x[1] * (1 - wei)
        #print("AFF xo size ", xo.shape)
        return xo

if __name__ == '__main__':
    module = AFF(64)

    testx1 = torch.randn(8, 64, 160, 160)
    testx2 = torch.randn(8, 64, 160, 160)
    print("input shape ", testx1.shape)
    output = module(testx1, testx2)
    print("output shape ", output.shape)
