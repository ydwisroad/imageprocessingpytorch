import math
import numpy as np

import torch
import torch.nn as nn
#https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247510991&idx=3&sn=f6991d8bd20d63dd06a02f6011ebeed2&chksm=ec1c4636db6bcf20bed4a96e7b68d4bb6b72adb76186b6cf693c41fcfa643d7254839ec0e304&mpshare=1&scene=1&srcid=1019jxEY69lcUhJgYsMA0Q0q&sharer_sharetime=1603064716043&sharer_shareid=03101a931987a40bb1c69d01fec93b52&exportkey=AbCoFZ08zc3IyR1jtqVKGvY%3D&pass_ticket=KxAVxjqQ4Tok2JkD1jwdy7aa52e4fLkJ5TPDMuurQ%2BBoJD3TDbRrfRS18LwGuQch&wx_header=0#rd

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1/3)*(x_out + x_out11 + x_out21)
        else:
            x_out = (1/2)*(x_out11 + x_out21)
        return x_out


if __name__ == '__main__':
    print("start to test attention modules")

    import torch as t
    inputX = t.randn(4, 3, 20, 20)

    basicConv = BasicConv(3, 1, 3)
    outBasicConv = basicConv(inputX)
    print("outBasicConv size ", outBasicConv.size())

    print("input X size ", inputX.size())
    tripletAttention = TripletAttention()
    outTripletAttention = tripletAttention(inputX)
    print("out Triplet Attention ", outTripletAttention.size())


