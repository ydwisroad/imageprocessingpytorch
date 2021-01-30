import torch.nn as nn

"""
https://zhuanlan.zhihu.com/p/76378871
arxiv: 1804.03821
ExFuse
"""

class SematicEmbbedBlock(nn.Module):
    def __init__(self, high_in_plane, low_in_plane, out_plane):
        super(SematicEmbbedBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(high_in_plane, out_plane, 3, 1, 1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1x1 = nn.Conv2d(low_in_plane, out_plane, 1)

    def forward(self, high_x, low_x):
        high_x = self.upsample(self.conv3x3(high_x))
        low_x = self.conv1x1(low_x)
        return high_x * low_x

if __name__ == '__main__':
    semanticEmb = SematicEmbbedBlock(64, 32, 16)

    import torch

    testx1 = torch.randn(8, 64, 160, 160)
    testx2 = torch.randn(8, 32, 320, 320)

    print("RFBSmall input shape ", testx1.shape, " ", testx2.shape)
    output = semanticEmb(testx1, testx2)
    print("RFBSmall output shape ", output.shape)