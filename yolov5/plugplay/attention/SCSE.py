import torch
import torch.nn as nn
import torch


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class csSE(nn.Module):
    def __init__(self, in_channels, psudoChannel):
        super().__init__()
        print("csSE in_channels ", in_channels)
        self.psudoChannel = psudoChannel
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        #print("csSE input size ", U.shape)
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        output = U_cse+U_sse
        #print("csSE output size ", output.shape)
        return output

if __name__ == "__main__":

    import torch

    bs, c, h, w = 8, 64, 160, 160
    in_tensor = torch.randn(bs, c, h, w)

    cs_se = csSE(c)
    print("in shape:",in_tensor.shape)
    out_tensor = cs_se(in_tensor)
    print("out shape:", out_tensor.shape)
