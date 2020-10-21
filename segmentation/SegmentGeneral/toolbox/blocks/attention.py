import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable



class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim = 8):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query_pam = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        #print("pam proj_query size ", proj_query_pam.size())
        proj_key_pam = self.key_conv(x).view(m_batchsize, -1, width * height)
        #print("pam proj_key size " , proj_key_pam.size())
        energy = torch.bmm(proj_query_pam, proj_key_pam)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim = 8):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query_cam = x.view(m_batchsize, C, -1)
        #print("cam proj_query size ", proj_query_cam.size())
        proj_key_cam = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        #print("cam proj_key size ", proj_key_cam.size())

        energy = torch.bmm(proj_query_cam, proj_key_cam)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

if __name__ == '__main__':
    print("start to test attention modules")
    import torch as t
    inputX = t.randn(4, 8, 20, 20)

    camModule = CAM_Module(8)
    camModule.eval()
    resCam = camModule(inputX)
    print("res CAM ", resCam.size())

    pamModule = PAM_Module(8)
    pamModule.eval()

    resPam = pamModule(inputX)
    print("res PAM ", resPam.size())