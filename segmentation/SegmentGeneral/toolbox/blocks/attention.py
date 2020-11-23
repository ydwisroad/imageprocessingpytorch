import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

import math
import numpy as np

import pdb
__all__ = ['PAM_Module', 'CAM_Module', 'semanticModule']

class _EncoderBlock(nn.Module):
    """
    Encoder block for Semantic Attention Module
    """

    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    """
    Decoder Block for Semantic Attention Module
    """

    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class semanticModule(nn.Module):
    """
    Semantic attention module
    """

    def __init__(self, in_dim):
        super(semanticModule, self).__init__()
        self.chanel_in = in_dim

        self.enc1 = _EncoderBlock(in_dim, in_dim * 2)
        self.enc2 = _EncoderBlock(in_dim * 2, in_dim * 4)
        self.dec2 = _DecoderBlock(in_dim * 4, in_dim * 2, in_dim * 2)
        self.dec1 = _DecoderBlock(in_dim * 2, in_dim, in_dim)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        dec2 = self.dec2(enc2)
        dec1 = self.dec1(F.upsample(dec2, enc1.size()[2:], mode='bilinear'))

        return enc2.view(-1), dec1


class PAM_Module2(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module2, self).__init__()
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
        #print("PAM_Module input ", x.size())
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x)
        #print("PAM_Module proj_query1 ", proj_query.size())
        proj_query = proj_query.view(m_batchsize, -1, width * height)
        #print("PAM_Module proj_query2 ", proj_query.size())
        proj_query = proj_query.permute(0, 2, 1)
        #print("PAM_Module proj_query3 ", proj_query.size())

        proj_key = self.key_conv(x)
        #print("PAM_Module proj_key1 ", proj_key.size())
        proj_key = proj_key.view(m_batchsize, -1, width * height)
        #print("PAM_Module proj_key2 ", proj_key.size())

        energy = torch.bmm(proj_query, proj_key)
        #print("PAM_Module energy after bmm ", energy.size())
        attention = self.softmax(energy)
        #print("PAM_Module attention ", attention.size())

        proj_value = self.value_conv(x)
        #print("PAM_Module proj_value1 ", proj_value.size())
        proj_value = proj_value.view(m_batchsize, -1, width * height)
        #print("PAM_Module proj_value2 ", proj_value.size())

        attenPermute = attention.permute(0, 2, 1)
        #print("PAM_Module attenPermute ", attenPermute.size())
        out = torch.bmm(proj_value, attenPermute)
        #print("PAM_Module out1  after bmm", out.size())

        out = out.view(m_batchsize, C, height, width)
        #print("PAM_Module out2 ", out.size())

        out = self.gamma * out     #+ x
        #print("PAM_Module output ", out.size())
        return out


class CAM_Module2(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module2, self).__init__()
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
        #print("CAM_Module input ", x.size())
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        #print("CAM_Module proj_query ", proj_query.size())
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        #print("CAM_Module proj_key ", proj_key.size())

        energy = torch.bmm(proj_query, proj_key)
        #print("CAM_Module energy ", energy.size())

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        #print("CAM_Module energy_new ", energy_new.size())

        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out     # + x
        #print("CAM_Module output ", out.size())
        return out


class PAM_CAM_Layer2(nn.Module):
    """
    Helper Function for PAM and CAM attention

    Parameters:
    ----------
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """

    def __init__(self, in_ch, use_pam=True):
        super(PAM_CAM_Layer2, self).__init__()

        self.attnIn = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU())

        self.attn = PAM_Module(in_ch) if use_pam else CAM_Module(in_ch)

        self.attnOut = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )

    def forward(self, x):
        print("PAM_CAM_Layer input ", x.size())
        x = self.attnIn(x)
        print("PAM_CAM_Layer attnIn ", x.size())
        x = self.attn(x)
        print("PAM_CAM_Layer attn ", x.size())
        out = self.attnOut(x)
        print("PAM_CAM_Layer output ", out.size())
        return out


class MultiConv(nn.Module):
    """
    Helper function for Multiple Convolutions for refining.

    Parameters:
    ----------
    inputs:
        in_ch : input channels
        out_ch : output channels
        attn : Boolean value whether to use Softmax or PReLU
    outputs:
        returns the refined convolution tensor
    """

    def __init__(self, in_ch, out_ch, attn=True):
        super(MultiConv, self).__init__()

        self.fuse_attn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Softmax2d() if attn else nn.PReLU()
        )

    def forward(self, x):
        return self.fuse_attn(x)


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()

        self.chanel_in = in_dim

        self.conv2dsame = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(in_dim),
                nn.PReLU())

        self.conv2dsame1 = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(in_dim),
                nn.PReLU())

        self.conv2dsame2 = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(in_dim),
                nn.PReLU())

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
        #print("PAM_Module input ", x.size())
        m_batchsize, C, height, width = x.size()

        sameConvX = self.conv2dsame(x)
        proj_query = sameConvX.view(m_batchsize, C, -1).permute(0, 2, 1)
        #print("PAM_Module proj_query ", proj_query.size())
        proj_key = sameConvX.view(m_batchsize, C, -1)
        #print("PAM_Module proj_key ", proj_key.size())

        energy = torch.bmm(proj_query, proj_key)
        #print("PAM_Module energy ", energy.size())

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        #print("PAM_Module energy_new ", energy_new.size())

        attention = self.softmax(energy_new)

        convX = self.conv2dsame2(x)

        proj_value = sameConvX.view(m_batchsize, -1, width * height)
        #print("PAM_Module proj_value ", proj_value.size())

        attenPermute = attention.permute(0, 2, 1)
        #print("PAM_Module attenPermute ", attenPermute.size())
        out = torch.bmm(proj_value, attenPermute)
        #print("PAM_Module out1 ", out.size())
        out = out.view(m_batchsize, C, height, width)
        #print("PAM_Module out2 ", out.size())

        mainX = self.conv2dsame(x)
        out = self.gamma * out + sameConvX   #+ (1 - self.gamma) * sameConvX
        #print("PAM_Module output ", out.size())
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv2dsame = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(in_dim),
                nn.ReLU())

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
        #print("CAM_Module input ", x.size())
        m_batchsize, C, height, width = x.size()

        sameConvX = self.conv2dsame(x)

        proj_query = sameConvX.view(m_batchsize, C, -1)
        #print("CAM_Module proj_query ", proj_query.size())
        proj_key = sameConvX.view(m_batchsize, C, -1).permute(0, 2, 1)
        #print("CAM_Module proj_key ", proj_key.size())

        energy = torch.bmm(proj_query, proj_key)
        #print("CAM_Module energy ", energy.size())

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        #print("CAM_Module energy_new ", energy_new.size())

        attention = self.softmax(energy_new)

        proj_value = sameConvX.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + sameConvX  #+ (1 - self.gamma) * sameConvX
        #print("CAM_Module output ", out.size())
        return out


class PAM_CAM_Layer(nn.Module):
    """
    Helper Function for PAM and CAM attention

    Parameters:
    ----------
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """

    def __init__(self, in_ch, use_pam=True):
        super(PAM_CAM_Layer, self).__init__()

        self.attnIn = nn.Sequential(
             nn.Conv2d(in_ch * 2, in_ch * 2, kernel_size=3, padding=1, stride=1),
             nn.BatchNorm2d(in_ch * 2),
             nn.PReLU())

        self.attn = PAM_Module(2 * in_ch) if use_pam else CAM_Module(2 * in_ch)

        self.attnOut = nn.Sequential(
             nn.Conv2d(2 * in_ch, 2 * in_ch, kernel_size=3, padding=1, stride=1),
             nn.BatchNorm2d(2 * in_ch),
             nn.PReLU()
        )

    def forward(self, x):
        #print("PAM_CAM_Layer input ", x.size())
        x = self.attnIn(x)
        #print("PAM_CAM_Layer attnIn ", x.size())
        x = self.attn(x)
        #print("PAM_CAM_Layer attn ", x.size())
        out = self.attnOut(x)
        #print("PAM_CAM_Layer output ", out.size())

        return out


if __name__ == '__main__':
    #print("start to test attention modules")
    import torch as t
    inputX = t.randn(2, 64, 112, 112)

    camModule = CAM_Module(64)
    camModule.eval()
    resCam = camModule(inputX)
    #print("res CAM ", resCam.size())

    pamModule = PAM_Module(64)
    pamModule.eval()

    resPam = pamModule(inputX)
    print("res PAM ", resPam.size())