import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

class LNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, padding=0, bias=False):
        super(LNBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, 3, stride, bias=False),
                            nn.BatchNorm2d(out_planes))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class LNEncoder(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, padding=0, bias=False):
        super(LNEncoder, self).__init__()
        self.block1 = LNBasicBlock(in_planes, out_planes, stride, padding, bias)
        self.block2 = LNBasicBlock(out_planes, out_planes, 1, padding, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x

class LNDecoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False):
        super(LNDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)

        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)

        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class MyLinkUnet(nn.Module):
    def __init__(self, n_classes = 2):
        super(MyLinkUnet, self).__init__()
        base = models.resnet18(pretrained=False)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = LNDecoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = LNDecoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = LNDecoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = LNDecoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))

        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)


    def forward(self, x):
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return x

class EnInitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, relu=True):
        super().__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_branch = nn.Conv2d( in_channels, out_channels - 3, kernel_size=3, stride=2,
                                        padding=1, bias=bias)
        # Extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)
        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()



    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concatenate branches
        out = torch.cat((main, ext), 1)

        # Apply batch normalization
        out = self.batch_norm(out)
        return  self.out_activation(out)


class EnRegularBottleneck(nn.Module):
    def __init__(self,channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super(EnRegularBottleneck).__init__()

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels,internal_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())

        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1), stride=1,
                          padding=(padding, 0), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(internal_channels,internal_channels,kernel_size=(1, kernel_size),
                    stride=1,padding=(0, padding), dilation=dilation,bias=bias),
                nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size,
                         stride=1, padding=padding, dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels,channels, kernel_size=1,  stride=1, bias=bias),
            nn.BatchNorm2d(channels),
            activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)

class EnDownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, return_indices=False,
                 dropout_prob=0, bias=False, relu=True):
        super(EnDownsamplingBottleneck).__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(2, stride=2, return_indices=return_indices)

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(nn.Conv2d( in_channels, internal_channels, kernel_size=2, stride=2, bias=bias),
                         nn.BatchNorm2d(internal_channels), activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d( internal_channels, internal_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d( internal_channels, out_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out), max_indices


class EnUpsamplingBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0, bias=False, relu=True):
        super(EnUpsamplingBottleneck).__init__()

        internal_channels = in_channels // internal_ratio
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
                           nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
                           nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation())

        # Transposed convolution
        self.ext_tconv1 = nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=2, stride=2, bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        # 1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices, output_size=output_size)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)



class MyEnet(nn.Module):
    def __init__(self, n_classes = 2, encoder_relu=False, decoder_relu=True):
        super(MyEnet, self).__init__()
        self.initial_block = EnInitialBlock(3, 16, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = EnDownsamplingBottleneck( 16, 64, return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = EnRegularBottleneck( 64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = EnRegularBottleneck( 64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = EnRegularBottleneck(  64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = EnRegularBottleneck( 64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = EnDownsamplingBottleneck(64,128, return_indices=True, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = EnRegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = EnRegularBottleneck( 128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = EnRegularBottleneck( 128, kernel_size=5, padding=2, asymmetric=True,
                                                dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_4 = EnRegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = EnRegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = EnRegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = EnRegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_8 = EnRegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder
        self.regular3_0 = EnRegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = EnRegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = EnRegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_3 = EnRegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = EnRegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = EnRegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = EnRegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_7 = EnRegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 4 - Decoder
        self.upsample4_0 = EnUpsamplingBottleneck(
            128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = EnRegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = EnRegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = EnUpsamplingBottleneck(
            64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = EnRegularBottleneck(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(
            16,
            n_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

    def forward(self, x):
        # Initial block
        input_size = x.size()
        x = self.initial_block(x)

        # Stage 1-Encoder
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage2 -Encoder
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage3-Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Stage4 -Decoder
        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage5-Decoder
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x)
        x = self.transposed_conv(x, output_size=input_size)

        return x

if __name__ == "__main__":
    print("ENet Linknet started")

    inputX = torch.randn(4, 3, 448, 448)

    lnBasicBlock = LNBasicBlock(3,16, 2)
    #outLnBasicBlock = lnBasicBlock(inputX)
    #print("Basic Block output size ", outLnBasicBlock.size())

    ResnetBase = models.resnet50(pretrained=False)

    myin_block = nn.Sequential(
        ResnetBase.conv1,
        ResnetBase.bn1,
        ResnetBase.relu,
        ResnetBase.maxpool
    )
    print("conv1 ",ResnetBase.conv1)
    print("bn1 ",ResnetBase.bn1)
    print("relu ",ResnetBase.relu)
    print("maxpool ",ResnetBase.maxpool)

    print("layer1 ", ResnetBase.layer1)
    print("layer2 ", ResnetBase.layer2)
    print("layer3 ", ResnetBase.layer3)
    print("layer4 ", ResnetBase.layer4)

    myLinkNet = MyLinkUnet(2)
    outputLinkNet = myLinkNet(inputX)
    print("outputLinkNet size ", outputLinkNet.size())

    enInitialBlk = EnInitialBlock(3, 64)
    outputEnInitialBlk = enInitialBlk(inputX)
    print("output blk size " , outputEnInitialBlk.size())

    myEnet = MyEnet()
    outputMyENet = myEnet(inputX)
    print("my ENet out put ", outputMyENet.size())







