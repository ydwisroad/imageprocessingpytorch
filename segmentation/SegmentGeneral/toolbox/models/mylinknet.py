import torch
import torch.nn as nn
import torchvision

import sys
import os
#sys.path.append(os.path.abspath("../blocks"))
from toolbox.blocks.attention import PAM_CAM_Layer
from toolbox.blocks.TripletAttention import TripletAttention

#from attention import (
#    CAM_Module,
#    PAM_Module,
#    semanticModule,
#    PAM_CAM_Layer,
#    MultiConv
#)

#https://blog.csdn.net/googler_offer/article/details/79521453
#https://blog.csdn.net/wchzh2015/article/details/93883771

class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
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

class MyResnet34(nn.Module):
    def __init__(self, n_classes=12):
        super(MyResnet34, self).__init__()
        resnet34base = torchvision.models.resnet34(pretrained=False)
        self.in_block = nn.Sequential(
            resnet34base.conv1,
            resnet34base.bn1,
            resnet34base.relu,
            resnet34base.maxpool
        )   #1, 64, 112, 112

        self.encoder1 = resnet34base.layer1
        self.encoder2 = resnet34base.layer2
        self.encoder3 = resnet34base.layer3
        self.encoder4 = resnet34base.layer4

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)

    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        #print("1.after in block ", x.size())
        # Encoder blocks
        e1 = self.encoder1(x)
        #print("2.after encoder1 ", e1.size())
        e2 = self.encoder2(e1)
        #print("3.after encoder2 ", e2.size())
        e3 = self.encoder3(e2)
        #print("4.after encoder3 ", e3.size())
        e4 = self.encoder4(e3)
        #print("5.after encoder4 ", e4.size())

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return y

class MyResnet34WithAttention(nn.Module):
    def __init__(self, n_classes=2):
        super(MyResnet34WithAttention, self).__init__()
        resnet34base = torchvision.models.resnet34(pretrained=False)
        self.in_block = nn.Sequential(
            resnet34base.conv1,
            resnet34base.bn1,
            resnet34base.relu,
            resnet34base.maxpool
        )   #1, 64, 112, 112

        self.encoder1 = resnet34base.layer1
        self.encoder2 = resnet34base.layer2
        self.encoder3 = resnet34base.layer3
        self.encoder4 = resnet34base.layer4

        self.pam_attention_1_1= PAM_CAM_Layer(32, True)
        self.cam_attention_1_1= PAM_CAM_Layer(32, False)

        self.pam_attention_1_2= PAM_CAM_Layer(64, True)
        self.cam_attention_1_2= PAM_CAM_Layer(64, False)

        self.pam_attention_1_3= PAM_CAM_Layer(128, True)
        self.cam_attention_1_3= PAM_CAM_Layer(128, False)

        self.pam_attention_1_4= PAM_CAM_Layer(256, True)
        self.cam_attention_1_4= PAM_CAM_Layer(256, False)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)

    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        #print("1.after in block ", x.size())
        # Encoder blocks
        e1 = self.encoder1(x)
        #print("2.after encoder1 ", e1.size())
        attn_pam1 = self.pam_attention_1_1(e1)
        #print("2.after e1 attention attn_pam1 ", attn_pam1.size())
        attn_cam1 = self.cam_attention_1_1(e1)
        #print("2.after e1 attention attn_cam1 ", attn_cam1.size())
        fusionattn1 = torch.cat((attn_pam1,attn_cam1),1)
        #print("fusion attn1" , fusionattn1.size())

        e2 = self.encoder2(e1)
        #print("3.after encoder2 ", e2.size())
        attn_pam2 = self.pam_attention_1_2(e2)
        #print("3.after e1 attention attn_pam2 ", attn_pam2.size())
        attn_cam2 = self.cam_attention_1_2(e2)
        #print("3.after e1 attention attn_cam2 ", attn_cam2.size())
        fusionattn2 = torch.cat((attn_pam2, attn_cam2), 1)
        #print("fusion attn2", fusionattn2.size())

        e3 = self.encoder3(e2)
        #print("4.after encoder3 ", e3.size())
        attn_pam3 = self.pam_attention_1_3(e3)
        #print("4.after e1 attention attn_pam3 ", attn_pam3.size())
        attn_cam3 = self.cam_attention_1_3(e3)
        #print("4.after e1 attention attn_cam3 ", attn_cam3.size())
        fusionattn3 = torch.cat((attn_pam3, attn_cam3), 1)
        #print("fusion attn3", fusionattn3.size())

        e4 = self.encoder4(e3)
        #print("5.after encoder4 ", e4.size())
        attn_pam4 = self.pam_attention_1_4(e4)
        #print("5.after e1 attention attn_pam4 ", attn_pam4.size())
        attn_cam4 = self.cam_attention_1_4(e4)
        #print("5.after e1 attention attn_cam4 ", attn_cam4.size())
        fusionattn4 = torch.cat((attn_pam4, attn_cam4), 1)
        #print("fusion attn4", fusionattn4.size())

        # Decoder blocks
        d4 = e3 + self.decoder4(e4 + fusionattn4) + fusionattn3
        d3 = e2 + self.decoder3(d4) + fusionattn2
        d2 = e1 + self.decoder2(d3) + fusionattn1
        d1 = x + self.decoder1(d2)

        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return y
class MyResnet34WithTripletAttention(nn.Module):
    def __init__(self, n_classes=12):
        super(MyResnet34WithTripletAttention, self).__init__()
        resnet34base = torchvision.models.resnet34(pretrained=False)
        self.in_block = nn.Sequential(
            resnet34base.conv1,
            resnet34base.bn1,
            resnet34base.relu,
            resnet34base.maxpool
        )   #1, 64, 112, 112

        self.encoder1 = resnet34base.layer1
        self.encoder2 = resnet34base.layer2
        self.encoder3 = resnet34base.layer3
        self.encoder4 = resnet34base.layer4

        self.tripletAttention = TripletAttention()

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)

    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        #print("1.after in block ", x.size())
        # Encoder blocks
        e1 = self.encoder1(x)
        #print("2.after encoder1 ", e1.size())
        e1Attention = self.tripletAttention(e1)
        #print("2.triplet attention size ", e1Attention.size())

        e2 = self.encoder2(e1)
        e2Attention = self.tripletAttention(e2)
        #print("3.after encoder2 ", e2.size())
        e3 = self.encoder3(e2)
        e3Attention = self.tripletAttention(e3)
        #print("4.after encoder3 ", e3.size())
        e4 = self.encoder4(e3)
        e4Attention = self.tripletAttention(e4)
        #print("5.after encoder4 ", e4.size())

        # Decoder blocks
        d4 = e3 + self.decoder4(e4+e4Attention) + e3Attention
        d3 = e2 + self.decoder3(d4) + e2Attention
        d2 = e1 + self.decoder2(d3) + e1Attention
        d1 = x + self.decoder1(d2)

        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return y

class MyResnet50(nn.Module):
    def __init__(self, n_classes=12):
        super(MyResnet50, self).__init__()
        resnet50base = torchvision.models.resnet50(pretrained=False)
        self.in_block = nn.Sequential(
            resnet50base.conv1,
            resnet50base.bn1,
            resnet50base.relu,
            resnet50base.maxpool
        )   #1, 64, 112, 112

        self.encoder1 = resnet50base.layer1
        self.encoder2 = resnet50base.layer2
        self.encoder3 = resnet50base.layer3
        self.encoder4 = resnet50base.layer4

        self.decoder1 = Decoder(256, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(512, 256, 3, 2, 1, 1)
        self.decoder3 = Decoder(1024, 512, 3, 2, 1, 1)
        self.decoder4 = Decoder(2048, 1024, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)

    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        #print("1.after in block ", x.size())
        # Encoder blocks
        e1 = self.encoder1(x)
        #print("2.after encoder1 ", e1.size())
        e2 = self.encoder2(e1)
        #print("3.after encoder2 ", e2.size())
        e3 = self.encoder3(e2)
        #print("4.after encoder3 ", e3.size())
        e4 = self.encoder4(e3)
        #print("5.after encoder4 ", e4.size())

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        #print("6.after decoder4 ", self.decoder4(e4).size())
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return y

if __name__ == '__main__':
    #resnet34base = torchvision.models.resnet34(pretrained=False)

    #print(resnet34base.conv1)
    #print(resnet34base.bn1)
    #print(resnet34base.relu)
    #print(resnet34base.relu)

    #print(resnet34base.layer1)
    #print(resnet34base.layer2)
    #print(resnet34base.layer3)
    #print(resnet34base.layer4)

    inputs = torch.randn(1, 3, 448, 448)
    resnet34Model = MyResnet34WithAttention(n_classes=12)

    out = resnet34Model(inputs)
    print(out.size())

    #myresnet50 = MyResnet50(n_classes=12)

    #out = myresnet50(inputs)
    #print(out.size())









