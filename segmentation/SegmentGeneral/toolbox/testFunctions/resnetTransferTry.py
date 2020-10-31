import torch
import torch.nn as nn
import torchvision

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

class MyTestnet(nn.Module):
    def __init__(self, n_classes=12):
        super(MyTestnet, self).__init__()
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

        print("1.after in block ", x.size())
        # Encoder blocks
        e1 = self.encoder1(x)
        print("2.after encoder1 ", e1.size())
        e2 = self.encoder2(e1)
        print("3.after encoder2 ", e2.size())
        e3 = self.encoder3(e2)
        print("4.after encoder3 ", e3.size())
        e4 = self.encoder4(e3)
        print("5.after encoder4 ", e4.size())

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        y = self.tp_conv1(d1)
        y = self.conv2(y)

        return y


if __name__ == '__main__':
    resnet34base = torchvision.models.resnet34(pretrained=False)

    print(resnet34base.conv1)
    print(resnet34base.bn1)
    print(resnet34base.relu)
    print(resnet34base.relu)

    print(resnet34base.layer1)
    print(resnet34base.layer2)
    print(resnet34base.layer3)
    print(resnet34base.layer4)

    inputs = torch.randn(1, 3, 448, 448)
    model = MyTestnet(n_classes=12)

    out = model(inputs)
    print(out.size())









