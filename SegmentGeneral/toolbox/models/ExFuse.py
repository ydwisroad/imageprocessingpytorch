import torch
from torch import nn
from torchvision import models


class _GlobalConvModule(nn.Module):
    def __init__(self, in_channels, num_class, k=15):
        super(_GlobalConvModule, self).__init__()

        pad = (k-1) // 2

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, num_class, kernel_size=(1, k), padding=(0, pad), bias=False),
                                   nn.Conv2d(num_class, num_class, kernel_size=(k, 1), padding=(pad, 0), bias=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, num_class, kernel_size=(k, 1), padding=(pad, 0), bias=False),
                                   nn.Conv2d(num_class, num_class, kernel_size=(1, k), padding=(0, pad), bias=False))

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)

        assert x1.shape == x2.shape

        return x1 + x2


class SEB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x1, x2 = x
        return x1 * self.upsample(self.conv(x2))


class ECRE(nn.Module):
    def __init__(self, in_c, up_scale=2):
        super(ECRE, self).__init__()
        self.ecre = nn.Sequential(nn.Conv2d(in_c, in_c*(up_scale**2), kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(in_c*(up_scale**2)),
                                  nn.PixelShuffle(up_scale))

    def forward(self, input_):
        return self.ecre(input_)


class GCNFuse(nn.Module):
    def __init__(self, n_classes=21):
        super(GCNFuse, self).__init__()
        self.n_classes = n_classes
        self.resnet_features = models.resnet101(pretrained=False)
        self.layer0 = nn.Sequential(self.resnet_features.conv1, self.resnet_features.bn1, self.resnet_features.relu)
        self.layer1 = nn.Sequential(self.resnet_features.maxpool, self.resnet_features.layer1)
        self.layer2 = self.resnet_features.layer2
        self.layer3 = self.resnet_features.layer3
        self.layer4 = self.resnet_features.layer4

        self.gcm4 = _GlobalConvModule(2048, n_classes)
        self.gcm3 = _GlobalConvModule(1024, n_classes)
        self.gcm2 = _GlobalConvModule(512, n_classes)
        self.gcm1 = _GlobalConvModule(256, n_classes)

        self.deconv3 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv0 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)

        self.ecre = ECRE(n_classes)

        self.seb3 = SEB(2048, 1024)
        self.seb2 = SEB(3072, 512)
        self.seb1 = SEB(3584, 256)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, x):
        f0 = self.layer0(x)  # 256, 256, 64
        f1 = self.layer1(f0); #print(f1.size())  # 128, 128, 256
        f2 = self.layer2(f1); #print(f2.size())  # 64, 64, 512
        f3 = self.layer3(f2); #print(f3.size())  # 32, 32, 1024
        f4 = self.layer4(f3); #print(f4.size())  # 16, 16, 2048

        gcm4 = self.gcm4(f4)   # 16, 16, 21
        out4 = self.ecre(gcm4)  # 32, 32, 21

        seb3 = self.seb3([f3, f4])  # 32, 32, 1024
        gcm3 = self.gcm3(seb3)  # 32, 32, 21

        seb2 = self.seb2([f2, torch.cat([f3, self.upsample2(f4)], dim=1)])  # 64, 64, 512
        gcm2 = self.gcm2(seb2)  # 64, 64, 21

        seb1 = self.seb1([f1, torch.cat([f2, self.upsample2(f3), self.upsample4(f4)], dim=1)])  # 128, 128, 256
        gcm1 = self.gcm1(seb1)  # 128, 128, 21

        y = self.deconv3(gcm3 + out4)
        y = self.deconv2(gcm2 + y)
        y = self.deconv1(gcm1 + y)
        y = self.deconv0(y)

        return y


if __name__ == '__main__':
    model = GCNFuse(21)
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    res1 = model(image)
    print('result:', res1.size())