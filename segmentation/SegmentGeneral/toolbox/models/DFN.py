import torch
import torch.nn as nn
from torchvision import models

resnet101 = models.resnet101(pretrained=False)


class CAB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # high, low
        x = torch.cat([x1, x2], dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res


class RRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)


class DFN(nn.Module):
    def __init__(self, num_class=21):
        super(DFN, self).__init__()
        self.num_class = num_class
        self.layer0 = nn.Sequential(resnet101.conv1, resnet101.bn1, resnet101.relu)
        self.layer1 = nn.Sequential(resnet101.maxpool, resnet101.layer1)
        self.layer2 = resnet101.layer2
        self.layer3 = resnet101.layer3
        self.layer4 = resnet101.layer4

        # this is for smooth network
        self.out_conv = nn.Conv2d(2048, self.num_class, kernel_size=1, stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cab1 = CAB(self.num_class*2, self.num_class)
        self.cab2 = CAB(self.num_class*2, self.num_class)
        self.cab3 = CAB(self.num_class*2, self.num_class)
        self.cab4 = CAB(self.num_class*2, self.num_class)

        self.rrb_d_1 = RRB(256, self.num_class)
        self.rrb_d_2 = RRB(512, self.num_class)
        self.rrb_d_3 = RRB(1024, self.num_class)
        self.rrb_d_4 = RRB(2048, self.num_class)

        self.upsample = nn.Upsample(scale_factor=2,mode="bilinear")
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.upsample_8 = nn.Upsample(scale_factor=8, mode="bilinear")

        self.rrb_u_4 = RRB(self.num_class,self.num_class)
        self.rrb_u_3 = RRB(self.num_class,self.num_class)
        self.rrb_u_2 = RRB(self.num_class,self.num_class)
        self.rrb_u_1 = RRB(self.num_class,self.num_class)

        # this is for boarder net work
        self.rrb_db_1 = RRB(256, self.num_class)
        self.rrb_db_2 = RRB(512, self.num_class)
        self.rrb_db_3 = RRB(1024, self.num_class)
        self.rrb_db_4 = RRB(2048, self.num_class)

        self.rrb_trans_1 = RRB(self.num_class,self.num_class)
        self.rrb_trans_2 = RRB(self.num_class,self.num_class)
        self.rrb_trans_3 = RRB(self.num_class,self.num_class)

    def forward(self, x):
        f0 = self.layer0(x)  # 256, 256, 64
        f1 = self.layer1(f0)  # 128, 128, 256
        f2 = self.layer2(f1)  # 64, 64, 512
        f3 = self.layer3(f2)  # 32, 32, 1024
        f4 = self.layer4(f3)  # 16, 16, 2048

        # for border network
        res1 = self.rrb_db_1(f1)
        res1 = self.rrb_trans_1(res1 + self.upsample(self.rrb_db_2(f2)))
        res1 = self.rrb_trans_2(res1 + self.upsample_4(self.rrb_db_3(f3)))
        res1 = self.rrb_trans_3(res1 + self.upsample_8(self.rrb_db_4(f4)))      # 128, 128, 21

        # for smooth network
        res2 = self.out_conv(f4)    # 16, 16, 21
        res2 = self.global_pool(res2)  #
        res2 = nn.Upsample(size=f4.size()[2:],mode="nearest")(res2)     # 16, 16, 21

        f4 = self.rrb_d_4(f4)
        res2 = self.cab4([res2, f4])
        res2 = self.rrb_u_4(res2)

        f3 = self.rrb_d_3(f3)
        res2 = self.cab3([self.upsample(res2), f3])
        res2 =self.rrb_u_3(res2)

        f2 = self.rrb_d_2(f2)
        res2 = self.cab2([self.upsample(res2), f2])
        res2 =self.rrb_u_2(res2)

        f1 = self.rrb_d_1(f1)
        res2 = self.cab1([self.upsample(res2), f1])
        res2 = self.rrb_u_1(res2)

        return res1, res2


if __name__ == '__main__':
    import torch as t
    model = DFN(21)
    model.eval()
    image = t.randn(1, 3, 512, 512)
    res1, res2 = model(image)
    print(res1.size(), res2.size())