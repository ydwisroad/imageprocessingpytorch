import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b,c,h,w = x.size()
        y = self.avgpool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)

if __name__ == '__main__':

    import torch
    
    module = SELayer(64)

    testx1 = torch.randn(8, 64, 160, 160)
    testx2 = torch.randn(8, 64, 160, 160)
    print("input shape ", testx1.shape)
    output = module(testx1)
    print("output shape ", output.shape)