import torch.nn as nn
import torch

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, inputDim=3, kernel_size=9, patch_size=2, n_classes=1000):
        super().__init__()
        self.conMixer = nn.Sequential(
            nn.Conv2d(inputDim, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)])

    def forward(self, x):
        return self.conMixer(x)

def ConvMixer1(dim, depth, inputDim=3, kernel_size=9, patch_size=2, n_classes=1000):
    return nn.Sequential(
            nn.Conv2d(inputDim, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)]
    )

#https://blog.csdn.net/qq_38863413/article/details/104108808
if __name__ == "__main__":
    #denseTest
    inputs = torch.randn(4, 3, 320, 320)
    convMixer = ConvMixer(3,1)

    convMixerOutput = convMixer(inputs)
    print("convMixer", convMixer)
    print("output shape" , convMixerOutput.size())

    inputsA = torch.randn(4, 64, 320, 320)
    patchConv = nn.Conv2d(64, 64, kernel_size=2, stride=2)
    outPatch = patchConv(inputsA)
    print("outPatch",   outPatch.size())

    depthConv = nn.Conv2d(64, 64, 9, groups=64, padding="same")
    outDepthConv = depthConv(inputsA)
    print("outDepthConv", outDepthConv.size())

    pointConv = nn.Conv2d(64, 64, kernel_size=1)
    pointConvOut = pointConv(inputsA)
    print("pointConvOut", pointConvOut.size())






