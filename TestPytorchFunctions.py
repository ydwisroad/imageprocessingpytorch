import torch.nn as nn
import torch

#https://blog.csdn.net/disanda/article/details/105762054?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param
#Input (N, Cin, Hin, Win)  Output: (N, Cout, Hout, Wout)
#Conv2d (in_channels, out_channels, kernel_size, optional: stride=1, padding=0, dilation=1, groups=1)

m = nn.Conv2d(3, 32, 3, stride=1)

inputConv = torch.randn(200, 3, 50, 100)
outputConv = m(inputConv)
print("inputConv size ", inputConv.size())
print("outputConv size ", outputConv.size())

#torch.nn.ConvTranspose2d
#Input (N, Cin, Hin, Win)  Output: (N, Cout, Hout, Wout)
#ConvTranspose2d(in_channels,out_channels,kernel_size,optional:stride=1, padding=0,output_padding=0,groups=1,dilation=1)
m = nn.ConvTranspose2d(16, 33, 3, stride=1)
input = torch.randn(20, 16, 50, 100)
output = m(input)
print("inputConvTranspose size ", input.size())
print("outputConvTranspos size ", output.size())

#UpsamplingNearest2d
#Input: (N, Cin, Hin, Win)  output: (N, Cout, Hout, Wout)
input = torch.randn(3, 2, 2, 2)
print(input)
m = nn.UpsamplingNearest2d(scale_factor=2)
output = m(input)
print(output)

input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # align_corners=False
outUpSample = m(input)
print(outUpSample)

#MaxUnpool2d
#kernel_size， stride
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)
output, indices = pool(input)
outunpool = unpool(output, indices)
print(outunpool)

print("done")















