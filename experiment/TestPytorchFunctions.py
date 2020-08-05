import torch.nn as nn
import torch


m = nn.Conv2d(3, 32, 3, stride=1)

inputConv = torch.randn(200, 3, 50, 100)
outputConv = m(inputConv)

print("outputConv size ", outputConv.size())

