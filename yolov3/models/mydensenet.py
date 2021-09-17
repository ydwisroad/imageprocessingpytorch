# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

#####This is the test of Densenet blocks ##########################
###################################################################

class DenseLayer1(nn.Module):
    def __init__(self, num_input_features,bn_size, growth_rate):
        super(DenseLayer1, self).__init__()

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, bn_size * growth_rate,
                      kernel_size=1, stride=1, bias=False)
        )

        self.block2 = nn.Sequential(
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x
    
class DenseBlock1(nn.Module):
    def __init__(self, num_input_features,bn_size, growth_rate):
        super(DenseBlock1, self).__init__()
        self.num_input_features = num_input_features
        self.bn_size = bn_size
        self.growth_rate = growth_rate

    def forward(self, x):
        denseLayer1 = DenseLayer1(self.num_input_features, self.bn_size, self.growth_rate)
        block1Out = denseLayer1(x)
        block2In = torch.cat((block1Out, x), 1)
        print("block2In size ", block2In.size())

        denseLayer2 = DenseLayer1(self.num_input_features+self.growth_rate,
                                  self.bn_size, self.growth_rate)
        block2Out = denseLayer2(block2In)
        block3In = torch.cat((block1Out, block2Out,x), 1)
        print("block3In size ", block3In.size())

        denseLayer3 = DenseLayer1(self.num_input_features+self.growth_rate*2,
                                  self.bn_size, self.growth_rate)
        block3Out = denseLayer3(block3In)
        block4In = torch.cat((block1Out, block2Out,block3Out , x), 1)
        print("block4In size ", block4In.size())

        denseLayer4 = DenseLayer1(self.num_input_features+self.growth_rate * 3,
                                  self.bn_size, self.growth_rate)
        block4Out = denseLayer4(block4In)
        block5In = torch.cat((block1Out, block2Out,block3Out , block4Out , x), 1)
        print("block5In size ", block5In.size())

        return block5In

class TransitionLayer1(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(TransitionLayer1, self).__init__()

        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features,
                      kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.transition(x)

        return x

class MyDenseNet(nn.Module):
    def __init__(self, num_layers = 4, num_input_features =128,
                 num_original = 32, num_output_features = 1024):
        super(MyDenseNet, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.num_original = num_original

    def forward(self, x):  #x： 4, 128, 160, 160
        denseBlock1 = DenseBlock1(self.num_input_features, 4, self.num_original)
        denseBlockOut1 = denseBlock1(x)
        #print("denseBlockOut1 size ", denseBlockOut1.size())

        transitionLayer1 = TransitionLayer1(self.num_input_features * 2, self.num_input_features * 2)
        transitionLayer1Out = transitionLayer1(denseBlockOut1)
        #print("transitionLayer1Out size ", transitionLayer1Out.size())

        denseBlock2 = DenseBlock1(self.num_input_features * 2, 4, self.num_original * 2)
        denseBlockOut2 = denseBlock2(transitionLayer1Out)
        #print("denseBlockOut2 size ", denseBlockOut2.size())

        transitionLayer2 = TransitionLayer1(self.num_input_features * 4, self.num_input_features * 4)
        transitionLayer2Out = transitionLayer2(denseBlockOut2)
        #print("transitionLayer2Out size ", transitionLayer2Out.size())

        denseBlock3 = DenseBlock1(self.num_input_features * 4, 4, self.num_input_features)
        denseBlockOut3 = denseBlock3(transitionLayer2Out)
        #print("denseBlockOut3 size ", denseBlockOut3.size())

        transitionLayer3 = TransitionLayer1(self.num_input_features * 8, self.num_output_features)
        transitionLayer3Out = transitionLayer3(denseBlockOut3)
        #print("transitionLayer3Out size ", transitionLayer3Out.size())

        return transitionLayer1Out, transitionLayer2Out, transitionLayer3Out

if __name__ == "__main__":
    #denseTest
    denseInputs = torch.randn(4, 128, 160, 160)

    myDenseNet1 = MyDenseNet(num_layers = 4, num_input_features =128,
                 num_original = 32, num_output_features = 1024)
    myDenseOutput1,myDenseOutput2, myDenseOutput3  = myDenseNet1(denseInputs)
    print("myDenseOutput out " , myDenseOutput1.size() , " ", myDenseOutput2.size(), " ", myDenseOutput3.size())

























