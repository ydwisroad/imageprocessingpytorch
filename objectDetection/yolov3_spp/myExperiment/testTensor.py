import os

from torch import nn
import torch
import math

def testTorchArray():
    testx = torch.randn(4, 5, 3, dtype=torch.cfloat)
    print("testx shape ", testx.shape)

    zerosArr = torch.zeros([2, 4], dtype=torch.int32)
    print("zeros ", zerosArr)

    onesArr = torch.ones([3, 4], dtype=torch.float64)
    print("ones ", onesArr)

    #torch.arange
    rangeTensor1 = torch.arange(15)
    print("rangeTensor1 " , rangeTensor1)
    rangeTensor2 = torch.arange(3, 15, 2)
    print("rangeTensor2 ", rangeTensor2)

    #torch.meshgrid
    a = torch.tensor([1, 2, 3, 4])
    print(a)
    b = torch.tensor([5, 6, 7])
    print(b)
    x, y = torch.meshgrid(a, b)
    print("mesh out1", x)
    print("mesh out2", y)

    ny = 4
    nx = 6
    yv, xv = torch.meshgrid([torch.arange(ny),
                             torch.arange(nx)])
    # batch_size, na, grid_h, grid_w, wh
    newgrid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    print("newgrid size ", newgrid.size())
    print("newgrid ", newgrid)

    #torch.stack
    print("exp output", torch.exp(testx))

    print("sigmoid output ", torch.sigmoid(testx))

    #repeat
    #torch.exp
    #torch.sigmoid
    #



if __name__ == '__main__':
    print("This is the start of main program")

    testTorchArray()



