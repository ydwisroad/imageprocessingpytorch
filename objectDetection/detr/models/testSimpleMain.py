import torch
from torchvision.models import resnet50
from DeformableConv import DeformableConv2d
from plugplay import *
from PEG import *

def testDeformableConv():
    print("start of test LeFF")
    testx = torch.randn(8, 32, 123, 25)
    print("input shape ", testx.shape)
    deformConv = DeformableConv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
    output = deformConv(testx)
    print("output shape ", output.shape)

def testProjAttention():
    testx = torch.randn(8, 3, 25, 25)
    print("input shape ", testx.shape)
    #s3_proj_kernel = 3 s3_kv_proj_stride = 2, s3_heads = 4, s3_depth = 10,
    projAttention = ProjAttention(dim = 3, proj_kernel = 3, kv_proj_stride = 2, heads = 4, dim_head = 64, dropout = 0.)
    print("proj attention created ")
    output = projAttention(testx)
    print("output shape ", output.shape)

def testLeFF():
    print("start of test LeFF")
    testx = torch.randn(8, 196, 3)
    print("input shape ", testx.shape)
    leFF = LeFF(dim = 3, scale = 4, depth_kernel = 3, h =14,w=14)
    print("leFF created ")
    output = leFF(testx)
    print("output shape ", output.shape)

def testPEG():
    print("start of test PEG")
    testx = torch.randn(8, 196, 3)
    print("input shape ", testx.shape)
    peg = PEG(3)
    output = peg(testx, 14, 14)
    print("output shape ", output.shape)


if __name__ == "__main__":
    print("This is the start of main program")
    testDeformableConv()
    print("=============================================")
    testProjAttention()
    print("=============================================")
    testLeFF()
    print("=============================================")
    testPEG()

    testX = torch.randn(400, 2, 256)  #=> 2, 256, 20,20
    testX = testX.permute(1, 2, 0)   #Now 2,256,400
    B, C, HW = testX.shape
    H = 20
    W = 20
    testX = testX.view(B, C, H, W)
    print("testX shape:", testX.shape)
    testX = testX.view(B, C, -1)
    print("testX shape:", testX.shape)


