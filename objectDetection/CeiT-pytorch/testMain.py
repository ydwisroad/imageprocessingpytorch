import torch
from ceit import CeiT
from module import LeFF

def testCeit():
    img = torch.ones([1, 3, 224, 224])

    model = CeiT(image_size = 224, patch_size = 4, num_classes = 100)
    out = model(img)

    print("Shape of out :", out.shape)      # [B, num_classes]

    model = CeiT(image_size = 224, patch_size = 4, num_classes = 100, with_lca = True)
    out = model(img)

    print("Shape of out :", out.shape)      # [B, num_classes]


def testLeFF():
    print("start of test LeFF")
    testx = torch.randn(8, 196, 3)
    print("input shape ", testx.shape)
    leFF = LeFF(dim = 3, scale = 4, depth_kernel = 3, h =14,w=14)
    print("leFF created ")
    output = leFF(testx)
    print("output shape ", output.shape)


if __name__ == "__main__":
    print("This is the start of main program")
    testCeit()
    #testLeFF()