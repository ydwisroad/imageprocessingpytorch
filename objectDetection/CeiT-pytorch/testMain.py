
import torch
from ceit import CeiT

img = torch.ones([1, 3, 224, 224])

model = CeiT(image_size = 224, patch_size = 4, num_classes = 100)
out = model(img)

print("Shape of out :", out.shape)      # [B, num_classes]

model = CeiT(image_size = 224, patch_size = 4, num_classes = 100, with_lca = True)
out = model(img)

print("Shape of out :", out.shape)      # [B, num_classes]



if __name__ == "__main__":
    print("This is the start of main program")