import torch
import torch.nn as nn
from torchvision import transforms

import PIL.Image as Image
import torchvision.transforms

import torch
import os
from torchvision import datasets
import torchvision.transforms as transforms
import PIL as pil
import numpy as np
from torchvision.utils import save_image
from torch.utils import data


def load_data(root,save_root):
    transform = transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(448)])
    image_list = os.listdir(root)
    for image_file in image_list:
        file = root + '/' + image_file
        image = pil.Image.open(file)
        image = transform(image)
        image.save(save_root+'/'+image_file)
root =  "../../../carDetectionSimple/train/bus/"
save_root = "./"
os.makedirs(save_root,exist_ok=True)
load_data(root,save_root)
