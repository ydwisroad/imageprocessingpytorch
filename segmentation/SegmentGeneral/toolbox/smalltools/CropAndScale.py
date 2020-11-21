import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import cv2

def img_resize(data, resize_size):
    data = ff.resize(data, resize_size, interpolation=2)
    return data

def center_crop(data, crop_size):
    data = ff.center_crop(data, crop_size)
    return data

def convertAllFilesInFolder(inFolder, outFolder, crop_size):
    for eachFile in os.listdir(inFolder):
        if(eachFile.find('.png')!=-1 or eachFile.find('.jpg')!=-1 ):
            img = Image.open(inFolder + eachFile)
            img = img.convert('RGB')
            minSize = min(img.size[0], img.size[1])
            img = center_crop(img, [minSize, minSize])
            img = img_resize(img,  crop_size)
            img.save(outFolder + eachFile)

convertAllFilesInFolder('../../data/YdCrackDataset/full/train/', '../../data/YdCrackDataset/full/train2/', [448,448])
convertAllFilesInFolder('../../data/YdCrackDataset/full/val/', '../../data/YdCrackDataset/full/val2/', [448,448])
