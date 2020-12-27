import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import cv2


def transRedToWhite(img):
    sp = img.size
    width = sp[0]
    height = sp[1]
    for yh in range(height):
        for xw in range(width):
            dot = (xw, yh)
            color_d = img.getpixel(dot)
            #print(color_d)
            #print(color_d[0])
            #if (color_d[0] == 128):
            #    color_d = (255, 255, 255, 255)
            #    img.putpixel(dot, color_d)
    return img

#img=Image.open('../../data/ydCrackSimple/test/5042.png')
#img=transRedToWhite(img)
#img.save('../../data/ydCrackSimple/test/504White.png')


def transparence2white(img):
    sp = img.shape
    width = sp[0]
    height = sp[1]
    for yh in range(height):
        for xw in range(width):
            color_d = img[xw, yh]
            #print(color_d)
            if (color_d[0] == 128 or color_d[1] == 128 or color_d[2] == 128):
                img[xw, yh] = [255, 255, 255] #, 255]
    return img

def convertAllFilesInFolder(inFolder, outFolder):
    for eachFile in os.listdir(inFolder):
        if(eachFile.find('.png')!=-1):
            filePath = inFolder + eachFile
            print("filePath", filePath)
            img = cv2.imread(filePath, -1)
            img = transparence2white(img)
            cv2.imwrite(outFolder + '/' + eachFile, img)
            print(eachFile)

convertAllFilesInFolder('F:/segmentRunResults/Final/CrackSilver/gthmanual/SegmentationClassPNG/', 'F:/segmentRunResults/Final/CrackSilver/gthmanual/gth')

#img = cv2.imread('../../data/ydCrackSimple/test/504.png', -1)
#img = transparence2white(img)
#cv2.imwrite('../../data/ydCrackSimple/test/504white.png',img)

