import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def dilate(img, outputImage):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img =img.astype("uint16")
    print("img ", img.shape)
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    dst = cv2.dilate(binary, kernel,  borderType=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.imwrite(outputImage, dst)
    print("save results to ", outputImage)

def erode(img, outputImage):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img =img.astype("uint8")
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.erode(binary, kernel)
    cv2.imwrite(outputImage, dst)
    print("save results to ", outputImage)

def iterateImagePath(inputPath, outputPath):
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    if not os.path.exists(outputPath + "/dilate/"):
        os.mkdir(outputPath + "/dilate/")
    if not os.path.exists(outputPath + "/erode/"):
        os.mkdir(outputPath + "/erode/")

    images = os.listdir(inputPath)
    for img_name in images:
        img = cv2.imread(inputPath + "/" + img_name)
        dilate(img, outputPath + "/dilate/" + img_name)
        erode(img, outputPath + "/erode/" + img_name)

if __name__ == "__main__":
    print("Start to do image filtering with opencv")

    inputPath = "E:/ubuntushare/data/warehousetools01/segmentopencv/filter/medianblur/"
    #inputPath = "E:/ubuntushare/data/warehousetools01/original/"
    outputPath = "E:/ubuntushare/data/warehousetools01/segmentopencv/dilateerode/"

    iterateImagePath(inputPath, outputPath)
