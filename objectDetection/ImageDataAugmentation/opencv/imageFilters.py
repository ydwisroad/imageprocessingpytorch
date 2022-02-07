import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def filterBlur(img, outputImage):
    img_mean = cv2.blur(img, (5, 5))
    cv2.imwrite(outputImage, img_mean)
    print("save results to ", outputImage)

def filterGaussian(img, outputImage):
    img_Guassian = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(outputImage, img_Guassian)
    print("save results to ", outputImage)

def filterMedianBlur(img, outputImage):
    img_median = cv2.medianBlur(img, 9)
    cv2.imwrite(outputImage, img_median)
    print("save results to ", outputImage)

def filterBlater(img, outputImage):
    img_bilater = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite(outputImage, img_bilater)
    print("save results to ", outputImage)

def iterateImagePath(inputPath, outputPath):
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    if not os.path.exists(outputPath + "/blur/"):
        os.mkdir(outputPath + "/blur/")
    if not os.path.exists(outputPath + "/gaussian/"):
        os.mkdir(outputPath + "/gaussian/")
    if not os.path.exists(outputPath + "/medianblur/"):
        os.mkdir(outputPath + "/medianblur/")
    if not os.path.exists(outputPath + "/blater/"):
        os.mkdir(outputPath + "/blater/")

    images = os.listdir(inputPath)
    for img_name in images:
        img = cv2.imread(inputPath + "/" + img_name)
        filterBlur(img, outputPath + "/blur/" + img_name)
        filterGaussian(img, outputPath + "/gaussian/" + img_name)
        filterMedianBlur(img, outputPath + "/medianblur/" + img_name)
        filterBlater(img, outputPath + "/blater/" + img_name)

if __name__ == "__main__":
    print("Start to do image filtering with opencv")

    inputPath = "E:/ubuntushare/data/warehousetools01/segmentopencv/scharr/"
    outputPath = "E:/ubuntushare/data/warehousetools01/segmentopencv/filter/"

    iterateImagePath(inputPath, outputPath)