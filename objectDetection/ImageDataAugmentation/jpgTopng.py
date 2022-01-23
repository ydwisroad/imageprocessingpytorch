from PIL import Image
import os
from glob import glob
import shutil
import os
import cv2
import sys
import numpy as np


def convertJPGtoPNG(sourceJPG, destPNG):
    #im1 = Image.open(sourceJPG)
    #im1.save(destPNG)
    img = cv2.imread(sourceJPG)
    cv2.imwrite(destPNG,img)

if __name__ == "__main__":
    print("start of the program")

    sourcePath = "E:/ubuntushare/data/warehousetools/topredict/original_/"
    destPath = "E:/ubuntushare/data/warehousetools/topredict/original/"

    if not os.path.exists(destPath):
        os.makedirs(destPath)

    files = glob(sourcePath + "*.jpg")
    files = [i.replace("\\", "/").split("/")[-1].split(".jpg")[0] for i in files]

    for jpg_file_ in files:
        jpg_filename = sourcePath + jpg_file_ + ".jpg"
        png_filename = destPath + jpg_file_ + ".png"
        convertJPGtoPNG(jpg_filename,png_filename )
