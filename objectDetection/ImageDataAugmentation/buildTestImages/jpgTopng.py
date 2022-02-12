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

def convertJPGtoPNGGrey(sourceJPG, destPNG):
    img = cv2.imread(sourceJPG, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(destPNG, img)

def folderJPGtoPNG(sourcePath, destPath, grey=False):
    if not os.path.exists(destPath):
        os.makedirs(destPath)

    files = glob(sourcePath + "*.jpg")
    files = [i.replace("\\", "/").split("/")[-1].split(".jpg")[0] for i in files]

    for jpg_file_ in files:
        jpg_filename = sourcePath + jpg_file_ + ".jpg"
        png_filename = destPath + jpg_file_ + ".png"
        if not grey:
            convertJPGtoPNG(jpg_filename,png_filename )
        else:
            convertJPGtoPNGGrey(jpg_filename, png_filename)

def jpgtopngfullfolder(sourcePath, destPath):
    print("This is the start of jpgtopngfull folder")
    if os.path.exists(destPath):
        os.rmdir(destPath)
    shutil.copytree(sourcePath, destPath)

    files = glob(destPath + "/**/*.jpg", recursive=True)
    files = [i.replace("\\", "/") for i in files]

    for file_ in files:
        print("each file ", file_)
        destFileName = file_[0:file_.rfind(".")]+".png"
        convertJPGtoPNG(file_, destFileName)
        os.remove(file_)

if __name__ == "__main__":
    print("start of the program")

    sourcePath = "E:/ubuntushare/data/warehousetools01/original/"
    destPath = "E:/ubuntushare/data/warehousetools01/originalgrey/"
    folderJPGtoPNG(sourcePath, destPath, True)

