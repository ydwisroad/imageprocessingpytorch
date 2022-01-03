
from coco import *

import os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
from PIL import Image

def copyFilesInFolder(folderSrc, folderDest):
    for fileName in os.listdir(folderSrc):
        fullNamePath = folderSrc + "/" + fileName
        print("selected ", fullNamePath)
        shutil.copy(fullNamePath, folderDest + "/" + fileName)

def changeCocoFormat(inputFolder, outputFolder):
    print("This is the start of change cocoFormat")
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    if not os.path.exists(outputFolder + "/images"):
        os.mkdir(outputFolder+ "/images")
    if not os.path.exists(outputFolder + "/images/train"):
        os.mkdir(outputFolder+ "/images/train")
    if not os.path.exists(outputFolder + "/images/val"):
        os.mkdir(outputFolder+ "/images/val")
    if not os.path.exists(outputFolder + "/labels"):
        os.mkdir(outputFolder+ "/labels")
    if not os.path.exists(outputFolder + "/labels/train"):
        os.mkdir(outputFolder+ "/labels/train")
    if not os.path.exists(outputFolder + "/labels/val"):
        os.mkdir(outputFolder+ "/labels/val")

    copyFilesInFolder(inputFolder + "/train/images", outputFolder +"/images/train")
    copyFilesInFolder(inputFolder + "/train/labels", outputFolder + "/labels/train")
    copyFilesInFolder(inputFolder + "/val/images", outputFolder +"/images/val")
    copyFilesInFolder(inputFolder + "/val/labels", outputFolder + "/labels/val")

if __name__ == "__main__":
    print("This is the start of coco transform format")
    inputCoco = "E:/ubuntushare/data/warehousetools/outputYolo"
    outputCoco = "E:/ubuntushare/data/warehousetools/yoloFinal"
    changeCocoFormat(inputCoco, outputCoco)


