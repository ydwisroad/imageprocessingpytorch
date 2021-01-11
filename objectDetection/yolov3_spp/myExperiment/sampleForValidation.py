import os
from lxml import etree
import json
import shutil
import cv2

root_path = "../../../../data/TSRD/"

def writeLabelsFile():
    itemsPath = os.path.join(root_path, 'val/images')
    outputPath = os.path.join(root_path, 'val/selected')
    count = 0
    allFiles = os.listdir(itemsPath)
    allFiles.sort()
    for eachFile in allFiles:
        count = count + 1
        if (count % 10 == 0):
            fullLabelPath = itemsPath + '/' + eachFile
            fileParts = eachFile.split(".")
            labelFile = root_path +  'val/labels' + '/' + fileParts[0] + '.txt'
            print("move image File ", fullLabelPath)
            shutil.move(fullLabelPath, outputPath)
            print("move labelFile File ", labelFile)
            shutil.move(labelFile, outputPath)

if __name__ == "__main__":
    writeLabelsFile()



