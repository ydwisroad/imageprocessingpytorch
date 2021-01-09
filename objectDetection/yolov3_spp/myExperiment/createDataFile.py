import os,sys
from lxml import etree
import json
import shutil
import cv2

root_path = "../../../data/TrafficSign/"

save_file_root = "../../../data/TrafficSign/"

line_path = "../../../data/TrafficSign/"

def writeDataFile():

    list_file = open(os.path.join(save_file_root, 'valtraffic.data'), 'w')

    trainPath = os.path.join(root_path, 'val/images/')
    for eachFile in os.listdir(trainPath):
        line = os.path.join(line_path, 'val/images/') + eachFile
        print(line)
        list_file.write(line)
        list_file.write("\n")

    list_file.close()

def deleteZeroFile():
    txtPath = os.path.join(root_path, 'val/labels')
    count = 0
    for eachTextFile in os.listdir(txtPath):
        size = os.path.getsize(txtPath + "/" + eachTextFile)
        if (size == 0):
            print(eachTextFile + " 0 ")
            fileName = eachTextFile.split('.')[0]
            imgPath = os.path.join(root_path, 'val/images/', fileName + '.jpg')
            print("imgPath ", imgPath)
            os.remove(imgPath)

    print("count ", count)

if __name__ == "__main__":
    #deleteZeroFile()
    writeDataFile()


