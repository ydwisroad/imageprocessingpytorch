import os,sys
from lxml import etree
import json
import shutil
import cv2

root_path = "../../../../data/trafficMini/yolo/"
save_file_root = "../../../../data/trafficMini/yolo/"

line_path = "../../../data/trafficMini/yolo/"

def main():

    list_file = open(os.path.join(save_file_root, 'datatrafficmini.data'), 'w')

    trainPath = os.path.join(root_path, 'val/images/')
    for eachFile in os.listdir(trainPath):
        line = os.path.join(line_path, 'val/images/') + eachFile
        print(line)
        list_file.write('%s ' % (line))

    list_file.close()

if __name__ == "__main__":
    main()


