import os
from lxml import etree
import json
import shutil
import cv2

root_path = "../../../../data/TSRD/"

#http://www.nlpr.ia.ac.cn/pal/trafficdata/index.html
#All images are annotated the four corrdinates of the sign and the category.
#fileName.png; width;height;topLeftX;topLeftY;bottomRightX;bottomRightY; category
def writeLabelsFile():

    labelsAllPath = os.path.join(root_path + "val/val.txt")
    with open(labelsAllPath) as f:
        lineContent = f.readlines()
        for eachLine in lineContent:
            parts = eachLine.split(';')
            fileName = parts[0]
            width = int(parts[1])
            height = int(parts[2])
            xmin = int(parts[3])
            ymin = int(parts[4])
            xmax = int(parts[5])
            ymax = int(parts[6])
            category = parts[7]

            fileParts = fileName.split(".")
            list_file = open(os.path.join(root_path, 'val/labels/', '%s.txt' % (fileParts[0])), 'w')
            list_file.write('%s ' % (category))

            xCenter = (xmax + xmin) / (2 * width)
            yCenter = (ymin + ymax ) / (2 * height)
            widthPer = (xmax - xmin) / width
            heightPer = (ymax - ymin)/ height
            list_file.write('%.6f ' % (xCenter))
            list_file.write('%.6f ' % (yCenter))
            list_file.write('%.6f ' % (widthPer))
            list_file.write('%.6f ' % (heightPer))
            list_file.write('\n')

            list_file.close()



if __name__ == "__main__":
    writeLabelsFile()
