from imgUtils import *
from augmentImages import *
from Helpers import *
from util import *

import os
import shutil
import glob

import cv2
import os

xml_head = '''<annotation>
    <folder>VOC2007</folder>
    <!--File Name-->
    <filename>{}</filename>
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>325991873</flickrid>
    </source>
    <owner>
        <flickrid>null</flickrid>
        <name>null</name>
    </owner>    
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>{}</depth>
    </size>
    <segmented>0</segmented>
    '''

xml_obj = '''
    <object>        
        <name>{}</name>
        <pose>Rear</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <!--bounding box-->
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
    '''
xml_end = '''
</annotation>'''

labels = ['0', '1', '2', '3']  # label for datasets

def writeToVOCXML(image_path, yoloLabelPath, annotationOutputPath, outputListFilePath):
    outputListFile = open(outputListFilePath, 'w')

    cnt = 0
    for (root, dirname, files) in os.walk(image_path):  # iterate all images
        for ft in files:
            ftxt = ft.replace('jpg', 'txt')  #
            fxml = ft.replace('jpg', 'xml')
            xml_path = annotationOutputPath + "/" + fxml
            obj = ''

            imageFullPath = root + "/" + ft
            img = cv2.imread(imageFullPath)
            img_h, img_w = img.shape[0], img.shape[1]
            head = xml_head.format(str(fxml), str(img_w), str(img_h), 3)

            #writeStr = imageFullPath + " "
            writeStr = ft.split(".")[0]
            with open(yoloLabelPath + "/" +  ftxt, 'r') as f:  # read text file content
                for line in f.readlines():
                    yolo_datas = line.strip().split(' ')
                    label = int(float(yolo_datas[0].strip()))
                    center_x = round(float(str(yolo_datas[1]).strip()) * img_w)
                    center_y = round(float(str(yolo_datas[2]).strip()) * img_h)
                    bbox_width = round(float(str(yolo_datas[3]).strip()) * img_w)
                    bbox_height = round(float(str(yolo_datas[4]).strip()) * img_h)

                    xmin = str(int(center_x - bbox_width / 2))
                    ymin = str(int(center_y - bbox_height / 2))
                    xmax = str(int(center_x + bbox_width / 2))
                    ymax = str(int(center_y + bbox_height / 2))

                    obj += xml_obj.format(label, xmin, ymin, xmax, ymax)
                    #writeStr = writeStr + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + str(label) + " "
            outputListFile.write(writeStr)
            outputListFile.write("\n")

            with open(xml_path, 'w') as f_xml:
                f_xml.write(head + obj + xml_end)
            cnt += 1
    outputListFile.close()

def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

#Annotations JPEGImages
def yoloDataToVOCData(yoloDataPath, vocOutputPath, outputListFilePath):
    print("will use yoloDataPath ", yoloDataPath)

    annotationPath = vocOutputPath + "/" + "Annotations"
    jpegImagesPath = vocOutputPath + "/" + "JPEGImages"

    create_dir_not_exist(annotationPath)
    create_dir_not_exist(jpegImagesPath)

    #copy images and transform labels
    for imageName in os.listdir(yoloDataPath + "/images" ):
        shutil.copy(yoloDataPath + "/images" + "/" + imageName, jpegImagesPath)

    writeToVOCXML(yoloDataPath + "/images", yoloDataPath + "/labels",
                  annotationPath, outputListFilePath)




if __name__ == "__main__":
    print("This is the start of Yolo Dataset to VOC Dataset ")

    yoloDataToVOCData("/Users/i052090/Downloads/segmentation/data/trafficsign512TT/train/",
                      "/Users/i052090/Downloads/segmentation/data/trafficsign512TT/VOC/train",
                      "/Users/i052090/Downloads/segmentation/data/trafficsign512TT/VOC/train/ImageSets/Main/train.txt")




