#---------------------------------------------#
#   运行前一定要修改classes
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#---------------------------------------------#
import xml.etree.ElementTree as ET
from os import getcwd
import os
import cv2

sets=[('2012', 'trainval'), ('2012', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def centerxywhYoloTox1y1x2y2(width, height, xCenter, yCenter, widthPer, heightPer):
    x = xCenter * width
    y = yCenter * height

    return x - (width * widthPer)/2, y - (height * heightPer)/2, x + (width * widthPer) /2, y + (height * heightPer)/2

def convert_annotation(year, image_id, list_file):
    in_file = open('../../../data/VOCdevkitTest/VOC%s/Annotations/%s.xml'%(year, image_id), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
            
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

def getYoloFormatImagesList(imagesPath, labelsPath, exportListPath):
    print(" This is the start of getYoloFormatImagesList ")

    fOutFile = open(exportListPath, "w")

    for imageFileName in os.listdir(imagesPath):
        eachImageFileData = imagesPath + imageFileName
        print("image file Name ", imageFileName)
        labelFileFullPath = labelsPath + "/" + imageFileName.split(".")[0] + ".txt"
        img_o = cv2.imread(imagesPath + imageFileName)
        print("img_o shape ", img_o.shape)
        with open(labelFileFullPath) as labelFile:
            for line in labelFile:
                lineItems = line.split(" ")
                x1,y1, x2,y2 = centerxywhYoloTox1y1x2y2(int(img_o.shape[0]),int(img_o.shape[1]),
                                                        float(lineItems[1]), float(lineItems[2]),
                                                        float(lineItems[3]), float(lineItems[4]))
                eachImageFileData=eachImageFileData+" " + str(int(x1)) + ","+ str(int(y1))+ "," + str(int(x2)) + "," + str(int(y2)) + ","+ lineItems[0]

        fOutFile.write(eachImageFileData)
        fOutFile.write('\n')

    fOutFile.close()

wd = getcwd()

if __name__ == "__main__":
    print("This is the start of main program of main image augmentation")

    getYoloFormatImagesList("/Users/i052090/Downloads/segmentation/data/trafficMini/images/val/",
                            "/Users/i052090/Downloads/segmentation/data/trafficMini/labels/val/",
                            "./newTrain.txt")

    # for year, image_set in sets:
    #     image_ids = open('../../../data/VOCdevkitTest/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    #     #print("image_ids ", image_ids)
    #     list_file = open('%s_%s.txt'%(year, image_set), 'w')
    #     for image_id in image_ids:
    #         if str(image_id) == '-1' or str(image_id) == '1':
    #             continue
    #         print("image_id ", image_id)
    #         list_file.write('../../../data/VOCdevkitTest/VOC%s/JPEGImages/%s.jpg'%(year, image_id))
    #         convert_annotation(year, image_id, list_file)
    #         list_file.write('\n')
    #     list_file.close()


