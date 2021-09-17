import sys
import os
import json
import cv2
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm

src_images_dir = '/Users/i052090/Downloads/roadproject/marks/rename/JPEGImages' # 原始图片文件夹
src_json_dir = '/Users/i052090/Downloads/roadproject/marks/rename/Annotations' # 原始json文件夹

dest_images_dir = '/Users/i052090/Downloads/roadproject/marks/output/JPEGImages' # 保存图片文件夹
dest_xml_dir = '/Users/i052090/Downloads/roadproject/marks/output/Annotations' # 保存xml文件夹

os.makedirs(dest_images_dir, exist_ok=True)
os.makedirs(dest_xml_dir, exist_ok=True)

PRE_DEFINE_CATEGORIES = {
       "spalling": 0,
       "potholes": 1,
       "crack": 2
}
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

def convertJsonToXML(img_path, json_path, dest_xml_path):
    datasetJson = json.load(open(json_path, mode='r', encoding='utf-8'))
    assert type(datasetJson) == dict, 'annotation file format {} not supported'.format(type(datasetJson))

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[0], img.shape[1]
    xmlName = str(os.path.basename(img_path)).replace("jpg","xml")
    head = xml_head.format(str(xmlName), str(img_w), str(img_h), 3)
    obj = ''

    for eachShape in datasetJson["shapes"]:
        label = PRE_DEFINE_CATEGORIES[eachShape["label"]]
        #print("shapes ", label)
        xmin = (int)(eachShape["points"][0][0])
        ymin = (int)(eachShape["points"][0][1])
        xmax = (int)(eachShape["points"][1][0])
        ymax = (int)(eachShape["points"][1][1])
        obj += xml_obj.format(label, xmin, ymin, xmax, ymax)

    with open(dest_xml_path, 'w') as f_xml:
        f_xml.write(head + obj + xml_end)
    f_xml.close()

def copyFiles(images, destRename):
    iCounter = 1000

    for img_name in tqdm(images):
        temp = img_name.replace('.jpg', '.json')
        temp = temp.replace('.JPG', '.json')
        json_name = temp
        img_path = os.path.join(src_images_dir, img_name)
        json_path = os.path.join(src_json_dir, json_name)

        if os.path.exists(img_path) and os.path.exists(json_path):
            shutil.copyfile(img_path, destRename + str(iCounter) + ".jpg")
            shutil.copyfile(json_path, destRename +  str(iCounter) + ".json")

        iCounter = iCounter + 1

    images = os.listdir(src_images_dir)


def batchConvertJsonToXML(images):
    for img_name in tqdm(images):
        json_name = img_name.replace('.jpg', '.json')
        dest_img_name = img_name.replace('.jpg', '.png')
        img_path = os.path.join(src_images_dir, img_name)
        json_path = os.path.join(src_json_dir, json_name)
        #print("image path ", img_path, " json_path ", json_path)
        dest_img_path = os.path.join(dest_images_dir, dest_img_name)
        xml_name = img_name.replace('.jpg', '.xml')
        dest_xml_path = os.path.join(dest_xml_dir, xml_name)
        if os.path.exists(img_path) and os.path.exists(json_path):
            print("process img ", img_path, " json ", json_path)
            convertJsonToXML(img_path, json_path, dest_xml_path)

if __name__ == '__main__':
    images = os.listdir(src_images_dir)

    destRename = "/Users/i052090/Downloads/roadproject/marks/rename/IMG_"
    #copyFiles(images,destRename)
    #rename bad names

    batchConvertJsonToXML(images)
