#!/usr/bin/env python
# coding: utf-8
import os
import glob
import albumentations as A
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
import shutil


rootDir = "E:/ubuntushare/data/warehousetools/"
src_images_dir = rootDir + 'VOC2007/JPEGImages' # 原始图片文件夹
src_xml_dir =  rootDir + 'VOC2007/Annotations' # 原始xml文件夹

dest_images_dir =  rootDir + 'centercrop/JPEGImages' # 保存图片文件夹
dest_xml_dir =  rootDir + 'centercrop/Annotations' # 保存xml文件夹
if not os.path.exists(dest_images_dir):
    os.makedirs(dest_images_dir, exist_ok=True)
if not os.path.exists(dest_xml_dir):
    os.makedirs(dest_xml_dir, exist_ok=True)

sourceImageSets = rootDir + 'VOC2007/ImageSets'
destImageSets = rootDir + 'centercrop/ImageSets'

def copyFolder(source_path, target_path):
    if os.path.exists(target_path):
        # 如果目标路径存在原文件夹的话就先删除
        shutil.rmtree(target_path)
    shutil.copytree(source_path, target_path)
# In[20]:


# 读图片
def read_img(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# 读xml
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # 从size节点中读取宽高
    size=root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
#     print(width,height)
    bboxes = []
    for obj in root.iter('object'):
        xml_box = obj.find('bndbox')
        xmin = (float(xml_box.find('xmin').text))
        ymin = (float(xml_box.find('ymin').text))
        xmax = (float(xml_box.find('xmax').text))
        ymax = (float(xml_box.find('ymax').text))

        name = obj.find('name').text
        bboxes.append([xmin, ymin, xmax, ymax, name])
    return bboxes

# crop
def crop_img(image, bboxes, width=640, height=640):
    transform = A.Compose([
        A.CenterCrop(width=width, height=height, p=1),
        ], bbox_params=A.BboxParams(format='pascal_voc'))
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    return transformed_image, transformed_bboxes

# 保存图片
def save_img(img_path, image):
    cv2.imwrite(img_path, image)

# 保存xml
def save_xml(save_path, im_name, im_shape,  bboxes):
    if not bboxes:
        return
    # 创建dom树
    doc = minidom.Document()

    # 创建根节点，并添加到dom树
    root_node = doc.createElement("annotation")
    doc.appendChild(root_node)

    # 创建元素，为其添加文本节点，以下都是如此
    folder_node = doc.createElement("folder")
    folder_value = doc.createTextNode('4')
    folder_node.appendChild(folder_value)
    root_node.appendChild(folder_node)

    filename_node = doc.createElement("filename")
    filename_value = doc.createTextNode(im_name)
    filename_node.appendChild(filename_value)
    root_node.appendChild(filename_node)

    size_node = doc.createElement("size")
    for item, value in zip(["width", "height", "depth"], im_shape):
        elem = doc.createElement(item)
        elem.appendChild(doc.createTextNode(str(value)))
        size_node.appendChild(elem)
    root_node.appendChild(size_node)

    for bbox in bboxes:
        obj_node = doc.createElement("object")
        name_node = doc.createElement("name")
        name_node.appendChild(doc.createTextNode(bbox[-1]))
        obj_node.appendChild(name_node)

        trun_node = doc.createElement("difficult")
        trun_node.appendChild(doc.createTextNode(str(0)))
        obj_node.appendChild(trun_node)

        bndbox_node = doc.createElement("bndbox")
        for item, value in zip(["xmin", "ymin", "xmax", "ymax"], bbox[:-1]):
            elem = doc.createElement(item)
            elem.appendChild(doc.createTextNode(str(round(value))))
            bndbox_node.appendChild(elem)
        obj_node.appendChild(bndbox_node)
        root_node.appendChild(obj_node)

#     save_path = os.path.join(save_dir, filename.split('.')[0]+'.xml')
    with open(save_path, "w", encoding="utf-8") as f:
        # 将dom树写入文件
        doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")


# In[22]:

def main():
    images = os.listdir(src_images_dir)

    for img_name in tqdm(images):
        xml_name = img_name.replace('.jpg', '.xml')
        dest_img_name = img_name.replace('.jpg', '.png')
        img_path = os.path.join(src_images_dir, img_name)
        xml_path = os.path.join(src_xml_dir, xml_name)
        print("image path ", img_path, " xml_path ", xml_path)
        dest_img_path = os.path.join(dest_images_dir, dest_img_name)
        dest_xml_path = os.path.join(dest_xml_dir, xml_name)
        if os.path.exists(img_path) and os.path.exists(xml_path):
            print("process img ", img_path, " xml ", xml_path)
            image = read_img(img_path)
            bboxes = parse_xml(xml_path)

            flag = True
            for eachBb in bboxes:
                if (int)(eachBb[0]) > (int)(eachBb[2]):
                    flag = False
                if (int)(eachBb[1]) > (int)(eachBb[3]):
                    flag = False
            if (flag == False):
                print("find abnornal bbox ", bboxes)
                continue
            h, w, c = image.shape
            h = w if h > w else h  # 判断宽高，按短边算
            print("bboxes ", bboxes)
            print("w, h ", h)
            t_image, t_bboxes = crop_img(image, bboxes, width=h, height=h)
            # print(t_image.shape, t_bboxes)
            save_img(img_path=dest_img_path, image=t_image)
            save_xml(save_path=dest_xml_path, im_name=img_name, im_shape=(h, h, c),  bboxes=t_bboxes)

if __name__ == "__main__":
    copyFolder(sourceImageSets, destImageSets)
    main()





