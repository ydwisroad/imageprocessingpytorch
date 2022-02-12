#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import os
from tqdm import tqdm
from lxml import etree
import json
import shutil
import glob

def parse_xml_to_dict(xml):
    """
    ?xml????????????tensorflow?recursive_parse_xml_to_dict
    Args?
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # ??????????tag?????
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # ????????
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # ??object???????????????
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def translate_info(file_names, save_root, class_dict, voc_images_path, voc_xml_path, imgformat="png", train_val='train'):
    """
    :param file_names:
    :param save_root:
    :param class_dict:
    :param train_val:
    :return:
    """
    save_txt_path = os.path.join(save_root, train_val, "labels")
    if os.path.exists(save_txt_path) is False:
        os.makedirs(save_txt_path)
    save_images_path = os.path.join(save_root, train_val, "images")
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path)

    for file in tqdm(file_names, desc="translate {} file...".format(train_val)):
        print("handling file ", file)
        img_path = os.path.join(voc_images_path, file + "." + imgformat)
        if not os.path.exists(img_path):
            continue

        xml_path = os.path.join(voc_xml_path, file + ".xml")
        if not os.path.exists(xml_path):
            continue

        # read xml
        with open(xml_path) as fid:
            xml_str = fid.read().encode()
        #xml_str = xml_str.decode('utf-8').encode('ascii')
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data["size"]["height"])
        img_width = int(data["size"]["width"])

        if not "object" in data:   #If there is nothing in the object, continue next one.
            continue
        # write object info into txt
        with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
            for index, obj in enumerate(data["object"]):
                # ????object?box??
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"]
                class_index = class_dict[class_name]  #- 1  # ??id?0??

                # ?box?????yolo??
                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                # ????????????6???
                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

        # copy image into save_images_path
        shutil.copyfile(img_path, os.path.join(save_images_path, img_path.split(os.sep)[-1]))


def create_class_names(class_dict, classesFile):
    keys = class_dict.keys()
    with open(classesFile, "w") as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + "\n")

def voc2yolo(sourceVOCPath, destYoloPath, labelJson, imgformat = "png"):
    # voc??????????
    voc_root = sourceVOCPath
    voc_version = "./"

    train_txt = "train.txt"
    val_txt = "val.txt"

    save_file_root = destYoloPath

    imageSurfix = ".png"
    # label????json??
    #label_json_path = rootDir + '/voc_classes.json'

    # ???voc?images???xml???txt??
    voc_images_path = os.path.join(voc_root, voc_version, "JPEGImages")
    voc_xml_path = os.path.join(voc_root, voc_version, "Annotations")
    train_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", train_txt)
    val_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", val_txt)

    # ????/????????
    assert os.path.exists(voc_images_path), "VOC images path not exist..."
    assert os.path.exists(voc_xml_path), "VOC xml path not exist..."
    assert os.path.exists(train_txt_path), "VOC train txt file not exist..."
    assert os.path.exists(val_txt_path), "VOC val txt file not exist..."
    assert os.path.exists(labelJson), "label_json_path does not exist..."
    if os.path.exists(save_file_root) is False:
        os.makedirs(save_file_root)

    # read class_indict
    json_file = open(labelJson, 'r')
    class_dict = json.load(json_file)

    # ??train.txt????????????
    with open(train_txt_path, "r") as r:
        train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc???yolo???????????????
    translate_info(train_file_names, save_file_root, class_dict, voc_images_path, voc_xml_path,imgformat, "train")
    # ??val.txt????????????
    with open(val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc???yolo???????????????
    translate_info(val_file_names, save_file_root, class_dict,  voc_images_path, voc_xml_path, imgformat, "val")

    #
    classesFile = "../data/datalabel.names"
    #create_class_names(class_dict, classesFile)

def copyFilesFromFolder(source, dest, suffix):
    for file in glob.glob(source + '/**/*.' + suffix, recursive=True):
        file = file.replace("\\","/")
        shutil.copyfile(file, dest + file[(file.rfind("/")+1):])

def transform2yolo(inputFolder, outputFolder, imgformat ="png"):
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    if not os.path.exists(outputFolder + "/images"):
        os.mkdir(outputFolder + "/images")
    if not os.path.exists(outputFolder + "/images/train"):
        os.mkdir(outputFolder + "/images/train")
    if not os.path.exists(outputFolder + "/images/val"):
        os.mkdir(outputFolder + "/images/val")
    if not os.path.exists(outputFolder + "/labels"):
        os.mkdir(outputFolder + "/labels")
    if not os.path.exists(outputFolder + "/labels/train"):
        os.mkdir(outputFolder + "/labels/train")
    if not os.path.exists(outputFolder + "/labels/val"):
        os.mkdir(outputFolder + "/labels/val")

    copyFilesFromFolder(inputFolder + "/train", outputFolder + "/images/train/", imgformat)
    copyFilesFromFolder(inputFolder + "/train", outputFolder + "/labels/train/", "txt")

    copyFilesFromFolder(inputFolder + "/val", outputFolder + "/images/val/", imgformat)
    copyFilesFromFolder(inputFolder + "/val", outputFolder + "/labels/val/", "txt")

if __name__ == "__main__":
    print("This is the start of voc 2 yolo")
    voc2yolo()
