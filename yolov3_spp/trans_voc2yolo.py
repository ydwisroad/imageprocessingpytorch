#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
"""
?????????
1.?voc???????(.xml)??yolo????(.txt)???????????????
2.??json?????????names??(my_data_label.names)
"""

import os
from tqdm import tqdm
from lxml import etree
import json
import shutil


# voc??????????
voc_root = "/Users/i052090/Downloads/segmentation/data/ydbridge/all/VOCAll"
voc_version = "./"

# ?????????????txt??
train_txt = "train.txt"
val_txt = "val.txt"

# ??????????
save_file_root = "/Users/i052090/Downloads/segmentation/data/ydbridge/all/VOCAll/yolo"

# label????json??
label_json_path = '/Users/i052090/Downloads/segmentation/data/markedhkbridge/VOC/VOC2012/voc_classes.json'

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
assert os.path.exists(label_json_path), "label_json_path does not exist..."
if os.path.exists(save_file_root) is False:
    os.makedirs(save_file_root)


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


def translate_info(file_names, save_root, class_dict, train_val='train'):
    """
    ???xml??????yolo????txt????
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
        # ???????????
        img_path = os.path.join(voc_images_path, file + ".jpg")
        assert os.path.exists(img_path), "file:{} not exist...".format(img_path)

        # ??xml??????
        xml_path = os.path.join(voc_xml_path, file + ".xml")
        assert os.path.exists(xml_path), "file:{} not exist...".format(xml_path)

        # read xml
        with open(xml_path) as fid:
            xml_str = fid.read().encode()
        #xml_str = xml_str.decode('utf-8').encode('ascii')
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data["size"]["height"])
        img_width = int(data["size"]["width"])

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


def create_class_names(class_dict):
    keys = class_dict.keys()
    with open("./data/my_data_label.names", "w") as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + "\n")


def main():
    # read class_indict
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)

    # ??train.txt????????????
    with open(train_txt_path, "r") as r:
        train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc???yolo???????????????
    translate_info(train_file_names, save_file_root, class_dict, "train")

    # ??val.txt????????????
    with open(val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc???yolo???????????????
    translate_info(val_file_names, save_file_root, class_dict, "val")

    # ??my_data_label.names??
    create_class_names(class_dict)


if __name__ == "__main__":
    main()
