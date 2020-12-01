from torch.utils.data import Dataset
import os
import torch
import json

from torchvision.transforms import functional as ff

from PIL import Image
from lxml import etree


def parse_xml_to_dict( xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def readImageAndTarget(vocPath, imageIndex):
    voc_root = ""
    root = os.path.join(vocPath, "../../../../data", "VOCdevkit/VOC2012")
    img_root = os.path.join(root, "JPEGImages")
    annotations_root = os.path.join(root, "Annotations")

    txt_list = os.path.join(root, "ImageSets", "Main", "train.txt")

    with open(txt_list) as read:
        xml_list = [os.path.join(annotations_root, line.strip() + ".xml")
                         for line in read.readlines()]

    try:
        json_file = open('../pascal_voc_classes.json', 'r')
        class_dict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    xml_path = xml_list[imageIndex]
    print("xml path " , xml_path)
    with open(xml_path) as fid:
        xml_str = fid.read()

    xml = etree.fromstring(xml_str)
    data = parse_xml_to_dict(xml)["annotation"]
    img_path = os.path.join(img_root, data["filename"])
    image = Image.open(img_path)
    print("image ", image)

    boxes = []
    labels = []
    iscrowd = []
    for obj in data["object"]:
        xmin = float(obj["bndbox"]["xmin"])
        xmax = float(obj["bndbox"]["xmax"])
        ymin = float(obj["bndbox"]["ymin"])
        ymax = float(obj["bndbox"]["ymax"])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_dict[obj["name"]])
        iscrowd.append(int(obj["difficult"]))

    print("boxes " ,boxes)
    print("labels ", labels)

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
    image_id = torch.tensor([imageIndex])

    print("boxes tensor " , boxes)
    print("labels tensor " , labels)

    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    print("area ", area)

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    image = ff.to_tensor(image)
    print(" image ", image)
    print(" target ", target)

    return image, target

if __name__ == "__main__":
    readImageAndTarget("./", 5)
