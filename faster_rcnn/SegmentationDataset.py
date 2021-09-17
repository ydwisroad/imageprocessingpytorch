from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree
import transforms

class SegmentationDataset(Dataset):
    """Parse Dataset for image segmentation"""
    def __init__(self, root, transforms, classesJson= "./classes.json",imageFolder = "JPEGImages" , 
                 annotationFolder="Annotations", txtFile="train.txt"):

        self.root = root

        self.img_root = os.path.join(self.root, imageFolder)
        self.annotations_root = os.path.join(self.root, annotationFolder)

        txt_list = os.path.join(self.root, txtFile)

        with open(txt_list) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines()]

        try:
            json_file = open(os.path.join(self.root, classesJson), 'r')
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def parse_xml_to_dict(self, xml):
        """
        Args：
            xml: xml tree obtained by parsing XML file contents using lxml.etree
        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}


#a simple test example
from draw_box_utils import draw_box
from PIL import Image
import json
import matplotlib.pyplot as plt
import torchvision.transforms as ts
import random

data_transform = {
     "train": transforms.Compose([transforms.ToTensor(),
                                  transforms.RandomHorizontalFlip(0.5)]),
     "val": transforms.Compose([transforms.ToTensor()])
 }

#test case for above class
train_data_set = SegmentationDataset("E:\\roadproject\\experiment\\data\\VOC2007", data_transform["train"],
                                     "pascal_voc_classes.json", "JPEGImages","Annotations", "ImageSets\\Main\\train.txt")

print(len(train_data_set))
category_index = {}
try:
     json_file = open(os.path.join("E:\\roadproject\\experiment\\data\\VOC2007", "pascal_voc_classes.json"),
                                   'r')
     class_dict = json.load(json_file)
     category_index = {v: k for k, v in class_dict.items()}
except Exception as e:
     print(e)
     exit(-1)
for index in random.sample(range(0, len(train_data_set)), k=5):
     img, target = train_data_set[index]
     img = ts.ToPILImage()(img)
     draw_box(img,
              target["boxes"].numpy(),
              target["labels"].numpy(),
              [1 for i in range(len(target["labels"].numpy()))],
              category_index,
              thresh=0.5,
              line_thickness=5)
     plt.imshow(img)
     plt.show()







