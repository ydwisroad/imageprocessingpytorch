from torch.utils.data import Dataset

import os.path as osp
import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
from PIL import Image



class my_data(Dataset):
    def __init__(self, image_path, annotation_path, transform=None):
        return None

    def __len__(self):
        return None

    def __getitem__(self, index):
        return None


VOC_CLASSES = ( 'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')

# 处理xml文件
class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):

            difficult = int(obj.find('difficult').text) == 1
            # 是否保留难以分类的图片
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_sets (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load (default: 'VOC2007')
        label_show (bool, default:False): when True ,display images and labels only used in demonstrating data
    """

    def __init__(self,
                 root,
                 image_sets=[('2012', 'trainval_demoData')],
                 transform=None,
                 target_transform=VOCAnnotationTransform(),  # 实例化标签处理的对象
                 dataset_name='VOC0712',
                 label_show=False):

        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.label_show = label_show

        self._annopath = osp.join('%s', 'Annotations', '%s.xml')  # %s 字符串通配符
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()  # 用于保存图片数据的id

        for (year, name) in image_sets:
            # data/VOCdevkit/VOC2012
            rootpath = osp.join(self.root, 'VOC' + year)
            # data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))  # 2008_000002

    # 重写数据集大小
    def __len__(self):
        return len(self.ids)

    # 重写指定图片获取函数
    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        img_id = self.ids[index]  # trainval_demoData.txt 中有10条图片的id
        print('数据索引ids[{}]:'.format(index), img_id)
        # print(self._annopath)
        print('label文件路径：', self._annopath % img_id)  # 使用字符串通配符 设置对应路径
        print('img文件路径：', self._imgpath % img_id)
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)  # 将图片读入为3维张量
        # img = Image.open(self._imgpath % img_id)
        # plt.imshow(img)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        if self.label_show:
            self.draw_labels(index)

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    # 演示使用
    def draw_labels(self, index):
        img_id = self.ids[index]
        img_path = self._imgpath % img_id

        img = Image.open(img_path)
        target = ET.parse(self._annopath % img_id).getroot()
        target = self.target_transform(target, 1, 1)

        plt.figure(figsize=(12, 8))
        plt.axis('off')
        plt.imshow(img)

        current_axis = plt.gca()
        # Set the colors for the bounding boxes
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        # Just so we can print class names onto the image instead of IDs
        classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']

        # Draw the ground truth boxes in green (omit the label for more clarity)
        for box in target:
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            label = '{}'.format(classes[int(box[4])])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='green', fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='red', bbox={'facecolor': 'white', 'alpha': 0.6})

#dataset = my_data('../../../../data/VOCdevkit/VOC2012/JPEGImages/', '../../../../data/VOCdevkit/VOC2012/Annotations')
dataset = VOCDetection(root='../../../../data/VOCdevkit',image_sets=[('2012', 'trainval')],
                       transform=None, label_show=True)
dataset.pull_item(0)
dataset.draw_labels(8)

#for data in dataset:
    #print(data)

