import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from toolbox import class_weight


class LabelProcessor:   # 对应data process and load.ipynb 1.处理标签文件中colormap的数据
    """对标签图像的编码"""

    def __init__(self, file_path):

        self.colormap = self.read_color_map(file_path)

        self.cm2lbl = self.encode_label_pix(self.colormap)

    # 静态方法装饰器， 可以理解为定义在类中的普通函数，可以用self.<name>方式调用
    # 在静态方法内部不可以示例属性和实列对象，即不可以调用self.相关的内容
    # 使用静态方法的原因之一是程序设计的需要（简洁代码，封装功能等）
    @staticmethod
    def read_color_map(file_path):  # data process and load.ipynb: 处理标签文件中colormap的数据
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap):     # data process and load.ipynb: 标签编码，返回哈希表
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):

        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class CamVid(Dataset):
    def __init__(self, file_path=[], crop_size=None, num_class=None):

        self.file_path = file_path
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        # 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        # 初始化数据处理函数设置
        self.crop_size = crop_size
        self.num_class = num_class

        self.label_processor = LabelProcessor(file_path[2])
        self.class_weight = class_weight.compute_weight(file_path[3], self.num_class)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        img = Image.open(img)
        label = Image.open(label).convert('RGB')

        img, label = self.center_crop(img, label, self.crop_size)

        img, label = self.img_transform(img, label)
        # print('处理后的图片和标签大小：',img.shape, label.shape)
        sample = {'image': img, 'label': label}

        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size):
        """裁剪输入的图片和标签大小"""
        data = ff.center_crop(data, crop_size)
        label = ff.center_crop(label, crop_size)
        return data, label

    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        label = np.array(label)  # 以免不是np格式的数据
        label = Image.fromarray(label.astype('uint8'))
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform_img(img)
        label = self.label_processor.encode_label_img(label)
        label = t.from_numpy(label)

        return img, label




