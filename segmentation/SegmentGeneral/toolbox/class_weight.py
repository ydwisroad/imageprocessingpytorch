import os
from PIL import Image
import numpy as np
import torch


def linknet_class_weight(num_classes):
    p_class = num_classes / num_classes.sum()
    return 1 / (np.log(1.02 + p_class))


def compute_weight(root, n_classes):
    num_classes = np.zeros(n_classes)
    for image in os.listdir(root):
        image = Image.open(os.path.join(root, image))
        image = np.asarray(image)   # 360, 480
        image = np.asarray(image).reshape(-1)   # 360 * 480
        num = np.bincount(image)        # len = 12
        num_classes += num      # 每个类别出现的总次数

    weight = linknet_class_weight(num_classes)

    return torch.Tensor(weight.tolist())