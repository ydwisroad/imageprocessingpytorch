import cv2

import sys
sys.path.append('./')

from DataProcessingLoadParseData1 import *

xmlFile = '../../../../data/VOCdevkit/VOC2012/Annotations/2008_000003.xml'
imgFile = '../../../../data/VOCdevkit/VOC2012/JPEGImages/2008_000003.jpg'


class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            # 首先将图像像素值从整型变成浮点型
            ConvertFromInts(),
            # 将标签中的边框从比例坐标变换为真实坐标
            ToAbsoluteCoords(),
            # 因此进行亮度、对比度、色相与饱和度的随机调整，然后随机调换通道
            PhotometricDistort(),
            Expand(self.mean),  # 随机扩展图像大小，图像仅靠右下方
            RandomSampleCrop(),  # 随机裁剪图像
            RandomMirror(),  # 随机左右镜像
            ToPercentCoords(),  # 从真实坐标变回比例坐标
            Resize(self.size),  # 缩放到固定的300*300大小
            SubtractMeans(self.mean)  # 最后进行均值化
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

def image_compare(filePath, transform):
    """
    filePath (string): 图片文件路径
    transform (object): 可传参的对象
    """
    # 读取原始图片
    img = cv2.imread(filePath)
    plt.figure(figsize=(16,14))
    plt.subplot(2,1,1)
    plt.imshow(img)
    # 调用这个实列化后的对象
    image, _, _ = transform(img)
    # 读取第二张图片
    plt.subplot(2,1,2)
    plt.imshow(image)
    plt.show()
    #print(image)  # image 一个三维的张量

from numpy import random
import numpy as np

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        # 随机确定是否变换
        # if random.randint(2):
        # image.dtype = np.uint8  转换到 np.float64
        image = image.astype(np.float64)
        image[:, :, 1] *= random.uniform(self.lower, self.upper)  # 以标准分布的形式生成0.5到1.5之间的随机数
        # 有np.float64 再转回到 np.uint8 方便图片输出
        image = image.astype(np.uint8)
        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        # if random.randint(2):
        image = image.astype(np.float64)
        image[:, :, 0] += random.uniform(-self.delta, self.delta)
        image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
        image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        image = image.astype(np.uint8)
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        # if random.randint(2):
        # 随机选取一个通道的交换顺序，交换图像三个通道的值
        swap = self.perms[random.randint(len(self.perms))]
        shuffle = SwapChannels(swap)  # shuffle channels  # 实例化一个SwapChannels对象
        image = shuffle(image)
        return image, boxes, labels


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        #if random.randint(2):
        alpha = random.uniform(self.lower, self.upper)
        image = image.astype(np.float64)
        image *= alpha
        image = image.astype(np.uint8)
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        # if random.randint(2):
        # 随机选取一个位于[-32, 32)区间的数，相加到图像上
        delta = random.uniform(-self.delta, self.delta)
        image = image.astype(np.float64)
        image += delta
        image = image.astype(np.uint8)
        return image, boxes, labels


# 计算图像边框交并比
def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4] data_type: [[xmin, ymin, xmax, ymax],[...],[...]}
        box_b: Single bounding box, Shape: [4]  data_type: [xmin, ymin, xmax, ymax]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def intersect(box_a, box_b):
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    # print('min_xy\n', min_xy)
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    # print('max_xy\n', max_xy)
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def image_label_tran(filePath, labelPath, transform):
    """
    filePath (string): 图片文件路径
    labelPath (string): 标签文件路径
    transform (object): 可传参的对象
    """
    # 读取原始图片
    img = cv2.imread(filePath)  # 三维张量

    # 首先实例化一个xml文件处理对象
    xml_parser = VOCAnnotationTransform()
    target = ET.parse(labelPath).getroot()
    gt_boxes = xml_parser(target, 1, 1)
    # 画出原标签
    plt.figure(figsize=(16, 14))
    plt.subplot(2, 1, 1)
    plt.title('Original')
    plt.imshow(img)
    current_axis = plt.gca()
    # Just so we can print class names onto the image instead of IDs
    classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    # Draw the ground truth boxes in green (omit the label for more clarity)
    for box in gt_boxes:
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        label = '{}'.format(classes[int(box[4])])
        current_axis.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='yellow', fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='red', bbox={'facecolor': 'white', 'alpha': 0.6})
    # 调用这个实列化后的对象
    image, new_boxes, _ = transform(img, gt_boxes, labels=None)
    # print(new_boxes)
    # 画出转换换后的图片
    plt.subplot(2, 1, 2)
    plt.title('Transformed')
    plt.imshow(image)

    current_axis = plt.gca()
    for box in new_boxes:
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        label = '{}'.format(classes[int(box[4])])
        current_axis.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='yellow', fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='red', bbox={'facecolor': 'white', 'alpha': 0.6})
    plt.show()


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels=None):
        if random.randint(2):
            return image, boxes, labels
        # 求取原图像在新图像中的左上角坐标值
        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        # 建立新的图像，并依次赋值
        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        # 对边框也进行相应变换
        boxes = np.array(boxes.copy())
        boxes_cor = boxes[:, : -1]
        boxes_cor[:, :2] += (int(left), int(top))
        boxes_cor[:, 2:] += (int(left), int(top))

        return image, boxes, labels


MEANS = (104, 117, 123)
trans = Expand(MEANS)
image_label_tran(imgFile, xmlFile, trans)

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                   self.size))
        return image, boxes, labels

class RandomMirror(object):
    def __call__(self, image, boxes, classes=None):
        _, width, _ = image.shape
        #if random.randint(2):
        # 这里的::代表反向，即将每一行的数据反向遍历，完成镜像
        image = image[:, ::-1]
        boxes = np.array(boxes.copy())
        boxes_coor = boxes[:,: -1]  # boxes_coor 引用 boxes中的元素
        boxes_coor[:, 0::2] = width - boxes_coor[:, 2::-2]  # 改变boxes_coor 中的相应元素，也改变了boxes中的相应元素
        return image, boxes, classes  # 这里返回的boxer[x1, y1, x2, y2, class] 其中x1, x2 做了相应变换

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):

        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        if boxes is not None:
            boxes = np.array(boxes)
            boxes_cor = boxes[:, :-1]  # 处理 boxes的坐标部分 boxes:[xmin, ymin, xmax, ymax]
            labels = boxes[:, -1]
        while True:  # 为什么要一直循环呢？  对应与 mode = None 和 其他选择，反正一定要选上一个mode
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            print('mode', mode)

            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # 循环50次要干什么？
            # 反复循环找到图像合适的长宽比 有一个不合适就循环下一个，直到合适后退出（return）
            for _ in range(50):  # 最多尝试50次

                current_image = image

                w = random.uniform(0.3 * width, width)  # 取[0.3width ,width) 之间的一个随机数
                h = random.uniform(0.3 * height, height)  # 取[0.height ,height) 之间的一个随机数

                # aspect ratio constraint b/t .5 & 2  限制图像长宽比
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                print('rect=[int(left), int(top), int(left+w), int(top+h)] = ', rect)

                # 计算裁剪后的图片和原bbox的交并比
                overlap = jaccard_numpy(boxes_cor, rect)
                print('jaccard_numpy = ', overlap)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                # plt.imshow(image)
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                # plt.imshow(current_image)
                # plt.show()
                # keep overlap with gt box IF center in sampled patch
                centers = (boxes_cor[:, :2] + boxes_cor[:, 2:]) / 2.0  # 坐标(xmin, ymin)和坐标(xmax， ymax)的中点
                print('boxes_cor', boxes_cor)
                print('centers: \n', centers)

                # mask in all gt boxes that above and to the left of centers
                # rect[0] = left ; rect[1] = top
                # centers所有box的中点坐标  centers[:, 0]中点的y坐标 ；centers[:, 1]中点的x坐标
                # 判断图像剪裁后 所有原来边框的中点是不是在剪裁的图片内

                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])  #
                print('边框中心centers与剪裁图片左上角rect[:2]关系:', m1)
                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                print('边框中心centers与剪裁图片右下角rect[2:]关系:', m2)
                # mask in that both m1 and m2 are true
                mask = m1 * m2
                print("是否保留边框：", mask)

                # 如果没有有效边框的话，就再随机剪裁一次
                if not mask.any():
                    continue

                # 选择保留下来的边框
                current_boxes = boxes_cor[mask, :]

                # 选择对应保留下来的边框标签
                current_labels = labels[mask]

                # 剪裁图片后 标注框限定在剪裁后的图片内
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                crop_box = np.zeros((current_boxes.shape[0], current_boxes.shape[1] + 1))

                crop_box[:, :-1] = current_boxes
                crop_box[:, -1] = current_labels
                print('剪裁后得到的图片边框：\n', crop_box)
                return current_image, crop_box, labels


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            # ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            # ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


# 应这是测试演示notebook 综合光学和几何变换后图片和边框的位置会有出入
class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            # 首先将图像像素值从整型变成浮点型
            # ConvertFromInts(),
            # 将标签中的边框从比例坐标变换为真实坐标
            # ToAbsoluteCoords(),
            # 因此进行亮度、对比度、色相与饱和度的随机调整，然后随机调换通道
            PhotometricDistort(),
            RandomMirror(),  # 随机左右镜像
            Expand(self.mean),  # 随机扩展图像大小，图像仅靠右下方
            RandomSampleCrop(),  # 随机裁剪图像
            ToPercentCoords(),  # 从真实坐标变回比例坐标
            Resize(self.size),  # 缩放到固定的300*300大小
            # SubtractMeans(self.mean)  # 最后进行均值化
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


ssd_trans = SSDAugmentation()
image_label_tran(imgFile, xmlFile, ssd_trans)

trans = PhotometricDistort()
image_label_tran(imgFile, xmlFile, trans)

trans = RandomSampleCrop()
image_label_tran(imgFile, xmlFile, trans)


from matplotlib import pyplot as plt
from numpy import random


# 示例化一个对象
resize = Resize()
image_compare(imgFile, resize)


trans = RandomBrightness()
image_compare(imgFile, trans)

trans = RandomContrast()
image_compare(imgFile, trans)

trans = RandomLightingNoise()
image_compare(imgFile, trans)

# 实例化一个对象
randomsaturation = RandomSaturation()
image_compare(imgFile, randomsaturation)

randomhue = RandomHue()
image_compare(imgFile, randomhue)

