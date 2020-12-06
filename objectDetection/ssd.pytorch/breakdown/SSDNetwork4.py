import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./')
sys.path.append('../')

from layers.modules.l2norm import *

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.Relu(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [pool5,
               conv6,
               nn.ReLU(inplace=True),
               conv7,
               nn.ReLU(inplace=True)]

    return layers

base = [64, 64, 'M',
        128, 128, 'M',
        256, 256, 256, 'C',
        512, 512, 512, 'M',
        512, 512, 512]

vgg_base = vgg(base, 3)
print("vgg_base ", vgg_base)
def add_extra(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':  # 缩放特征图 stride=2
                layers += [
                    nn.Conv2d(in_channels, cfg[k + 1],
                            kernel_size=(1,3)[flag],
                            stride=2, padding=1)
                ]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]  # vgg网络的第21层和倒数第2层

    for k, v in enumerate(vgg_source):
        loc_layers += [
            nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [
            nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [
            nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [
            nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


from math import sqrt as sqrt
from itertools import product as product

voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'image_size': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


class PriorBox(nn.Module):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    # 由配置文件而来的一些配置信息
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    # 生成所有的priorbox需要相应特征图的信息
    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):  # 'feature_maps': [38, 19, 10, 5, 3, 1],
            for i, j in product(range(f), repeat=2):
                # f_k 为每个特征图的尺寸
                f_k = self.image_size / self.steps[k]  # self.image_size=300 'steps': [8, 16, 32, 64, 100, 300]
                # 求每个box的中心坐标  将中心点坐标转化为 相对于 特征图的 相对坐标 （0，1）
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # 对应{Sk,Sk}大小的priorbox
                s_k = self.min_sizes[k] / self.image_size  # 'min_sizes': [30, 60, 111, 162, 213, 264],
                mean += [cx, cy, s_k, s_k]
                # 对应{sqrt(Sk*Sk+1), sqrt(Sk*Sk+1)}大小的priorbox
                s_k_prime = sqrt(
                    s_k * (self.max_sizes[k] / self.image_size))  # 'max_sizes': [60, 111, 162, 213, 264, 315]
                mean += [cx, cy, s_k_prime, s_k_prime]
                # 对应比例为2、 1/2、 3、 1/3的priorbox
                for ar in self.aspect_ratios[k]:  # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # 将所有的priorbox汇集在一起
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

extra = [256, 'S', 512,
         128, 'S', 256,
         128, 256,
         128, 256]
conv_extras = add_extra(extra, 1024)

print("conv_extras " , conv_extras)

# 每个特征图上一个点对应priorbox的数量
mbox = [4, 6, 6, 6, 4, 4]
base_, extra_, head_ = multibox(vgg_base, conv_extras, mbox, num_classes=21)
print("head_ ", head_)

print("==============" * 12)
priorbox = PriorBox(voc)  # 实例化一个对象 之后 才可调用对象里的输出
output = priorbox()
print("output ", output, " size ", output.size())

mean = []
for k, f in enumerate(voc["feature_maps"]):     # 'feature_maps': [38, 19, 10, 5, 3, 1],
    print("k , f ", k , " ", f, " rangf ", range(f))
    productRes = product(range(f), repeat=2)
    print("productRes ", productRes)
    # 300 / 8 , 16, 32.... 300
    f_k = voc["min_dim"] / voc["steps"][k]
    print("f_k ", f_k)

    s_k = voc["min_sizes"][k] / voc["image_size"]
    print("s_k", s_k)
    for i, j in productRes:
        #traverse every cell of feature map
        #print(" i j ", i , " " , j)
        #central point
        cx = (j + 0.5) / f_k
        cy = (i + 0.5) / f_k
        #print(" cx cy ", cx, " ", cy)
        mean += [cx, cy, s_k, s_k]
        # 对应{sqrt(Sk*Sk+1), sqrt(Sk*Sk+1)}大小的priorbox
        s_k_prime = sqrt(s_k * (voc["max_sizes"][k] / voc["image_size"]))  # 'max_sizes': [60, 111, 162, 213, 264, 315]
        mean += [cx, cy, s_k_prime, s_k_prime]

        for ar in voc["aspect_ratios"][k]:  # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
            mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
mean = torch.Tensor(mean).view(-1, 4)
print("mean ", mean)

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size (int): input image size
        base (model_object): VGG16 layers for input, size of either 300 or 500
        extras (model_object): extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc

        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()

        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])  # head 由muti_layer传进来的参数
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        # sources保存特征图，loc与conf保存所有PriorBox的位置与类别预测特征
        sources = list()
        loc = list()
        conf = list()

        # 对输入图像卷积到conv4_3，将特征添加到sources中
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # 继续卷积到conv7，将特征添加到sources中
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # 继续利用额外的卷积层计算，并将特征添加到sources中
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:  # 间隔一层
                sources.append(x)

        # 对sources中的特征图利用类别与位置网络进行卷积计算，并保存到loc与conf中
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            # 对于训练来说，output包括了loc与conf的预测值以及PriorBox的信息
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized， it must be either 'test' or 'train'")
        return

    # 利用上面的vgg_base与conv_extras网络，生成类别与位置预测网络head_
    base_, extras_, head_ = multibox(
        vgg(base, 3),
        conv_extras,
        mbox,
        num_classes)

    return SSD(phase, size, base_, extras_, head_, num_classes)


ssd = build_ssd('train')
print("ssd ", ssd)







