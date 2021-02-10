# -*- coding: utf-8 -*-
# @Author  : LG
from yacs.config import CfgNode as CN
import os

### 参数请结合自身项目设定，才能跑出较好的效果。

project_root = os.getcwd()

_C = CN()


_C.FILE = CN()

_C.FILE.PRETRAIN_WEIGHT_ROOT = '../../../data/weights/'   # 会使用到的预训练模型
_C.FILE.MODEL_SAVE_ROOT = project_root+'/Weights/trained'           # 训练模型的保存
_C.FILE.VGG16_WEIGHT = 'vgg16_reducedfc.pth'                        # vgg预训练模型

_C.DEVICE = CN()

_C.DEVICE.MAINDEVICE =  'cpu'   #'cuda:0' # 主gpu
_C.DEVICE.TRAIN_DEVICES = [0,1] # 训练gpu
_C.DEVICE.TEST_DEVICES = [0,1]  # 检测gpu

_C.MODEL = CN()

_C.MODEL.INPUT = CN()
_C.MODEL.INPUT.IMAGE_SIZE = 300         # 模型输入尺寸
_C.MODEL.INPUT.PIXEL_MEAN = [0, 0, 0]   # 数据集均值
_C.MODEL.INPUT.PIXEL_STD = [1, 1, 1]    # 数据集方差

_C.MODEL.ANCHORS = CN()
_C.MODEL.ANCHORS.FEATURE_MAPS = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]  # 特征图大小
_C.MODEL.ANCHORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]   # 检测框大小
_C.MODEL.ANCHORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]  # 检测框大小
_C.MODEL.ANCHORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]    # 不同特征图上检测框绘制比例
_C.MODEL.ANCHORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]  # 不同特征图上特征点的检测框数量
_C.MODEL.ANCHORS.OUT_CHANNELS = [512, 1024, 512, 256, 256, 256] # 特征图数量
_C.MODEL.ANCHORS.CLIP = True            # 越界检测框截断,0~1
_C.MODEL.ANCHORS.THRESHOLD = 0.5        # 交并比阈值
_C.MODEL.ANCHORS.CENTER_VARIANCE = 0.1  # 解码
_C.MODEL.ANCHORS.SIZE_VARIANCE = 0.2    # 解码

_C.TRAIN = CN()

_C.TRAIN.NEG_POS_RATIO = 3      # 负正例比例
_C.TRAIN.MAX_ITER = 100      # 训练轮数
_C.TRAIN.BATCH_SIZE = 10        # 训练批次
_C.TRAIN.NUM_WORKERS = 4        # 数据数据所使用的线程数
_C.OPTIM = CN()

_C.OPTIM.LR = 1e-3              # 初始学习率.默认优化器为SGD
_C.OPTIM.MOMENTUM = 0.9         # 优化器动量.默认优化器为SGD
_C.OPTIM.WEIGHT_DECAY = 5e-4    # 权重衰减,L2正则化.默认优化器为SGD

_C.OPTIM.SCHEDULER = CN()       # 默认使用MultiStepLR
_C.OPTIM.SCHEDULER.GAMMA = 0.1  # 学习率衰减率
_C.OPTIM.SCHEDULER.LR_STEPS = [80000, 100000]


_C.MODEL.TEST = CN()

_C.MODEL.TEST.NMS_THRESHOLD = 0.45              # 非极大抑制阈值
_C.MODEL.TEST.CONFIDENCE_THRESHOLD = 0.01       # 分数阈值,
_C.MODEL.TEST.MAX_PER_IMAGE = 100               # 预测结果最大数量
_C.MODEL.TEST.MAX_PER_CLASS = -1                # 测试时,top-N


_C.DATA = CN()

# 由于在使用时,是自己的数据集.所以这里,并没有写0712合并的数据集格式,这里以VOC2007为例
_C.DATA.DATASET = CN()
_C.DATA.DATASET.NUM_CLASSES =221
#_C.DATA.DATASET.CLASS_NAME = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
_C.DATA.DATASET.CLASS_NAME = ( '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17',
    '18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33',
    '34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49',
    '50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65',
    '66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81',
    '82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97',
    '98','99','100','101','102','103','104','105','106','107','108','109','110','111',
    '112','113','114','115','116','117','118','119','120','121','122','123','124','125',
    '126','127','128','129','130','131','132','133','134','135','136','137','138','139',
    '140','141','142','143','144','145','146','147','148','149','150','151','152','153',
    '154','155','156','157','158','159','160','161','162','163','164','165','166','167',
    '168','169','170','171','172','173','174','175','176','177','178','179','180','181',
    '182','183','184','185','186','187','188','189','190','191','192','193','194','195',
    '196','197','198','199','200','201','202','203','204','205','206','207','208','209',
    '210','211','212','213','214','215','216','217','218','219','220')

#_C.DATA.DATASET.DATA_DIR = '/Users/i052090/Downloads/segmentation/data/VOCdevkit/VOC2012/'   # 数据集voc格式,根目录
_C.DATA.DATASET.DATA_DIR = '/Users/i052090/Downloads/segmentation/data/trafficsign512TT/VOC/train/'
_C.DATA.DATASET.TRAIN_SPLIT = 'trainMini'       # 训练集,对应于 /VOCdevkit/VOC2007/ImageSets/Main/train.txt'
_C.DATA.DATASET.TEST_SPLIT = 'trainMini'          # 测试集,对应于 /VOCdevkit/VOC2007/ImageSets/Main/val.txt'

_C.DATA.DATALOADER = CN()


_C.STEP = CN()
_C.STEP.VIS_STEP = 10           # visdom可视化训练过程,打印步长
_C.STEP.MODEL_SAVE_STEP = 10   # 训练过程中,模型保存步长
_C.STEP.EVAL_STEP = 1000        # 在训练过程中,并没有进行检测流程,建议保存模型后另外检测

