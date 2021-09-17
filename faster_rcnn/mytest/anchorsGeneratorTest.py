import torch
import torchvision
from torch.nn import functional as F
from torch import nn
from torch.jit.annotations import List, Optional, Dict, Tuple
from torch import Tensor

import sys
sys.path.append('../network_files')
sys.path.append('../backbone')

from resnet50_fpn_model import resnet50_fpn_backbone
from roi_head import *
from boxes import *

from det_utils import *
from dataRetrieval import *

class AnchorsGenerator(nn.Module):
    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device="cpu"):
        print("generate_anchors scales ", scales)
        print("generate_anchors aspect_ratios ", aspect_ratios)

        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        print("generate_anchors h_ratios ", h_ratios)
        w_ratios = 1.0 / h_ratios

        print("generate_anchors w_ratios ", w_ratios, " ", w_ratios[:, None])
        print("generate_anchors scales ", scales, " ", scales[None, :])
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        print("generate_anchors ws ", ws, " hs ", hs)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()

        print("generate_anchors base_anchors ", base_anchors)
        return base_anchors

    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def set_cell_anchors(self, dtype, device):
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def grid_anchors(self, grid_sizes, strides):
        #grid_sizes: feature_map different layers size
        #stride: times
        anchors = []

        #this is the template
        cell_anchors = self.cell_anchors

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # shape: [grid_height] 对应原图上的行坐标
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            print("grid_anchors shifts_x " , shifts_x, " shifts_y ", shifts_y)

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            print("grid_anchors after meshgrid shifts_x ", shift_x, " shifts_y ", shift_y)

            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            print("grid_anchors shifts ", shifts)

            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        print("grid_anchors return anchors ", anchors)
        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]])
        """将计算得到的所有anchors信息进行缓存"""
        key = str(grid_sizes) + str(strides)
        # self._cache是字典类型
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        image_size = image_list.shape[-2:]
        print("image_size ", image_size)

        # [-2:] got last two cells
        grid_sizes = list([feature_map_value.shape[-2:] for feature_mapKey, feature_map_value in feature_maps.items()])
        print("grid_sizes " ,grid_sizes)

        # one step in feature map equate n pixel stride in origin image
        # 计算特征层上的一步等于原始图像上的步长
        strides = [[torch.tensor(image_size[0] / g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] / g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        #width 原图width/feature map width 倍数,   原图height/feature mapheight 倍数 ,取整
        print("strides ", strides)

        # 根据提供的sizes和aspect_ratios生成anchors模板
        dtype = torch.float32
        self.set_cell_anchors(dtype, device)

        # 计算/读取所有anchors的坐标信息（这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # 得到的是一个list列表，对应每张预测特征图映射回原图的anchors坐标信息
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])

        anchors_in_image = []
        # 遍历每张预测特征图映射回原图的anchors坐标信息
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            anchors_in_image.append(anchors_per_feature_map)
        anchors.append(anchors_in_image)

        return anchors

class RPNHead(nn.Module):
    """
    add a RPN head with classification and regression
    通过滑动窗口计算预测目标概率与bbox regression参数

    Arguments:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors=2):
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # 计算预测的目标概率（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))

        return logits, bbox_reg

class RegionProposalNetwork(torch.nn.Module):
    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head

        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_similarity = box_iou

        self.proposal_matcher = Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True
        )
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )
        # use during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3




if __name__ == "__main__":
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    print("anchor_sizes ", anchor_sizes)
    print("aspect_ratios ", aspect_ratios)
    rpn_anchor_generator = AnchorsGenerator(
        anchor_sizes, aspect_ratios
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    rpn_anchor_generator.set_cell_anchors(torch.float32,device)
    numAnchors = rpn_anchor_generator.num_anchors_per_location()
    print(" numAnchors ", numAnchors)

    #images, targets = self.transform(images, targets)  # 对图像进行预处理
    #features = self.backbone(images.tensors)  # 将图像输入backbone得到特征图

    image1, target1 = readImageAndTarget("./", 5)
    image2, target2 = readImageAndTarget("./", 5)

    image_list = torch.Tensor(image1)

    image_list = torch.unsqueeze(image_list, 0)

    print("image_list ", image_list)

    backbone = resnet50_fpn_backbone()
    feature_maps = backbone(image_list)

    print("features shape", feature_maps.items())

    output = rpn_anchor_generator(image_list, feature_maps)

    print("output ", output)

    out_channels = backbone.out_channels
    rpn_head = RPNHead(
        out_channels, 2)

    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3  # rpn计算损失时，采集正负样本设置的阈值
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5  # rpn计算损失时采样的样本数，以及正样本占总样本的比例

    rpn_pre_nms_top_n_train = 2000
    rpn_pre_nms_top_n_test = 1000   # rpn中在nms处理前保留的proposal数(根据score)
    rpn_post_nms_top_n_train = 2000
    rpn_post_nms_top_n_test = 1000,  # rpn中在nms处理后保留的proposal数

    rpn_nms_thresh = 0.7,

    rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
    rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)


    regionProposalNetwork = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)


















