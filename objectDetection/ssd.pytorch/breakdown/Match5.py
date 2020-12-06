import torch
import numpy as np

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)  # 交集
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter  # 并集
    return inter / union  # [A,B]    # 交并比 IoU

def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


# 输入包括IoU阈值、真实边框位置、预选框、方差、真实边框类别
# 输出为每一个预选框的类别，保存在conf_t中，对应的真实边框位置，保存在loc_t中
def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """

    # 注意这里truth是最大最小值形式的,而prior是中心点与长宽形式
    # 求取真实框与预选框的IoU
    overlaps = jaccard(
        truths,
        point_form(priors)
    )

    # (Bipartite Matching) 双向匹配
    # 这里就表示得到和gt最匹配的priorbox(anchor)
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # 在零维度上取最大值 表示： 找到和每个anchor最匹配的gtbox
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    best_prior_idx.squeeze_(1)  # 与gtbox最匹配的anchor 索引值
    best_prior_overlap.squeeze_(1)  # 与gtbox最匹配的anchor 交并比

    best_truth_idx.squeeze_(0)  # 最匹配的gtbox 索引值
    best_truth_overlap.squeeze_(0)  # 最匹配的gtbox 交并比

    # 将每一个truth对应的最佳box的overlap设置为2
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior

    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap

    # 保证每一个gtbox都能对应到prior box上
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # 每一个prior对应的真实框的位置
    matches = truths[best_truth_idx]  # Shape: [num_priors,4]

    # 每一个prior对应的类别
    conf = labels[best_truth_idx] + 1  # Shape: [num_priors]

    # 如果一个PriorBox对应的最大IoU小于0.5，则视为负样本
    conf[best_truth_overlap < threshold] = 0  # label as background

    # 进一步计算定位的偏移真值
    loc = encode(matches, priors, variances)  # 返回编码后的中心坐标和宽高.
    loc_t[idx] = loc  # 设置第idx张图片的gt编码坐标信息
    conf_t[idx] = conf  # 设置第idx张图片的编号信息.(大于0即为物体编号, 认为有物体, 小于0认为是背景)


prior_boxes = torch.Tensor([[2,2,2,2], [4,4,3,3],[3,3,2,4],[4,4,2,2],[5,5,4,4],[6,6,4,4]])
truths = torch.Tensor([[1,1,2,2], [3,3,4,4]])
print(truths)
print(prior_boxes)
point_form(prior_boxes)

overlaps = jaccard(
        truths,
        point_form(prior_boxes)
    )
print(overlaps)
print(overlaps.max(1, keepdim=True))



