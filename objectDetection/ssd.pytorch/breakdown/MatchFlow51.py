import torch


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax  在列上拼接起来

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
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

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

truths = torch.Tensor([[1,1,3,3],  # 角标
                       [4,4,5,5],
                       [5,5,6,6]])
print(" truths " , truths)


prior_boxes = torch.Tensor([[2,2,2,2],  # 中心坐标
                            [4,4,3,3],
                            [3,3,2,4],
                            [4,4,2,2],
                            [5,5,4,4],
                            [6,6,4,4],
                            [7,7,2,2],
                            [8,9,2,2],
                            [9,9,2,2]])
print("prior_boxes central point", prior_boxes)

priors = point_form(prior_boxes)  # 中心坐标转化为角标
print("priors angle point " , priors)

A = truths.size(0)  # shape(2,4)
B = prior_boxes.size(0)  # shape(6,4)
print('A={}, B={}'.format(A, B))

intersectRes = intersect(truths, priors)
print("intersectRes ", intersectRes )

overlaps = jaccard(
        truths,
        point_form(prior_boxes)
    )
print("overlaps ", overlaps)

# 在维度1上返回最大值和相对应的下标
# 这里就表示得到和gt最匹配的priorbox(anchor)
best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

print(best_prior_overlap)
print(best_prior_idx )

print(overlaps)
best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
print(overlaps.max(0, keepdim=True))

best_truth_idx.squeeze_(0)
best_truth_overlap.squeeze_(0)

best_prior_idx.squeeze_(1)
best_prior_overlap.squeeze_(1)

print(best_prior_idx)  # 与3个btbox最匹配的anchor索引
print(best_truth_overlap)  # 9个anchor与每个btbox的最佳关系  每个anchor与gtbox的最大匹配值
best_truth_overlap.index_fill_(0, best_prior_idx, 2)

best_prior_idx.size(0)

for j in range(best_prior_idx.size(0)):
    print(best_prior_idx[j])  # 与当前 gtbox[j] 最匹配的 anchor索引idx
    best_truth_idx[best_prior_idx[j]] = j  # 当前这个anchor[idx]匹配最好的gtbox索引指定为 j
    print(best_truth_idx)

matches = truths[best_truth_idx]
print(matches)

print(best_truth_idx)
labels = torch.Tensor([0,2,3])
# 每一个prior对应的类别
# 由于背景也算一类这里要+1 conf = labels[best_truth_idx] + 1 对于测试需要，先不加
conf = labels[best_truth_idx]
print("conf(classification):", conf)

print(best_truth_overlap)
print(best_truth_overlap < 0.5)
# 如果一个PriorBox对应的最大IoU小于0.5，则视为负样本
conf[best_truth_overlap < 0.5] = 0
print(conf)

variances = [0.1, 0.2]
encodedVar = encode(matches, priors, variances)
print("matches in gth:", matches)
print("priors  all:", priors)
print("variances: ", variances)
print("encodedVar:", encodedVar)
































