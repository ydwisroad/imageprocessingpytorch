import torch
import torch.nn.functional as F


# 函数解析参见 匹配的notebook
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

# 上述操作总体为：
def log_sum_exp(x):
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# 求取真实框与预选框的IoU
def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)  # 交集
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter  # 并集
    return inter / union  # [A,B]    # 交并比


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    overlaps = jaccard(
        truths,
        priors
    )
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    # 每一个prior对应的真实框的位置
    matches = truths[best_truth_idx]  # Shape: [num_priors,4]
    # 每一个prior对应的类别
    conf = labels[best_truth_idx]  # Shape: [num_priors]
    # 如果一个PriorBox对应的最大IoU小于0.5，则视为负样本
    conf[best_truth_overlap < threshold] = 0  # label as background
    # 进一步计算定位的偏移真值
    loc = encode(matches, priors, variances)
    # 将匹配好的每个batch_item数据 放到batch的批量数据中
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

targets = torch.Tensor([
    [[2,2,3,4,1],[2,3,4,4,2]],  # 图片1的gtbox
    [[1,1,2,2,2],[3,3,5,5,1]],  # 图片2的gtbox
    ] )
print("gth box ", targets)

prior_boxes = torch.Tensor([[2,2,2,2],
                            [4,4,3,3],
                            [3,3,2,4],
                            [4,4,2,2],
                            [5,5,4,4],
                            [6,6,4,4],
                            [7,7,2,2],
                            [8,9,2,2],
                            [9,9,2,2]])

# 中心坐标转化为角标
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

priors = point_form(prior_boxes)
print(" priors after angle ", priors)

# 3 假定模型输出的定位数据为：
loc_data = torch.Tensor(
       [[[  5.0000,   6.0667,  -5.0931,  -2.9273],
         [  1.9091,   1.8182,  -4.0580,  -8.5237],
         [  2.5000,   5.0000,  -3.4657,  -7.0472],
         [  0.0000,   2.0000,  -4.5815,  -8.0472],
         [  0.0000,   5.7143,  -6.2638,  -9.7296],
         [ -1.8750,  -2.2500, -10.3972,  -6.9315],
         [ -4.3750,  -2.7500, -10.3972,  -6.9315],
         [ -3.0000,  -4.0000, -10.9861,  -8.0472],
         [ -5.5000,  -4.0000, -11.5129,  -8.0472]],

        [[  1.6667,   1.6667,  -5.4931,  -5.4931],
         [  2.7273,   4.7273,  -5.0580,  -5.0580],
         [  5.0000,   5.0000,  -3.4657,  -4.5815],
         [  3.0000,   3.0000,  -3.5815,  -3.5815],
         [  2.4286,   2.4286,  -2.2638,  -2.2638],
         [  0.0000,   1.0000,  -6.9315,  -6.9315],
         [ -5.6250,  -2.6250, -10.3972, -10.3972],
         [ -6.1111,  -4.5000, -10.9861, -11.5129],
         [ -6.5000,  -7.5000, -11.5129, -11.5129]]])
# 4 假定模型输出的分类数据为：
conf_data = torch.randn(2,9,3)  # 假设一共分三类（包括背景类）
print("confidence data ", conf_data)

# 编码标签数据
# 初始化两个数据容器 用于存放每个batch中标签的定位和分类数据
# 假设 batch_size=2, num_anchor=9, num_classes=2+1
loc_t = torch.Tensor(2, 9, 4)  # 定位  (batch_size, num_anchor，coordinates)
conf_t = torch.LongTensor(2, 9)  # 分类 (batch_size, num_classes)  包含背景类这里 num_classes=3
print("loc_t shape  ", loc_data.shape)
print("conf_t ", conf_data.shape)



# ##############################################################################
# 分离标签数据batch_item的类别和坐标
truths = targets[0][:, :-1].data
labels = targets[0][:, -1].data
variances = [0.1, 0.2]
print(truths,'\n',labels)

match(0.5, truths, priors, variances, labels, loc_t, conf_t, 0)
print('batch_item[0]匹配后的数据：')
print("loc_t[0] " , loc_t[0])
print("conf_t[0] ",  conf_t[0])

# 编码后的标签数据
batch_size = 2
variance = [0.1, 0.2]
for idx in range(batch_size):
    truths = targets[idx][:, :-1].data
    #print(truths)
    labels = targets[idx][:, -1].data
    #print(labels)
    defaults = priors.data
    #print(defaults)
    # 得到每一个prior对应的truth,放到loc_t与conf_t中,conf_t中是类别,loc_t中是[matches, prior, variance]
    match(0.5, truths, defaults, variance, labels, loc_t, conf_t, idx)
print(loc_t)
print(conf_t)  # 这里的分类数据是依据iou筛选出来的

pos = conf_t > 0
print(pos)

pos.unsqueeze(pos.dim())

pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
print(pos_idx)

# 提取相应的预测正样本坐标
# 在预测的定位数据中 将正样本的定位数据拿出来
loc_p = loc_data[pos_idx].view(-1, 4)
print(loc_p)

# 提取相应的标签正样本坐标
# 在编码后的标签定位数据中 将正样本的定位数据拿出来
loc_t = loc_t[pos_idx].view(-1, 4)
print(loc_t)

# 所有正样本的定位损失
loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
print("loss_l ", loss_l)

batch_conf = conf_data.view(-1, 3)
print(batch_conf)

# 正值化
torch.exp(batch_conf-batch_conf.max())

# 提取模型预测分类值
print(conf_t.view(-1, 1))  # 标签匹配的分类数据
print(" batch_conf shape ", batch_conf.shape)          # 模型输出的分类预测值
batch_conf.gather(1, conf_t.view(-1, 1))  # 提取anchor对应的分类数据

# loss_c: [batch*num_priors, 1], 计算每个priorbox预测后的损失
loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
print(" loss_c  shape ", loss_c.shape)

# Hard Negative Mining
loss_c = loss_c.view(pos.size()[0], pos.size()[1])
print("loss_c set pos to 0:", loss_c.shape, " pos shape ", pos.shape)

loss_c[pos] = 0
loss_c

loss_c = loss_c.view(2, -1)  # (batch_size, -1)
loss_c

loss_value, loss_idx = loss_c.sort(1, descending=True)
print(loss_value)
print(loss_idx)

ele, idx_rank = loss_idx.sort(1)
print(ele)
print(idx_rank)

num_pos = pos.long().sum(1, keepdim=True)
num_pos

num_neg = torch.clamp(3*num_pos, max=pos.size(1)-1)
num_neg

num_neg.expand_as(idx_rank)
# 获取到负样本的下标
neg = idx_rank < num_neg.expand_as(idx_rank)
neg

pos_idx = pos.unsqueeze(2).expand_as(conf_data)
pos_idx

neg_idx = neg.unsqueeze(2).expand_as(conf_data)
neg_idx

# 知识点 计算 ta > tb
ta = torch.Tensor([[1, 2], [3, 4]])
tb = torch.Tensor([[1, 1], [2, 4]])
print(ta)
print(tb)
torch.gt(ta, tb)

# 按照pos_idx和neg_idx指示的下标筛选参与损失计算的预测数据
(pos_idx+neg_idx).gt(0)

conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, 3)
conf_p

(pos+neg).gt(0)

conf_t[(pos+neg).gt(0)]

# 按照pos_idx和neg_idx筛选目标数据
# conf_t 是依据iou筛选出来的分类数据
targets_weighted = conf_t[(pos+neg).gt(0)]
targets_weighted

loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
print("loss_c" , loss_c)




















