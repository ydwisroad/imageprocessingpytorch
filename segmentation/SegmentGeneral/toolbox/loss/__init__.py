import torch.nn as nn
import torch as torch
import numpy as np


def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)

    N = gt.size(0)
    #print("original pred size ", pred.size())
    #print("original groundTruth size ", gt.size())
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N


class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, activation='sigmoid'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeff(y_pr, y_gt, activation=self.activation)


def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape to vector
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)   # convert to one hot
    size.append(N)  #  reshape to original size
    return ones.view(*size)

def get_loss(cfg, weight=None):

    '''

        :param cfg:
        :param weight: class weighting
        :param ignore_index: class to ignore, 一般为背景id
        :return:
    '''

    #assert cfg['loss'] in ['crossentropyloss2D']
    #assert len(weight) == cfg['n_classes']

    return {
        'crossentropyloss2D': nn.CrossEntropyLoss(weight=weight),
        'softDiceLoss': SoftDiceLoss(),
        'bceWithLogitsLoss':nn.BCEWithLogitsLoss()

    }[cfg['loss']]
