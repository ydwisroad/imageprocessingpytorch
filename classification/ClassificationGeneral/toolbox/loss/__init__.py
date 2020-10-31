import torch.nn as nn
import torch as torch
import numpy as np

def get_loss(cfg, weight=None):

    '''

        :param cfg:
        :param weight: class weighting
        :param ignore_index: class to ignore, 一般为背景id
        :return:
    '''

    return {
        'crossentropyloss': nn.CrossEntropyLoss(weight=weight),
        'bceWithLogitsLoss':nn.BCEWithLogitsLoss()

    }[cfg['loss']]
