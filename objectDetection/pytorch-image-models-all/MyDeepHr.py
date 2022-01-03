import torch
import torch.nn as nn
import math
from collections import OrderedDict
from typing import List
import torch.nn.functional as F
from timm.models.resnet import BasicBlock, Bottleneck
_BN_MOMENTUM = 0.1

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.fuse_act = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        error_msg = ''
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
        elif num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
        elif num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
        if error_msg:
            _logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=_BN_MOMENTUM),
            )

        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return nn.Identity()

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=_BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=_BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=_BN_MOMENTUM),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x: List[torch.Tensor]):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])

        x_fuse = []
        for i, fuse_outer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_outer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + fuse_outer[j](x[j])
            x_fuse.append(self.fuse_act(y))

        return x_fuse

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

if __name__ == "__main__":
    print("This is the start of My CrossVit")
    #num_branches =2  num_blocks = (2, 2)  num_inchannels = [16, 32]  num_channels = (16, 32)   fuse_method = 'SUM'
    # reset_multi_scale_output = True   
    hrModule = HighResolutionModule(num_branches = 2, blocks = blocks_dict['BASIC'], num_blocks = (2, 2), num_inchannels = [32, 64],
                                    num_channels = (16, 32) ,
                                    fuse_method= 'SUM', multi_scale_output = True)
    inputVecA = torch.randn(10, 32, 256, 256)
    inputVecB = torch.randn(10, 64, 128, 128)

    list = []
    list.append(inputVecA)
    list.append(inputVecB)
    hrOutput = hrModule(list)
    print("hroutput shape ", len(hrOutput))
    print("hroutput 0 ,1  ", hrOutput[0].shape, " ", hrOutput[1].shape)