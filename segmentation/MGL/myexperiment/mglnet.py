import torch
from torch import nn
import torch.nn.functional as F
#import model.resnet as models
import resnet as models
import numpy as np
#from model.basicnet import MutualNet, ConcatNet
from basicnet import MutualNet, ConcatNet
from config import *

# MGLNet init   layers  50  dropout  0.1  classes  1
# MGLNet init   zoom_factor  8  pretrained  False  args  arch: mgl
# aux_weight: 0.4
# base_lr: 1e-08
# base_size: 473
# batch_size: 3
# batch_size_val: 36F
# classes: 1
# data_root: /root/dataset/cod/COD_train/Image
# dist_backend: nccl
# dist_url: tcp://127.0.0.1:6789
# distributed: False
# epochs: 20
# evaluate: True
# has_prediction: False
# ignore_label: 0
# index_start: 0
# index_step: 0
# keep_batchnorm_fp32: None
# layers: 50
# loss_scale: None
# manual_seed: None
# model_path: model_file/mgl_s.pth
# momentum: 0.9
# multiprocessing_distributed: False
# ngpus_per_node: 1
# num_clusters: 32
# opt_level: O0
# power: 0.9
# print_freq: 20
# rank: 0
# resume: None
# rotate_max: 90
# rotate_min: -90
# save_folder: exp/result/
# save_freq: 5
# save_path: exp/model
# scale_max: 2.0
# scale_min: 0.5
# scales: [1.0]
# split: val
# stage: 1
# start_epoch: 0
# sync_bn: False
# test_batch_size: 1
# test_gpu: [0]
# test_h: 473
# test_list: /root/dataset/cod/COD_train/test.lst
# test_w: 473
# train_gpu: [0]
# train_h: 473
# train_list: /root/dataset/cod/COD_train/train.lst
# train_w: 473
# use_apex: True
# val_list: /root/dataset/cod/COD_train/train.lst
# weight: None
# weight_decay: 0.0001
# workers: 16
# world_size: 1
# zoom_factor: 8

class MGLNet(nn.Module):
    def __init__(self, layers=50, dropout=0.1, classes=1, zoom_factor=8, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True, args=None):
        super(MGLNet, self).__init__()
        print("MGLNet init ", " layers ", layers, " dropout ", dropout, " classes ", classes)
        print("MGLNet init ", " zoom_factor ", zoom_factor, " pretrained ", pretrained, " args ", args)
        assert layers in [50, 101, 152]
        assert classes == 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.args = args
        models.BatchNorm = BatchNorm
        self.gamma = 1.0
        self.training = False

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.dim = 512

        self.pred = nn.Sequential(
            nn.Conv2d(2048, self.dim, kernel_size=3, padding=1, bias=False),
            BatchNorm(self.dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(self.dim, classes, kernel_size=1)
        )

        self.region_conv = self.pred[0:4] # 2048 -> 512
        self.edge_cat = ConcatNet(BatchNorm) # concat low-level feature map to predict edge

        # cascade mutual net
        self.mutualnet0 = MutualNet(BatchNorm, dim=self.dim, num_clusters=args.num_clusters, dropout=dropout)
        if args.stage == 1:
            self.mutualnets = nn.ModuleList([self.mutualnet0])
        elif args.stage == 2:
            self.mutualnet1 = MutualNet(BatchNorm, dim=self.dim, num_clusters=args.num_clusters, dropout=dropout)
            self.mutualnets = nn.ModuleList([self.mutualnet0, self.mutualnet1])

    def forward(self, x, y=None, iter_num=0, y2=None):
        x_size = x.size()
        print("mgl forward starts... x_size", x_size)   #[3, 3, 473, 473]
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        print("mgl forward before layer0 size ", x.shape)  # torch.Size([3, 3, 473, 473])
        ##step1. backbone layer
        x_0 = self.layer0(x)
        print("mgl forward x_0 size ", x_0.shape)  #torch.Size([3, 128, 119, 119])
        x_1 = self.layer1(x_0)
        print("mgl forward x_1 size ", x_1.shape)   # torch.Size([3, 256, 119, 119])
        x_2 = self.layer2(x_1)
        print("mgl forward x_2 size ", x_2.shape)   #torch.Size([3, 512, 60, 60])
        x_3 = self.layer3(x_2)
        print("mgl forward x_3 size ", x_3.shape)    # torch.Size([3, 1024, 60, 60])
        x_4 = self.layer4(x_3)
        print("mgl forward x_4 size ", x_4.shape)    # torch.Size([3, 2048, 60, 60])

        ##step2. concat edge feature by side-output feature
        coee_x = self.edge_cat(x_1, x_2, x_3, x_4) # edge pixel-level feature
        print("mgl forward coee_x size ", coee_x.shape) #coee_x size  torch.Size([3, 512, 60, 60])
        cod_x = self.region_conv(x_4) # 2048 -> 512
        print("mgl forward cod_x size ", cod_x.shape)  # torch.Size([3, 512, 60, 60])

        main_loss = 0.
        for net in self.mutualnets:
            n_coee_x, coee, n_cod_x, cod = net(coee_x, cod_x)
            # torch.Size([3, 512, 60, 60]) torch.Size([3, 1, 60, 60])
            # torch.Size([3, 512, 60, 60]) torch.Size([3, 1, 60, 60])
            print("mgl forward n_coee_x, coee, n_cod_x, cod size ", n_coee_x.shape, coee.shape, n_cod_x.shape, cod.shape)
            coee_x = coee_x + n_coee_x
            print(" coee_x " , coee_x.shape)   # torch.Size([3, 512, 60, 60])
            cod_x = cod_x + n_cod_x
            print(" cod_x ", cod_x.shape)     # torch.Size([3, 512, 60, 60])

            if self.zoom_factor != 1:
                coee = F.interpolate(coee, size=(h, w),  mode='bilinear', align_corners=True)
                cod = F.interpolate(cod, size=(h, w), mode='bilinear', align_corners=True)
                if self.training:
                    main_loss += self.gamma * self.criterion(coee, y2) # supervise edge
                    main_loss += self.criterion(cod, y) # supervise region

        if self.training:
            return cod, coee, main_loss
        else:
            return cod, coee


if __name__ == '__main__':
    print("start to run main of mglnet")
    args = load_cfg_from_cfg_file("../config/cod_mgl50.yaml")

    mglNet = MGLNet(pretrained= False, args = args)

    denseInputs = torch.randn(3, 3, 473, 473)
    output = mglNet(denseInputs)




