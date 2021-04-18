from torch import nn
import torch
import torch.nn.functional as F
#https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247550944&idx=3&sn=562facd2ad9af1ef087ede6b9f66b54d&chksm=ec1cea19db6b630f2827a7ada6ee99580757aa4cc7073a65fa93da05dd69bdaa663d371a820b&mpshare=1&scene=1&srcid=0412Hl8FPlGwSrdm997YN7Dz&sharer_sharetime=1618188068550&sharer_shareid=03101a931987a40bb1c69d01fec93b52&exportkey=AYrpu%2BUT5zplHT%2FW0p0Mqwg%3D&pass_ticket=Mwvmf2YSuj%2F34nnYV%2FSCB5JR%2BHIey1PpZDZaGNT8PdnSYVXiANNAEYIi3b0g1%2BFV&wx_header=0#rd
#https://arxiv.org/pdf/2102.10882v2.pdf

class PEG(nn.Module):

    def __init__(self, dim=256, k=3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)

        # Only for demo use, more complicated functions are effective too.
    def forward(self, x):
        B, C, H, W = x.shape
        #cls_token, feat_token = x[:, 0], x[:, 1:] # cls token不参与PEG
        feat_token = x
        cnn_feat = feat_token #.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat # 产生PE加上自身
        #x = x.flatten(2).transpose(1, 2)
        #x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
