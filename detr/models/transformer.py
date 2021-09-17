# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .PEG import PEG, PEGThree
from torch import einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .plugplay import *
from .DeformableConv import DeformableConv2d

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        self.pos_block = PEG(d_model)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC  mask.shape: [2, 20, 20]  query_embed.shape: [100, 256] pos_embed.shape: [2, 256, 20, 20]
        bs, c, h, w = src.shape   #src [batch, 256, 20, 20]
        src = src.flatten(2).permute(2, 0, 1)

        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        #Add PEG layer: https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247550944&idx=3&sn=562facd2ad9af1ef087ede6b9f66b54d&chksm=ec1cea19db6b630f2827a7ada6ee99580757aa4cc7073a65fa93da05dd69bdaa663d371a820b&mpshare=1&scene=1&srcid=0412Hl8FPlGwSrdm997YN7Dz&sharer_sharetime=1618188068550&sharer_shareid=03101a931987a40bb1c69d01fec93b52&exportkey=AYrpu%2BUT5zplHT%2FW0p0Mqwg%3D&pass_ticket=Mwvmf2YSuj%2F34nnYV%2FSCB5JR%2BHIey1PpZDZaGNT8PdnSYVXiANNAEYIi3b0g1%2BFV&wx_header=0#rd
        #x = self.pos_block(pos_embed, h, w)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) #memory:[400, 2, 256]
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed) #hs: [6, 100, 2, 256]
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
            #[6, 2, 100, 256], [2,256,20,20]

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.peg = PEGThree(256)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        #src.shape  [400, 2, 256] mask: None src_key_padding_mask: [2,400] pos: [400, 2, 256]
        i = 0
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            if i ==0 :
                #print("Will add PEG result output shape ", output.shape)  #[400, 12, 256]
                temp = output.permute(1, 2, 0)
                temp = self.peg(temp, 20, 20)
                output = output + temp
            i = i + 1


        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()   #d_model: 256
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.self_attnPrj = ProjAttention(dim = d_model, proj_kernel = 3, kv_proj_stride = 2, heads = nhead, dim_head = 64, dropout = 0.)

        self.leFF = LeFF(dim=d_model, scale=4, depth_kernel=3, h=20, w=20)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        #will be used here.
        #src [400, 2, 256] src_mask: None src_key_padding_mask: [2,400] pos: [400,2,256]
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,  #q,k: [400, 2, 256]
                              key_padding_mask=src_key_padding_mask)[0]
        srcProj = src2.permute(1, 2, 0)  # Now 2,256,400
        B, C, HW = srcProj.shape
        #print("srcPrj shape ", B, " ", C, " ", HW)
        H = 20
        W = 20
        srcProj = srcProj.view(B, C, H, W)
        projAtten = self.self_attnPrj(srcProj)  #
        projAtten = projAtten.view(B, C, -1).permute(2,0,1)

        #src2: [400, 6, 256]
        src = src + self.dropout1(projAtten)
        src = self.norm1(src)  #src: [400,batchSize,256]
        #should be batch, hw, c
        src2 = self.leFF(src.permute(1, 0, 2)).permute(1, 0, 2)
        #src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  #feedforward: src2: [400,2,256]

        src = src + self.dropout2(src2)  #src: [400,2,256]

        src = self.norm2(src)   #src: [400, 2, 256]
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.self_attnPrj = ProjAttention(dim = d_model, proj_kernel = 3, kv_proj_stride = 2, heads = nhead, dim_head = 64, dropout = 0.)
        self.leFF = LeFF(dim=d_model, scale=4, depth_kernel=3, h=10, w=10)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, #[400,2,256]
                     query_pos: Optional[Tensor] = None): #[100, 2, 256]
        q = k = self.with_pos_embed(tgt, query_pos)   #q,k: [100, 2, 256]
        #tgt: [100, 2, 256], memory: [400, 2, 256] memory_mask: None tgt_key_padding_mask:None memory_key_padding_mask: [2,400]
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0] #tgt2: [100,2,256]

        tgtProj = tgt2.permute(1, 2, 0)  # Now 2,256,400
        B, C, HW = tgtProj.shape
        #print("srcPrj shape ", B, " ", C, " ", HW)
        H = 10
        W = 10
        tgtProj = tgtProj.view(B, C, H, W)
        projAtten = self.self_attnPrj(tgtProj)  #
        projAtten = projAtten.view(B, C, -1).permute(2,0,1)

        #src2: [400, 6, 256]
        tgt = tgt + self.dropout1(projAtten)

        #tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)  #tgt: [100,2,256]
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  #100,2,256
                                   key=self.with_pos_embed(memory, pos),       #400,2,256
                                   value=memory, attn_mask=memory_mask,        #400,2,256
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)  #tgt2: [100,2,256]
        tgt = self.norm2(tgt)

        tgt2 = self.leFF(tgt.permute(1, 0, 2)).permute(1, 0, 2)
        #tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))  #tgt2: 100,2,256

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
