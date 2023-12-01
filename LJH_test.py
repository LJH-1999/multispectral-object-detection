# initial_value = 0
# constant = 5
#
# for i in range(8):
#     initial_value = initial_value + constant
# print(initial_value)
import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
from timm.models.vision_transformer import Mlp

from torch.nn import init, Sequential
x = nn.Parameter(torch.zeros(2, 32, 1024, 20, 20))
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:N//2, ...]).reshape(B, N//2, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                           3)  # B(N//2)C -> B(N//2)H(C/H) -> BH(N//2)(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH(N//2H)(C/H) @ BH(C/H)N -> BH(N//2H)N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N//2, C)  # (BH(N//2)N @ BHN(C/H)) -> BH(N//2)(C/H) -> B(N//2)H(C/H) -> B(N//2)C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() #它不会对输入进行任何操作,只是简单地将输入返回。这个层通常被用来在神经网络中连接两个分支,或者在某些情况下,用来作为占位符
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B,N,C = x.shape
        x = x[:, 0:N//2, ...] + self.drop_path(self.attn(self.norm1(x))) #属于是一个残差连接，这个drop_path没有用，可以以后来测试一下
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class CrossViT(nn.Module):
    def __init__(self, dim, n_layer=8, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, vert_anchors=8, horz_anchors=8,
                 drop=0., embd_pdrop=0.1, attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()

        self.n_layer = n_layer
        self.n_embd = dim
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors


        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2*vert_anchors * horz_anchors, self.n_embd))

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

        # transformer
        self.fusion = nn.ModuleList()
        tmp = []
        for i in range(self.n_layer):
            tmp.append(CrossAttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                 drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, has_mlp=has_mlp))
        self.fusion.append(nn.Sequential(*tmp))


    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    x = nn.Parameter(torch.zeros(2, 32, 1024, 20, 20))
    print(x.shape)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        print(x.shape)
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        print(rgb_fea.shape)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)  # dim:(B, C, 8, 8)
        print(rgb_fea.shape)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature b,c n
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature b, c n
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2,
                                                    1).contiguous()  # dim:(B, 2*H*W, C) .contiguous()方法在底层开辟新内存，在内存上tensor是连续的
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2n, C)
        print(x.shape)

        for i in range(self.n_layers-1):
            x = torch.cat([self.fusion[i](x), ir_fea_flat], dim=2)
        x = self.fusion[-1](x)
        print(x.shape)

        # decoder head
        x = self.ln_f(x)  # dim:(B, H*W, C)
        x = x.view(bs, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 3, 1, 2)  # dim:(B, C, H, W)
        print(x.shape)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(x, size=([h, w]), mode='bilinear')
        print(rgb_fea_out.shape)
        return rgb_fea_out

x = nn.Parameter(torch.zeros(2, 32, 1024, 20, 20))

# 创建CrossViT类的实例
model = CrossViT(dim=384, n_layer=8, num_heads=8, mlp_ratio=4,
                     qkv_bias=False, qk_scale=None, vert_anchors=8, horz_anchors=8,
                     drop=0, embd_pdrop=0.1, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                     has_mlp=True)

# 调用forward方法
output = model.forward(x)

print(output.shape)



