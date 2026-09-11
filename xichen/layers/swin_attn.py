"""
Swin-V2 注意力与基础 Block

XiChen 全部模型的骨干层，本模块实现：

- ``window_partition`` / ``window_reverse``：将 2D 特征图划分为
  ``window_size x window_size`` 的窗口（用于窗口注意力）。
- ``WindowAttentionV2``：Swin-V2 余弦注意力 + 连续相对位置偏置
  (continuous-relative-position-bias via MLP)；额外维护一个
  ``logit_scale = log(10)`` 可学习参数，并通过 ``_max_logit`` buffer
  防止 ``logit_scale.exp()`` 在 bf16 下溢出。
- ``WindowCrossAttentionV2``：与 ``WindowAttentionV2`` 同结构，但 Q
  来自条件 ``c``（lead-time / 卫星上下文 / 梯度），K/V 来自主输入 ``x``；
  用于在 latent 序列上注入额外条件。
- ``SwinBlock``：包含 ``WindowAttentionV2`` + 残差 + 可选
  ``WindowCrossAttentionV2`` + GeGLUFFN，并预计算 SW-MSA mask。
- ``SwinLayer``：堆叠 ``depth`` 个 SwinBlock，奇 / 偶层使用窗口移动
  (shift) / 不移动 (no-shift) 交错。

上游依赖：``src/layers/mlp.py`` 提供 ``GeGLUFFN``、``Mlp``；
``einops`` 的 ``rearrange``、``timm`` 的 ``DropPath / trunc_normal_``。
下游调用：被所有 backbone ``arch.py``（forecast / compression /
obsoperator / DA）作为最深层级 transformer block 使用。
"""

from functools import partial, lru_cache
import os
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import collections.abc
from einops import repeat, rearrange
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .mlp import GeGLUFFN, Mlp


def exists(val):
    """``val is not None`` 的简洁别名；常用于可选参数检查。"""
    return val is not None


def window_partition(x, window_size):
    """把 ``(B, H, W, C)`` 的特征图切成 ``(num_windows*B, Wh, Ww, C)``。

    实现要点：

    - 先 ``view`` 把 H/W 维度拆出窗口大小，再 ``permute(0,1,3,2,4,5)`` 把
      ``(B, nH, nW, Wh, Ww, C)`` 中"窗口内"维度提到前面，最后合并
      ``nH * nW`` 得到 ``num_windows``。

    Args:
        x: shape ``(B, H, W, C)``。
        window_size: ``(Wh, Ww)``；要求 ``H % Wh == 0`` 且 ``W % Ww == 0``。

    Returns:
        ``(num_windows * B, Wh, Ww, C)``。
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0],W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """``window_partition`` 的逆操作：把窗口序列还原为 ``(B, H, W, C)``。

    Args:
        windows: shape ``(num_windows*B, Wh, Ww, C)``。
        window_size: ``(Wh, Ww)``。
        H, W: 还原后的特征图高 / 宽。

    Returns:
        ``(B, H, W, C)``。
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttentionV2(nn.Module):
    """Swin-V2 窗口自注意力（cosine attention + CPB）。

    关键设计（与 Swin v1 不同）：

    1. **余弦注意力**（cosine attention）：把 Q / K 在 ``dim=-1`` 上做
       L2 normalize 再点积，避免 ``logit_scale`` 变大时 attention 饱和。
    2. **可学习温度** ``logit_scale = log(10)``，每 head 一个标量；
       通过 ``_max_logit = ln(100)`` 截断防止 ``exp`` 溢出，并额外做
       ``nan → 1`` 与 ``[0.01, 100]`` clamp 双重保险（bf16 友好）。
    3. **连续相对位置偏置**（continuous relative position bias）：
       把 ``(Δh, Δw)`` 先 log-cp 分桶到 ``[-8, 8]``，再用一个小 MLP
       ``(2 → 512 → num_heads)`` 输出每 head 的 bias，最后用
       ``16 * sigmoid`` 限幅。

    Args:
        dim: 输入 / 输出 token 维度。
        window_size: ``(Wh, Ww)``，通常 ``(7, 7)``。
        num_heads: 注意力头数。
        qkv_bias: 是否给 Q / V 加可学习 bias（K 不加）。
        attn_drop: 注意力权重的 dropout。
        proj_drop: 输出投影的 dropout。
        pretrained_window_size: 预训练时的 ``(Wh, Ww)``；``(0, 0)`` 表示
            当前 ``window_size`` 即预训练尺寸，相对坐标按此归一化。
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        pretrained_window_size=[0, 0]
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        # 可学习温度：每 head 一个 log(10)；exp 后约 10
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # 预计算最大 logit 上界：ln(100)，防止 exp 溢出（bf16 安全）
        max_logit_value = float(torch.log(torch.tensor(1. / 0.01)).item()) # ln(100) ≈ 4.6052
        self.register_buffer("_max_logit", torch.tensor(max_logit_value, dtype=torch.float32))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # 构建 (2*Wh-1, 2*Ww-1, 2) 的相对坐标表 + log-cp 归一化
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)

        relative_coords_table = torch.stack(
            torch.meshgrid(
                [relative_coords_h, relative_coords_w], indexing='ij')
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            # 用预训练尺寸归一化；推理时窗口大小变化仍能正确插值
            relative_coords_table[:, :, :,
                                  0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :,
                                  1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        # log-cp：sign(x) * log2(|x| + 1) / log2(8)，将值压缩到 [-1, 1]
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # 构造 (Wh*Ww, Wh*Ww) 的相对位置 index；用整型编码加速 bias 查表
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV 投影（无 bias），通过外部 q_bias / v_bias 单独加 Q 与 V 的 bias
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """窗口自注意力前向。

        Args:
            x: input features with shape of (num_windows*B, N, C).
               输入特征,shape ``(num_windows*B, N, C)``，``N = Wh*Ww``。
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None.
                  SW-MSA 注意力 mask，shape ``(num_windows, Wh*Ww, Wh*Ww)``；
                  元素为 ``0`` 或 ``-inf``，``None`` 表示不做 SW-MSA。

        Returns:
            shape ``(num_windows*B, N, C)``，与 ``x`` 同形。
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            # K 的 bias 设为 0、不学习；只对 Q 和 V 加 bias
            qkv_bias = torch.cat((self.q_bias,
                                  torch.zeros_like(self.v_bias, requires_grad=False),
                                  self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention：Q / K 先 L2 normalize 再点积
        attn = (F.normalize(q, dim=-1) @
                F.normalize(k, dim=-1).transpose(-2, -1))
        # 可学习温度 clamp：上限为 _max_logit（防止 exp 溢出），再 exp
        logit_scale = self.logit_scale.float()
        max_logit = self._max_logit.to(logit_scale.device, logit_scale.dtype)
        logit_scale = torch.clamp(logit_scale, max=max_logit)
        logit_scale = logit_scale.exp()
        # cosine attention 双向 clamp 到 [exp(-4.6)=0.01, exp(4.6)=100.0],
        # 防止 softmax 在极值处梯度消失。下限 0.01 是 post-exp clamp,
        # 在 ``_max_logit`` 上限之外额外提供数值安全。
        logit_scale = torch.where(
            torch.isnan(logit_scale),
            torch.ones_like(logit_scale),
            logit_scale
        )
        logit_scale = torch.clamp(logit_scale, min=0.01, max=100.0)
        attn = attn * logit_scale

        # 连续相对位置偏置：cpb_mlp(relative_coords_table) -> bias
        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table.to(x)).view(-1, self.num_heads)
        # 用预计算的 relative_position_index 查表得到 (Wh*Ww, Wh*Ww, nH)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # 16 * sigmoid 把 bias 限幅到 [0, 16]
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = mask.to(x)
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x


class WindowCrossAttentionV2(nn.Module):
    """Swin-V2 窗口交叉注意力。

    与 ``WindowAttentionV2`` 的差异：

    - Q 来自条件输入 ``c``（例如 lead-time embedding、卫星上下文、
      梯度等），K / V 来自主输入 ``x``；通过 ``self.q`` / ``self.kv``
      两个独立投影实现。
    - 余弦注意力 + CPB + ``logit_scale`` 数值保护与 ``WindowAttentionV2``
      完全一致（共用同一套算法约定）。

    Args:
        dim: 主输入 / 输出 token 维度。
        window_size: ``(Wh, Ww)``。
        num_heads: 注意力头数。
        qkv_bias: 是否给 Q / V 加 bias。
        attn_drop: 注意力权重的 dropout。
        proj_drop: 输出投影的 dropout。
        pretrained_window_size: 预训练时的窗口大小。
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        pretrained_window_size=[0, 0]
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        max_logit_value = float(torch.log(torch.tensor(1. / 0.01)).item()) # ln(100) ≈ 4.6052
        self.register_buffer("_max_logit", torch.tensor(max_logit_value, dtype=torch.float32))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)

        relative_coords_table = torch.stack(
            torch.meshgrid(
                [relative_coords_h, relative_coords_w], indexing='ij')
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :,
                                  0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :,
                                  1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # 交叉：Q 来自 c，K/V 来自 x；分别用 self.q / self.kv 投影
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.q = nn.Linear(dim, dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, c, mask=None):
        """窗口交叉注意力前向。

        Args:
            x: input features with shape of (num_windows*B, N, C).
               主输入 token；shape ``(num_windows*B, N, C)``。
            c: 条件输入 token；shape 同 ``x``，通常来自 lead-time 等嵌入
                通过 condition_module 投影后的结果。
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None.
                  SW-MSA mask；可为 ``None``。

        Returns:
            shape ``(num_windows*B, N, C)``，与 ``x`` 同形。
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            # NOTE: kv_bias 被计算但未用于 K/V —— 有意保持与训练侧一致。
            # 训练代码 /workspace/DCU/XiChenPaper/src/layers/swin_attn.py 的交叉注意力
            # 前向存在同一实现（bias=qkv_bias 恒为 None），即 v_bias 在训练与推理均未生效；
            # 发布 checkpoint 与此行为自洽。请勿在此处"修复"为传 kv_bias，否则会改变
            # 模型推理数值、破坏与已发布权重的一致性。如需对齐用途请先与训练端沟通。
            kv_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False),
                                  self.v_bias))
        kv = F.linear(input=x, weight=self.kv.weight, bias=qkv_bias)
        q = F.linear(input=c, weight=self.q.weight, bias=self.q_bias)

        kv = kv.reshape(B_, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        k, v = kv[0], kv[1]
        q = q.reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @
                F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = self.logit_scale.float()
        max_logit = self._max_logit.to(logit_scale.device, logit_scale.dtype)
        logit_scale = torch.clamp(logit_scale, max=max_logit)
        logit_scale = logit_scale.exp()
        logit_scale = torch.where(
            torch.isnan(logit_scale),
            torch.ones_like(logit_scale),
            logit_scale
        )
        logit_scale = torch.clamp(logit_scale, min=0.01, max=100.0)
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table.to(x)).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = mask.to(x)
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    """单个 Swin-V2 block（窗口自注意力 + 可选条件交叉 + GeGLUFFN）。

    主要步骤：

    1. ``norm1 → swin_attn → drop_path``，与残差相加。
    2. 若 ``condition=True``，对 ``condition`` 做一次 ``condition_module``
       变换得到 ``c``，再 ``norm3 → swin_crossattn`` 加到主干。
    3. ``norm2 → GeGLUFFN → drop_path``，与残差相加。

    当 ``shift_size > 0`` 时，构造 SW-MSA mask 并以 buffer 形式缓存，
    forward 时用 ``torch.roll`` 做 cyclic shift。

    Args:
        dim: token 维度。
        num_heads: 注意力头数。
        input_size: 输入特征图 ``(H, W)``（已除 patch 数量）。
        window_size: 窗口大小 ``(Wh, Ww)``。
        shift_size: SW-MSA 偏移量 ``(sh, sw)``；``(0, 0)`` 表示 W-MSA。
        mask_type: SW-MSA mask 切分方式，``'h'`` / ``'w'`` / 其他。
        mlp_ratio: FFN 隐藏层相对 ``dim`` 的倍数（默认 4）。
        qkv_bias: 是否给 Q/V 加 bias。
        drop: dropout 概率。
        drop_path: 随机深度 drop probability。
        attn_drop: 注意力权重的 dropout。
        norm_layer: 归一化层类，默认 ``nn.LayerNorm``。
        condition: 是否启用条件交叉分支。
    """

    def __init__(
        self,
        dim,
        num_heads,
        input_size,
        window_size=7,
        shift_size=0,
        mask_type='h',
        mlp_ratio=4.,
        qkv_bias=True,
        drop=0.,
        drop_path=0.,
        attn_drop=0.,
        norm_layer=nn.LayerNorm,
        condition=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_size = input_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 若输入尺寸 <= 窗口尺寸则强制不使用 shift，并把窗口缩小为输入尺寸
        if self.input_size[0] <= self.window_size[0]:
            self.shift_size[0] = 0
            self.window_size[0] = self.input_size[0]

        if self.input_size[1] <= self.window_size[1]:
            self.shift_size[1] = 0
            self.window_size[1] = self.input_size[1]

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttentionV2(dim,
                                      window_size=self.window_size,
                                      num_heads=num_heads,
                                      qkv_bias=qkv_bias,
                                      attn_drop=attn_drop,
                                      proj_drop=drop)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.condition = condition
        if self.condition:
            # 条件 MLP：把外部条件 token 投影到主维度空间，再交给 cross-attention
            self.condition_module = nn.Sequential(
                nn.Linear(dim, 2 * dim, bias=True),
                nn.GELU(),
                nn.Linear(2 * dim, dim, bias=True),
            )
            self.cross_attn = WindowCrossAttentionV2(dim,
                                                    window_size=self.window_size,
                                                    num_heads=num_heads,
                                                    qkv_bias=qkv_bias,
                                                    attn_drop=attn_drop,
                                                    proj_drop=drop)
            self.norm3 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GeGLUFFN(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        if max(self.shift_size) > 0:
            # 预计算 SW-MSA 注意力 mask
            H, W = self.input_size
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    if mask_type == 'h':
                        img_mask[:, h, :, :] = cnt
                    elif mask_type == 'w':
                        img_mask[:, :, w, :] = cnt
                    else:
                        img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 把 SW-MSA mask 注册为 buffer，避免每次 forward 重建
        self.register_buffer("attn_mask", attn_mask)

    def swin_attn(self, x):
        """执行 W-MSA / SW-MSA 自注意力：循环 shift → 窗口划分 → 注意力 → 反 shift。"""
        H, W = self.input_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # cyclic shift
        if max(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if max(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        return x

    def swin_crossattn(self, x, c):
        """执行窗口交叉注意力：x 作为 K/V 主输入，c 作为 Q 条件。"""
        H, W = self.input_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        c = c.view(B, H, W, C)
        # cyclic shift
        if max(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            shifted_c = torch.roll(c, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x
            shifted_c = c

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        # partition windows
        # nW*B, window_size, window_size, C
        c_windows = window_partition(shifted_c, self.window_size)
        # nW*B, window_size*window_size, C
        c_windows = c_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.cross_attn(x_windows, c_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if max(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        return x

    def forward(self, x, condition):
        """Block 前向：自注意力 → 可选条件交叉 → GeGLUFFN。

        Args:
            x: 主输入 token，shape ``(B, H*W, C)``。
            condition: 条件 token，shape ``(B, H*W, C)``；仅在
                ``self.condition=True`` 时被使用。

        Returns:
            与 ``x`` 同形的输出 token。
        """
        x = x + self.drop_path1(self.swin_attn(self.norm1(x)))
        if self.condition:
            x = x + self.swin_crossattn(self.norm3(x), self.condition_module(condition))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class SwinLayer(nn.Module):
    """Swin-V2 layer：堆叠 ``depth`` 个 SwinBlock。

    奇 / 偶 block 交替使用 W-MSA（``shift_size=(0, 0)``）与 SW-MSA
    （``shift_size=(Wh//2, Ww//2)``），与原始 Swin 论文一致。

    Args:
        embed_dim: token 维度。
        input_size: ``(H, W)`` patch 数。
        window_size: ``(Wh, Ww)``。
        depth: block 数（默认 4）。
        num_heads: 注意力头数。
        mlp_ratio: FFN 隐藏层倍数。
        drop: dropout 概率。
        drop_path: 随机深度概率。
        attn_drop: 注意力权重 dropout。
        norm_layer: 归一化层类。
        condition: 是否启用条件交叉分支。
    """

    def __init__(
        self,
        embed_dim,
        input_size,
        window_size,
        depth=4,
        num_heads=8,
        mlp_ratio=4.,
        drop=0.,
        drop_path=0.,
        attn_drop=0.,
        norm_layer=nn.LayerNorm,
        condition=False,
    ):
        super().__init__()

        self.depth = depth
        self.input_size = input_size

        self.blocks = nn.ModuleList()

        for i in range(depth):
            blk = SwinBlock(
                dim=embed_dim,
                input_size=input_size,
                num_heads=num_heads,
                window_size=window_size,
                # 偶数 block 用 W-MSA；奇数用 SW-MSA（half-window shift）
                shift_size=[0, 0] if (i % 2 == 0) else [window_size[0] // 2, window_size[1] // 2],
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                condition=condition,
            )
            self.blocks.append(blk)

    def forward(self, h, temb=None):
        """依次通过 ``self.blocks`` 的每个 SwinBlock。

        Args:
            h: 主输入 token，shape ``(B, H*W, C)``。
            temb: 可选条件 token，传给每个 block 作为 ``condition``；
                即便 block 未启用 condition，多传也无害。

        Returns:
            与 ``h`` 同形的输出 token。
        """
        for i, blk in enumerate(self.blocks):
            h = blk(h, temb)
        return h