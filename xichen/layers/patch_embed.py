"""
图像 -> Patch 嵌入层

将单张 / 批量 2D 输入（如 ERA5 多通道场 ``(B, C, H, W)``）切分为
``patch_size x patch_size`` 的小块，并通过一次 ``Conv2d``（kernel ==
patch_size, stride == patch_stride）线性投影到 ``embed_dim`` 维 token。

上游依赖：``src/layers/pos_embed.py`` 在 ``PatchEmbed`` 输出后注入
``sin-cos`` 位置编码；``src/models/forecast/arch.py``、
``src/models/compression/arch.py`` 等几乎所有 backbone 都会实例化
``PatchEmbed`` 作为最底层。
下游调用：仅被 backbone ``forward`` 调用。
"""

from functools import partial, lru_cache
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
import collections.abc
from einops import repeat, rearrange
import torch.nn.functional as F


def to_2tuple(x):
    """把标量 / 序列统一转为 ``(x, x)`` 二元组。

    本模块内自带的简化版 ``to_2tuple``，避免循环依赖到 ``timm`` 的同名
    工具；若输入本身是可迭代对象则直接返回。

    Args:
        x: 整数或长度为 2 的可迭代对象。

    Returns:
        ``tuple(x, x)``（当 ``x`` 为标量时）或原 ``x``。
    """
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class PatchEmbed(nn.Module):
    """图像到 Patch 嵌入。

    实现要点：

    - ``patch_size`` 与 ``patch_stride`` 都可以是 ``int`` 或 ``(int, int)``；
      ``stride < size`` 时会产生重叠 patch，``stride > size`` 会留空，但
      目前项目内两种用法都用 ``stride == size``（如 ``patch_size=4, stride=4``）。
    - ``num_patches`` 由 ``img_size // patch_stride`` 计算，对应 token 数；
      注释 "# could be dynamic" 提示当输入 H/W 动态时需要外部覆盖。
    - ``forward`` 中先 ``Conv2d`` 投影，再 ``flatten(2).transpose(1, 2)``
      把 ``(B, embed_dim, H/patch, W/patch)`` 变为
      ``(B, num_patches, embed_dim)``，与 transformer 输入约定一致。

    Args:
        img_size: 输入 ``H, W``；默认 ``224``。
        patch_size: 卷积核大小；默认 ``16``。
        patch_stride: 卷积步长；默认 ``16``（与 ``patch_size`` 同）。
        in_chans: 输入通道数；默认 ``3``（实际项目中通常为 69）。
        embed_dim: 输出 token 维度；默认 ``768``。

    Note:
        项目内 1.0° ERA5 (181x360) + patch_size=4 的典型设置下，
        ``num_patches ≈ 45 * 90 = 4050``，是 attention 计算量的主要来源。
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            patch_stride=16,
            in_chans=3,
            embed_dim=768
        ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        # patch 数取决于 stride：stride < size 时相邻 patch 重叠
        self.patch_shape = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])  # could be dynamic
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]  # could be dynamic
        self.patch_size = patch_size
        # 用 Conv2d 同时实现"切块"与"线性投影"，等价于 unfold + Linear
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride)

    def forward(self, x):
        # proj: (B, embed_dim, H/patch, W/patch)
        x = self.proj(x)
        # flatten(2): 合并最后两维到 token 轴；transpose(1,2): (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        return x