"""
MLP / FFN 变体集合

本模块提供 Swin-V2 骨干网络所需的前馈子层（Feed-Forward Network, FFN），
包含三种实现：

- ``GEGLU``：Gated GELU 激活单元（无参数），将输入在最后一维切分为两半，
  对后半施加 GELU 后与前半逐元素相乘。
- ``GeGLUFFN``：完整的 GeGLU FFN（fc1 -> GEGLU -> dropout -> fc2 -> dropout），
  是 XiChen 中所有 SwinBlock 默认使用的 FFN 实现。
- ``Mlp``：经典的两层 GELU MLP（与原始 BERT / ViT 保持一致），
  在 ``self.attn`` 之后不使用 dropout，仅在输出前 dropout 一次，
  用于非 Swin 的轻量模型路径。

上游依赖：``src/layers/swin_attn.py`` 在每个 ``SwinBlock`` 内实例化 ``GeGLUFFN``；
``src/models/*`` 中的若干自编码器路径会用到 ``Mlp``。
下游调用：被各 backbone block 的 ``forward`` 调用，无反向依赖。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GEGLU(nn.Module):
    """Gated GELU 激活单元。

    将 ``x`` 在最后一维二等分，得到 ``x_part`` 和 ``gate_part``，输出
    ``x_part * gelu(gate_part)``。这是 Shazeer (2020) "Gated Linear Units"
    与 GELU 激活的结合，常用于替代标准 FFN 中的激活函数。

    该算子没有可学习参数，因此可直接作为 ``nn.Module`` 包裹以保持
    ``nn.Sequential`` 接口一致。

    Args:
        x: 输入张量，最后一维必须为偶数。

    Returns:
        与 ``x`` 形状相同的张量（最后一维保持不变）。
    """

    def forward(self, x):
        # 沿特征维切两半：前一半是直通信号，后一半是门控信号
        x, gate = x.chunk(2, dim=-1)
        # 门控：gelu(gate) * x 等价于一个可学习的、门控的"通路"
        return F.gelu(gate) * x


class GeGLUFFN(nn.Module):
    """GeGLU 形式的前馈网络（FFN）。

    设计要点：

    - ``fc1`` 输出 ``inner_dim * 2``，由 ``GEGLU`` 切两半并门控；
      因此等价于"宽为 ``inner_dim``"的隐藏层。
    - ``inner_dim = int(hidden_features * 2/3)``，这是 LLaMA / PaLM 等
      现代 transformer 中常见的 "2/3 缩放"，用来补偿门控带来的参数量。
    - 两个 ``fc`` 都不使用 ``bias``，与 LLaMA 风格一致。
    - 前后各接一次 ``Dropout``。

    Args:
        in_features: 输入特征维度。
        hidden_features: 隐藏层维度（门控前）；若为 ``None``，则取 ``in_features``。
        out_features: 输出维度；若为 ``None``，则取 ``in_features``。
        drop: dropout 概率，作用于激活后与输出前两个位置。

    Note:
        这是 ``SwinBlock`` 默认的 FFN 实现。在 NPU bf16 训练下，
        ``inner_dim * 2`` 的中间张量会显著占用显存，是 batch size 的主要约束。
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 2/3 缩放：与 LLaMA/PaLM 一致；GEGLU 切两半后相当于实际宽度为 inner_dim
        inner_dim = int(hidden_features * (2 / 3))
        # fc1 输出 *2 以便 GEGLU 切两半
        self.fc1 = nn.Linear(in_features, inner_dim * 2, bias=False)
        self.act = GEGLU()
        self.fc2 = nn.Linear(inner_dim, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp(nn.Module):
    """经典的两层 GELU MLP（BERT / ViT 风格）。

    与 ``GeGLUFFN`` 的关键差异：

    - 没有门控，使用普通 ``GELU`` 激活。
    - ``fc1`` 使用默认 bias，``fc2`` 的 bias 可由 ``bias`` 控制。
    - dropout 应用在 ``fc2`` 之后 (与 BERT 一致) ；激活后无独立 dropout。

    Args:
        in_features: 输入维度。
        hidden_features: 隐藏层维度；若为 ``None``，则取 ``in_features``。
        out_features: 输出维度；若为 ``None``，则取 ``in_features``。
        act_layer: 激活层类，默认 ``nn.GELU``。
        drop: dropout 概率（仅作用于输出前）。
        bias: ``fc2`` 是否使用 bias。

    Note:
        该实现与 HuggingFace BERT 的两层 GELU MLP 保持一致,激活后无 dropout,
        输出前 dropout 一次,方便与上游 checkpoint 对齐。
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.,
        bias=True
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x