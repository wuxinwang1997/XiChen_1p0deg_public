"""单观测源 (per-obs) 数据同化网络 ``XiChenDA``。

本模块实现 ``XiChenDA`` 类, 是 cascade DA Solver 使用的 per-obs DA 模型:
- 输入: ``(xb, grad)``, 其中 ``xb`` 是背景场, ``grad`` 是 ``∂J/∂xb``
  (L2 归一化后的 VarCost 梯度);
- 输出: ``(xa, log_var)``, 其中 ``xa`` 是该 obs 源的分析增量, ``log_var`` 是
  对数方差(由 softplus 双侧截断到 ``[-10, 10]``)。
- 网络结构: Swin-V2 encoder -> cross-attn latent (与 xb 拼接的 condition) ->
  Swin-V2 decoder -> patch 维上采样回 ``(B, V, H, W)``。

主要设计:
- **Encoder 阶段**: 对 ``grad`` 做 patch embedding + 多个 SwinLayer (无
  condition), 输出 ``enc_fpn`` 融合的多层特征, 通过残差加回;
- **Latent 阶段**: 多个 SwinLayer ``condition=True``(以 ``xb`` patch_embed
  作为 cross-attn 条件), 同样 FPN 融合;
- **Decoder 阶段**: SwinLayer + FPN 还原成 patch 网格, 再 ``up_forward``
  决定是 ``ConvTranspose2d``(奇数 ``img_size[0]``)还是 ``Linear + rearrange``
  还原到 ``(B, V, H, W)``;
- **log_var 双侧截断**: ``log_var = softplus(log_var + 10) - 10`` 把下界 clamp
  到 ``-10``;``10 - softplus(10 - log_var)`` 把上界 clamp 到 ``10``, 与
  ``src.models.forecast.arch.XiChenForecast`` 保持一致。
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

from xichen.layers.patch_embed import PatchEmbed
from xichen.layers.swin_attn import SwinLayer

class XiChenDA(nn.Module):
    """per-obs 数据同化网络 (cascade DA Solver 使用)。

    以 ``(xb, grad)`` 为输入 (其中 ``grad = ∂J/∂xb`` 已经过 L2 归一化),
    通过 patch embedding + Swin-V2 encoder + Swin-V2 latent (cross-attn with
    ``xb``) + Swin-V2 decoder 产生 patch 网格表示, 再上采样回 ``(B, V, H, W)``
    得到 ``(xa, log_var)``。

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        num_blocks (int): number of fno blocks
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        double_skip (bool): whether to use residual twice
    """

    def __init__(
        self,
        default_vars,
        img_size=[181, 360],
        window_size=[6, 12],
        patch_size=[6, 5],
        patch_stride=[5, 5],
        embed_dim=768,
        num_heads=12,
        encoder_depths=[2, 2, 2],
        latent_depths=[4, 4, 4],
        decoder_depths=[2, 2, 2],
        mlp_ratio=4,
        drop_path=0.2,
        drop_rate=0.2,
        attn_drop=0.,
    ):
        """初始化 ``XiChenDA`` 网络, 构造 patch_embed + 3 段 SwinLayer + FPN。

        三段结构:
            - ``encoder_layers`` (无 condition, depth = ``encoder_depths``):
              对 ``grad`` 做 Swin 编码;
            - ``latent_layers`` (``condition=True``, depth = ``latent_depths``):
              以 ``xb`` 为 cross-attn 条件, 融合背景信息;
            - ``decoder_layers`` (无 condition, depth = ``decoder_depths``):
              还原 patch 网格。

        ``final`` 与 ``log_var`` 上采样分支按 ``img_size[0]`` 奇偶选择:
            - 奇数: ``ConvTranspose2d(embed_dim, V, kernel=patch_size, stride=patch_stride)``;
            - 偶数: ``Linear(embed_dim, V * p1 * p2) + rearrange``。

        Args:
            default_vars (list[str]): 状态变量名列表(69 通道), 用于 ``var_map``。
            img_size (list[int]): 输入网格尺寸, 默认 ``[181, 360]``。
            window_size (list[int]): Swin 窗口大小, 默认 ``[6, 12]``。
            patch_size (list[int]): patch 卷积核, 默认 ``[5, 4]``。
            patch_stride (list[int]): patch 卷积步长, 默认 ``[4, 4]``。
            embed_dim (int): 隐向量维度, 默认 768。
            num_heads (int): 注意力头数, 默认 12。
            encoder_depths (list[int]): encoder 每段 SwinBlock 数, 默认 ``[2, 2, 2]``。
            latent_depths (list[int]): latent 每段 SwinBlock 数, 默认 ``[4, 4, 4]``。
            decoder_depths (list[int]): decoder 每段 SwinBlock 数, 默认 ``[2, 2, 2]``。
            mlp_ratio (float): FFN 隐层倍率, 默认 4。
            drop_path (float): stochastic depth, 默认 0.2。
            drop_rate (float): dropout 率, 默认 0.2。
            attn_drop (float): 注意力 dropout, 默认 0。
        """
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.c = len(self.default_vars)
        self.h = self.img_size[0] // patch_stride[0]
        self.w = self.img_size[1] // patch_stride[1]
        self.embed_dim = embed_dim
        self.num_enc_layers = len(encoder_depths)
        self.num_latent_layers = len(latent_depths)
        self.num_dec_layers = len(decoder_depths)
        self.feat_size = [self.h, self.w]

        # variable tokenization: separate embedding layer for each input variable
        self.var_map = self.create_var_map()
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            patch_stride=patch_stride,
            in_chans=self.c, 
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        self.patch_embed_grad = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            patch_stride=patch_stride,
            in_chans=self.c, 
            embed_dim=embed_dim
        )

        # --------------------------------------------------------------------------

        encoder_layers = []
        for i in range(self.num_enc_layers):
            layer = SwinLayer(
                embed_dim,
                self.feat_size,
                window_size,
                depth=encoder_depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                condition=False,
            )
            encoder_layers.append(layer)
            self.add_module(f"enc_norm{i}", norm_layer(embed_dim, eps=1e-6))

        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        self.enc_fpn = nn.Sequential(
            nn.Linear(embed_dim * self.num_enc_layers, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        latent_layers = []
        for i in range(self.num_latent_layers):
            layer = SwinLayer(
                embed_dim,
                self.feat_size,
                window_size,
                depth=latent_depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                condition=True,
            )
            latent_layers.append(layer)
            self.add_module(f"latent_norm{i}", norm_layer(embed_dim, eps=1e-6))

        self.latent_layers = nn.ModuleList(latent_layers)

        self.latent_fpn = nn.Sequential(
            nn.Linear(embed_dim * self.num_latent_layers, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        decoder_layers = []
        for i in range(self.num_dec_layers):
            layer = SwinLayer(
                embed_dim,
                self.feat_size,
                window_size,
                depth=decoder_depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                condition=False,
            )
            decoder_layers.append(layer)
            self.add_module(f"dec_norm{i}", norm_layer(embed_dim, eps=1e-6))

        self.decoder_layers = nn.ModuleList(decoder_layers)

        self.dec_fpn = nn.Sequential(
            nn.Linear(embed_dim * self.num_dec_layers, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.out_norm = norm_layer(embed_dim, eps=1e-6)
        # --------------------------------------------------------------------------

        # prediction head
        if self.img_size[0] % 2 == 1:
            self.final = nn.ConvTranspose2d(
                in_channels=embed_dim, 
                out_channels=len(default_vars),
                kernel_size=patch_size, 
                stride=patch_stride, 
                bias=False
            )
            self.log_var = nn.ConvTranspose2d(
                in_channels=embed_dim, 
                out_channels=len(default_vars),
                kernel_size=patch_size, 
                stride=patch_stride, 
                bias=False
            )
        else:
            self.final = nn.Linear(
                embed_dim, 
                len(default_vars) * patch_size[-1] * patch_size[-2], 
                bias=False
            )
            self.log_var = nn.Linear(
                embed_dim, 
                len(default_vars) * patch_size[-1] * patch_size[-2], 
                bias=False
            )

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        """初始化 PatchEmbed + 遍历 apply ``_init_weights``。

        - 对 ``patch_embed.proj.weight`` 单独做 ``trunc_normal_(std=0.02)``;
        - 其余 ``nn.Linear`` / ``nn.LayerNorm`` 由 ``_init_weights`` 统一初始化。
        """
        # token embedding layer
        w = self.patch_embed.proj.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """递归初始化 ``nn.Linear`` (trunc_normal std=0.02) 与 ``nn.LayerNorm`` (常数)。"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_map(self):
        """构造 ``{var_name: idx}`` 字典, 用于 ``get_var_ids`` 把变量名转索引。"""
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_map

    def get_var_ids(self, vars, device):
        """把变量名列表转成 ``torch.Tensor`` 索引, 移到 ``device``。

        Args:
            vars (list[str]): 变量名列表, 元素需在 ``self.var_map`` 中。
            device (torch.device): 输出张量所在设备。

        Returns:
            Tensor: 形状 ``(len(vars),)`` 的 int64 索引张量。
        """
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def forward_encoder(self, xb: torch.Tensor, grad: torch.Tensor, use_checkpoint=False):
        """Encoder 阶段: 对 ``grad`` 做 Swin 编码, 返回 ``(xb, grad)``。

        流程:
            1. ``patch_embed(xb)`` token 化 xb, ``patch_embed_grad(grad)`` 用
               独立的 patch 投影 (不与 xb 共享权重) token 化 grad;
            2. ``encoder_layers`` (无 condition) 逐层对 ``grad`` 编码;
            3. 收集每层 ``enc_norm{i}`` 输出, 拼接后经 ``enc_fpn`` 融合;
            4. 加回 ``grad`` 残差, 与 token 化的 ``xb`` 一起返回。

        Args:
            xb (Tensor): 背景场, 形状 ``(B, V, H, W)``。
            grad (Tensor): ``∂J/∂xb`` (L2 归一化), 形状 ``(B, V, H, W)``。
            use_checkpoint (bool): 是否对 SwinBlock 使用 ``torch.utils.checkpoint``
                以节省显存。

        Returns:
            tuple[Tensor, Tensor]: ``(hb, hg)``, ``hb`` = ``patch_embed(xb)``
                形状 ``(B, L, D)``, ``hg`` = encoder 输出的 ``grad`` 表示。
        """
        # x: `[B, V, H, W]` shape.
        # (B, 8, H, W)
        xb = self.patch_embed(xb)

        grad = self.patch_embed_grad(grad)
        residual = grad

        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.encoder_layers):
            if use_checkpoint:
                grad = checkpoint(blk, grad, use_reentrant=False)
            else:
                grad = blk(grad)
            out = getattr(self, f"enc_norm{i}")(grad)
            outs.append(out)

        if use_checkpoint:
            grad = checkpoint(self.enc_fpn, torch.cat(outs, dim=-1), use_reentrant=False)
        else:
            grad = self.enc_fpn(torch.cat(outs, dim=-1))

        grad = grad + residual

        return xb, grad

    def forward_latent(self, grad: torch.Tensor, xb: torch.Tensor, use_checkpoint=False):
        """Latent 阶段: ``condition=True`` 的 SwinBlock, 以 ``xb`` 为 cross-attn 条件。

        流程:
            1. 逐层 ``latent_layers`` 调 ``blk(grad, xb)`` 把 xb 作为 condition;
            2. 收集每层 ``latent_norm{i}`` 输出, ``latent_fpn`` 融合;
            3. 加回 ``grad`` 残差, 返回融合后的表示。

        Args:
            grad (Tensor): encoder 输出的 ``(B, L, D)`` 梯度表示。
            xb (Tensor): encoder 输出的 ``(B, L, D)`` 背景 patch 表示(作为 condition)。
            use_checkpoint (bool): 是否对 SwinBlock 使用 ``torch.utils.checkpoint``。

        Returns:
            Tensor: latent 融合后的 ``(B, L, D)`` 表示。
        """
        residual = grad

        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.latent_layers):
            if use_checkpoint:
                grad = checkpoint(blk, grad, xb, use_reentrant=False)
            else:
                grad = blk(grad, xb)
            out = getattr(self, f"latent_norm{i}")(grad)
            outs.append(out)

        if use_checkpoint:
            grad = checkpoint(self.latent_fpn, torch.cat(outs, dim=-1), use_reentrant=False)
        else:
            grad = self.latent_fpn(torch.cat(outs, dim=-1))

        grad = grad + residual

        return grad

    def forward_decoder(self, h: torch.Tensor, use_checkpoint=False):
        """Decoder 阶段: SwinBlock + FPN 还原 patch 网格表示, 再 ``out_norm``。

        Args:
            h (Tensor): latent 输出的 ``(B, L, D)`` 表示。
            use_checkpoint (bool): 是否对 SwinBlock 使用 ``torch.utils.checkpoint``。

        Returns:
            Tensor: decoder 输出的 ``(B, L, D)`` 表示, 已经过 ``out_norm``。
        """
        # x: `[B, V, H, W]` shape.
        # (B, 8, H, W)
        residual = h
        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.decoder_layers):
            if use_checkpoint:
                h = checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
            out = getattr(self, f"dec_norm{i}")(h)
            outs.append(out)

        if use_checkpoint:
            h = checkpoint(self.dec_fpn, torch.cat(outs, dim=-1), use_reentrant=False)
        else:
            h = self.dec_fpn(torch.cat(outs, dim=-1))

        h = h + residual

        h = self.out_norm(h)

        return h

    def up_forward(self, x):
        """把 ``(B, L, D)`` patch 网格表示上采样回 ``(B, V, H, W)``。

        按 ``img_size[0]`` 奇偶分两个分支:
            - 奇数: ``ConvTranspose2d(embed_dim, V, kernel=patch_size, stride=patch_stride)``;
            - 偶数: ``Linear(embed_dim, V * p1 * p2)`` + ``rearrange`` 拼回 ``(B, V, H, W)``。

        Args:
            x (Tensor): decoder 输出的 ``(B, L, D)`` 表示, ``L = (H/p) * (W/p)``。

        Returns:
            tuple[Tensor, Tensor]: ``(res, log_var)``, 都是 ``(B, V, H, W)`` 形状,
                分别作为分析增量 ``xa`` 与对数方差 ``log_var``。
        """
        x = x.view(x.size(0), self.h, self.w,-1)
        if self.img_size[0] % 2 == 1:
            res = self.final(x.permute(0, 3, 1, 2))
            log_var = self.log_var(x.permute(0, 3, 1, 2))
            return res, log_var
        else:
            x = self.final(x)
            res = rearrange(
                x,
                "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=self.patch_size[-2],
                p2=self.patch_size[-1],
                h=self.img_size[0] // self.patch_size[-2],
                w=self.img_size[1] // self.patch_size[-1],
            )
            log_var = self.log_var(x)
            log_var = rearrange(
                log_var,
                "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=self.patch_size[-2],
                p2=self.patch_size[-1],
                h=self.img_size[0] // self.patch_size[-2],
                w=self.img_size[1] // self.patch_size[-1],
            )
            return res, log_var

    def forward(self, xb, grad, variables, use_checkpoint=False):
        """前向计算: ``(xb, grad) -> (preds, log_var)``。

        流程:
            1. ``forward_encoder`` 编码 (xb, grad) -> (hb, hg);
            2. ``forward_latent(hg, hb)`` 用 xb 作为 cross-attn 条件融合;
            3. ``forward_decoder`` 还原 patch 表示;
            4. ``up_forward`` 上采样到 ``(B, V, H, W)``;
            5. ``log_var`` 双侧 softplus 截断到 ``[-10, 10]``;
            6. 按 ``variables`` 子集索引, 沿通道维挑选输出。

        Args:
            xb (Tensor): 背景场, 形状 ``(B, C, H, W)``。
            grad (Tensor): ``∂J/∂xb`` 归一化梯度, 形状 ``(B, C, H, W)``, 与 xb 一致。
            variables (list[str] | tuple[str]): 输出变量名子集, 沿通道维挑选输出。
            use_checkpoint (bool): 是否对 SwinBlock / CrossAttention 使用
                ``torch.utils.checkpoint``。默认 False。

        Returns:
            tuple[Tensor, Tensor]: ``(preds, log_var)``, 形状均为 ``(B, V_out, H, W)``
                (V_out = len(variables))。
        """
        hb, hg = self.forward_encoder(xb, grad, use_checkpoint)  # B, L, D

        ha = self.forward_latent(hg, hb, use_checkpoint)

        ha = self.forward_decoder(ha, use_checkpoint)

        preds, log_var = self.up_forward(ha)  # B, L, V*p*p

        # log_var 双侧 softplus 截断到 [-10, 10]:先 clamp 下界(-10 + softplus(log_var + 10)
        # 恒 >= -10),再 clamp 上界(10 - softplus(10 - log_var) 恒 <= 10),
        # 与 src.models.forecast.arch.XiChenForecast 的约定一致
        log_var= -10 + F.softplus(log_var + 10)
        log_var = 10 - F.softplus(10 - log_var)

        out_var_ids = self.get_var_ids(tuple(variables), preds.device)
        preds = preds[:, out_var_ids]
        log_var = log_var[:, out_var_ids]

        return preds, log_var
