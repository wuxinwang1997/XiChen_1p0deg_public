"""XiChen 观测算子 (Observation Operator) 模型。

本模块实现 :class:`XiChenObsOp`，基于 Swin-V2 Transformer 主干，将 ERA5 模式状态 (state)
与卫星辅助观测场 (cos/sin zenith、azimuth、scan/fov/orbit、satellite_height 等) 共同映射
到卫星辐射计观测的亮度温度 (brightness temperatures) 及其观测误差的对数方差
(log-variance)，即 H(x) 与 R 的可学习版本。

典型数据流 (sat_mask 同时进入 encoder 输入与 post-decoder 掩码)::

    state [B, Vc, H, W]  ─► encoder ──────────────────────┐
                                                            │
    sat   [B, Vs, H, W]  ─► concat ─► sat_encoder ─────┐   ├─► latent (cross-attn) ─► decoder ─► (tmbrs, log_var) ──┐
    sat_mask [B, 1, H, W] ─► (+1 ch)                  │   │                                                   │ × sat_mask
                                                        └───┘                                                   ▼

本模块对应 3.3 节任务家族 (paper §3.3) 中的 obsoperator，由 ``src.pipeline.obsoperator.trainer`` 调用，
并通过 ``XiChenObsOp`` 输出去驱动同化求解器 (cascade / multimodal DA) 中的 ``H(x)`` 算子。

注:
- ``patch_embed_sat`` 的 ``in_chans = len(in_sat_vars) + 1``，额外 +1 通道用于拼入 ``sat_mask``。
- ``log_var`` 通过 softplus 平滑截断到 ``[-10, 10]`` 区间，避免数值下溢/上溢。
- 输入图像高度 ``img_size[0]`` 为奇数时使用 ConvTranspose2d 上采样，偶数时使用
  Linear + einops ``rearrange`` 实现像素重组。
"""
from functools import partial, lru_cache
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.layers import trunc_normal_
import collections.abc
from einops import rearrange
import torch.nn.functional as F

from xichen.layers.patch_embed import PatchEmbed
from xichen.layers.swin_attn import SwinLayer

class XiChenObsOp(nn.Module):
    """XiChen 卫星观测算子骨架，基于 Swin-V2 Transformer。

    将模式状态变量与卫星辅助观测场联合编码，经由编码器 / 瓶颈 (cross-attention) / 解码器
    三个阶段，输出卫星亮温 (out_sat) 与对应的观测误差 log-variance (log_var)。最终输出
    与 ``sat_mask`` 逐元素相乘，仅保留有效观测位置的预测。

    算法契约 (H(x), R)::

        out_sat, log_var, tgt_sat = model(state, sat, sat_mask)

    其中 ``tgt_sat`` 是从 ``sat`` 中按 ``out_sat_vars`` 抽取并乘上 ``sat_mask`` 的目标值，
    用于训练时的损失计算。

    Note:
        与 ``XiChenAutoEncoder`` 不同,本类不暴露 ``compress`` / ``decompress`` 命名方法;
        推理入口统一为 ``forward(state, sat, sat_mask)``（见下方算法契约）。

    Args:
        default_vars (list): 训练使用的默认模式状态变量列表（用于状态 token 化）。
        all_sat_vars (list): 所有可用的卫星观测变量列表（用于辅助场 token 化）。
        in_sat_vars (list): 作为条件输入的卫星变量子集。
        out_sat_vars (list): 需要预测输出的卫星变量子集。
        img_size (list): 输入数据空间尺寸 ``[height, width]``，默认 ``[181, 360]``。
        window_size (list): Swin Transformer 的局部窗口大小。
        patch_size (list): Patch Embedding 的切块尺寸。
        patch_stride (list): Patch Embedding 的步长。
        embed_dim (int): Token 嵌入维度。
        num_heads (int): 自注意力头数。
        encoder_depths (list): 编码器各阶段的 Swin Block 层数。
        latent_depths (list): 潜在空间（瓶颈层）各阶段的 Swin Block 层数。
        decoder_depths (list): 解码器各阶段的 Swin Block 层数。
        mlp_ratio (float): FFN 隐藏层相对 ``embed_dim`` 的扩展比例。
        drop_path (float): Stochastic Depth 丢弃率。
        drop_rate (float): Dropout 丢弃率。
        attn_drop (float): 注意力权重丢弃率。

    Attributes:
        img_size (list): 网格尺寸。
        patch_size (list): patch 尺寸。
        default_vars / all_sat_vars / in_sat_vars / out_sat_vars (list): 变量元数据。
        c (int): 默认变量通道数。
        h, w (int): patch 化后特征图空间尺寸。
        embed_dim (int): 嵌入维度。
        num_enc_layers / num_latent_layers / num_dec_layers (int): 各阶段层数。
        feat_size (list): ``[h, w]``。
        var_map / sat_var_map (dict): 变量名 → 索引映射。
        patch_embed / patch_embed_sat (PatchEmbed): 状态 / 卫星辅助场投影。
        encoder_layers / latent_layers / decoder_layers (nn.ModuleList): 三阶段 Swin 层。
        enc_fpn / latent_fpn / dec_fpn (nn.Sequential): 各阶段 FPN 融合 MLP。
        enc_norm{i} / latent_norm{i} / dec_norm{i} (nn.LayerNorm): 各阶段 LayerNorm。
        out_norm (nn.LayerNorm): 解码器末端 LayerNorm。
        final_obs / final_obserr (nn.Module): 上采样预测头（``ConvTranspose2d`` 或
            ``Linear + rearrange``，依据 ``img_size[0]`` 的奇偶性二选一）。
    """

    def __init__(
        self,
        default_vars,
        all_sat_vars,
        in_sat_vars,
        out_sat_vars,
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
        super().__init__()

        # 存储核心配置与变量元数据
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        self.all_sat_vars = all_sat_vars
        self.in_sat_vars = in_sat_vars
        self.out_sat_vars = out_sat_vars
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)  # 固定 eps 的 LayerNorm 构造器

        self.c = len(self.default_vars)  # 默认变量通道数
        # 计算经过 stride 下采样后的特征图空间尺寸
        self.h = self.img_size[0] // patch_stride[0]
        self.w = self.img_size[1] // patch_stride[1]
        self.embed_dim = embed_dim

        # 记录各阶段层数，便于后续 FPN 拼接时动态计算维度
        self.num_enc_layers = len(encoder_depths)
        self.num_latent_layers = len(latent_depths)
        self.num_dec_layers = len(decoder_depths)
        self.feat_size = [self.h, self.w]

        # -------------------------------------------------------
        # 1. 变量 Token 化与 Patch Embedding
        # -------------------------------------------------------
        # 为每个默认状态变量创建独立的嵌入映射（实现类似 Word Embedding 的变量编码）
        self.var_map = self.create_var_map(self.default_vars)

        # 基础状态变量投影
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            patch_stride=patch_stride,
            in_chans=self.c, 
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches


        # 卫星条件变量映射与投影
        # in_chans 额外 +1 通道拼入 ``sat_mask`` (0/1 有效性掩码)
        self.sat_var_map = self.create_var_map(self.all_sat_vars)
        self.patch_embed_sat = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            patch_stride=patch_stride,
            in_chans=len(self.in_sat_vars) + 1, 
            embed_dim=embed_dim
        )

        # -------------------------------------------------------
        # 2. 编码器 (Encoder) 构建
        # -------------------------------------------------------
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
                condition=False,  # 编码器不直接注入卫星条件
            )
            encoder_layers.append(layer)
            # 动态注册归一化层，便于 forward 中按索引获取及 state_dict 管理
            self.add_module(f"enc_norm{i}", norm_layer(embed_dim, eps=1e-6))

        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        # 编码器 FPN 融合：将各阶段输出拼接后压缩回 embed_dim，保留多尺度语义
        self.enc_fpn = nn.Sequential(
            nn.Linear(embed_dim * self.num_enc_layers, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # -------------------------------------------------------
        # 3. 潜在空间/瓶颈层 (Latent/Bottleneck) 构建
        # -------------------------------------------------------
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
                condition=True,  # 瓶颈层注入卫星观测条件信息
            )
            latent_layers.append(layer)
            self.add_module(f"latent_norm{i}", norm_layer(embed_dim, eps=1e-6))

        self.latent_layers = nn.ModuleList(latent_layers)

        self.latent_fpn = nn.Sequential(
            nn.Linear(embed_dim * self.num_latent_layers, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # -------------------------------------------------------
        # 4. 解码器 (Decoder) 构建
        # -------------------------------------------------------
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

        # -------------------------------------------------------
        # 5. 预测头 (Prediction Head)
        # -------------------------------------------------------
        # 根据图像高度奇偶性选择上采样策略，以完美还原原始空间分辨率
        if self.img_size[0] % 2 == 1:
            # 奇数尺寸：使用转置卷积进行像素级上采样
            self.final_obs = nn.ConvTranspose2d(
                in_channels=embed_dim, 
                out_channels=len(out_sat_vars),
                kernel_size=patch_size, 
                stride=patch_stride, 
                bias=False
            )
            self.final_obserr = nn.ConvTranspose2d(
                in_channels=embed_dim, 
                out_channels=len(out_sat_vars),
                kernel_size=patch_size, 
                stride=patch_stride, 
                bias=False
            )
        else:
            self.final_obs = nn.Linear(
                embed_dim, 
                len(out_sat_vars) * patch_size[-1] * patch_size[-2], 
                bias=False
            )
            self.final_obserr = nn.Linear(
                embed_dim, 
                len(out_sat_vars) * patch_size[-1] * patch_size[-2], 
                bias=False
            )

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        """初始化模型参数。

        对状态与卫星两个 ``PatchEmbed`` 的卷积权重使用 ``trunc_normal_``（std=0.02）
        单独初始化；其余 ``nn.Linear`` 与 ``nn.LayerNorm`` 通过 ``self.apply`` 走
        :meth:`_init_weights` 的统一规则。
        """
        # token embedding layer
        w = self.patch_embed.proj.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        wc = self.patch_embed_sat.proj.weight.data
        trunc_normal_(wc.view([wc.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """逐模块初始化权重。

        Args:
            m (nn.Module): 由 ``self.apply`` 回调传入的子模块。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_map(self, all_vars):
        """构造变量名到通道索引的映射字典。

        Args:
            all_vars (list[str]): 变量名列表。

        Returns:
            dict[str, int]: ``{var_name: index}`` 形式的映射。
        """
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in all_vars:
            var_map[var] = idx
            idx += 1
        return var_map

    # @lru_cache(maxsize=None)
    def get_var_ids(self, var_map, vars, device):
        """根据变量名列表查表，得到对应的通道索引张量。

        Args:
            var_map (dict[str, int]): 由 :meth:`create_var_map` 构造的映射。
            vars (Sequence[str]): 待查询的变量名序列（会被转成 ``tuple`` 以便哈希）。
            device (torch.device | str): 目标设备。

        Returns:
            torch.Tensor: ``[len(vars)]`` 的 ``int64`` 张量，已移动到 ``device``。
        """
        ids = np.array([var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def forward_encoder(self, x: torch.Tensor, use_checkpoint=False):
        """状态编码器前向。

        将 ``state [B, V, H, W]`` patch 化后依次通过 ``encoder_layers``，各阶段输出经
        LayerNorm 后由 ``enc_fpn`` 拼接融合，再与 ``residual``（patch_embed 输出）相加。

        Args:
            x (torch.Tensor): ``[B, V, H, W]`` 形状，模式状态变量张量。
            use_checkpoint (bool): 是否对 Swin block 与 FPN 使用
                ``torch.utils.checkpoint`` 节省显存。

        Returns:
            torch.Tensor: ``[B, L, D]`` 的状态编码结果，``L = h * w``，``D = embed_dim``。
        """
        # x: `[B, V, H, W]` shape.
        # tokenize each variable separately
        # (B, 8, H, W)
        h = self.patch_embed(x)
        residual = h

        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.encoder_layers):
            if use_checkpoint:
                h = checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
            out = getattr(self, f"enc_norm{i}")(h)
            outs.append(out)

        if use_checkpoint:
            h = checkpoint(self.enc_fpn, torch.cat(outs, dim=-1), use_reentrant=False)
        else:
            h = self.enc_fpn(torch.cat(outs, dim=-1))

        h = h + residual

        return h

    def forward_encoder_sat(self, sat: torch.Tensor, sat_mask: torch.Tensor, use_checkpoint=False):
        """卫星辅助场编码前向。

        按 ``in_sat_vars`` 子集从 ``sat`` 中挑选对应通道，与 ``sat_mask`` 沿 channel 维
        拼接，再通过 ``patch_embed_sat`` 投影到与状态编码同一 ``embed_dim`` 的 token 序列。

        Args:
            sat (torch.Tensor): ``[B, V_all, H, W]`` 全量卫星观测张量。
            sat_mask (torch.Tensor): ``[B, 1, H, W]`` 有效性掩码（0/1）。
            use_checkpoint (bool): 预留参数（当前未使用，保留以备未来 checkpoint）。

        Returns:
            torch.Tensor: ``[B, L, D]`` 的卫星辅助场编码结果，与状态编码共享同一序列长度。
        """
        # x: `[B, V, H, W]` shape.
        # tokenize each variable separately
        # (B, 8, H, W)
        in_sat_var_ids = self.get_var_ids(self.sat_var_map, tuple(self.in_sat_vars), sat.device)
        in_sat = sat[:, in_sat_var_ids]
        h = self.patch_embed_sat(torch.concat([in_sat, sat_mask], dim=1))

        return h

    def forward_latent(self, h: torch.Tensor, condition: torch.Tensor, use_checkpoint=False):
        """瓶颈层 (cross-attention) 前向。

        ``condition``（卫星辅助场编码）经 ``SwinLayer(condition=True)`` 注入到
        ``latent_layers`` 的每个 block 中；各阶段输出经 LayerNorm 后由 ``latent_fpn``
        拼接融合，再与 ``residual`` 相加。

        Args:
            h (torch.Tensor): ``[B, L, D]`` 状态编码。
            condition (torch.Tensor): ``[B, L, D]`` 卫星辅助场编码，作为 cross-attn 的 KV。
            use_checkpoint (bool): 是否对 Swin block 与 FPN 使用 ``checkpoint``。

        Returns:
            torch.Tensor: ``[B, L, D]`` 融合后的 latent 特征。
        """
        residual = h

        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.latent_layers):
            if use_checkpoint:
                h = checkpoint(blk, h, condition, use_reentrant=False)
            else:
                h = blk(h, condition)
            out = getattr(self, f"latent_norm{i}")(h)
            outs.append(out)

        if use_checkpoint:
            h = checkpoint(self.latent_fpn, torch.cat(outs, dim=-1), use_reentrant=False)
        else:
            h = self.latent_fpn(torch.cat(outs, dim=-1))

        h = h + residual

        return h

    def forward_decoder(self, h: torch.Tensor, use_checkpoint=False):
        """解码器前向。

        将 latent 表示通过 ``decoder_layers``，各阶段输出经 LayerNorm 后由 ``dec_fpn``
        拼接融合，与 ``residual`` 相加后再经 ``out_norm`` 归一化，作为预测头的输入。

        Args:
            h (torch.Tensor): ``[B, L, D]`` 融合后的 latent 表示。
            use_checkpoint (bool): 是否对 Swin block 与 FPN 使用 ``checkpoint``。

        Returns:
            torch.Tensor: ``[B, L, D]`` 解码后的特征。
        """
        # x: `[B, V, H, W]` shape.
        # tokenize each variable separately
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

    def up_forward(self, h):
        """上采样预测头：还原到原始空间分辨率并生成两路输出。

        根据 ``img_size[0]`` 的奇偶性选择不同实现：

        - **奇数高度**：使用 ``ConvTranspose2d``（``final_obs`` / ``final_obserr``）直接
          上采样 ``[B, D, h, w]`` 到 ``[B, C_out, H, W]``。
        - **偶数高度**：使用 ``Linear + einops.rearrange`` 完成 ``patch`` 维度的像素重组
          （``b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)``）。

        Args:
            h (torch.Tensor): ``[B, L, D]`` 解码器输出，``L = h * w``。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(obs, log_var)``，形状均为
            ``[B, len(out_sat_vars), H, W]``。
        """
        h = h.view(h.size(0), self.h, self.w,-1)
        if self.img_size[0] % 2 == 1:
            obs = self.final_obs(h.permute(0, 3, 1, 2))
            log_var = self.final_obserr(h.permute(0, 3, 1, 2))
            return obs, log_var
        else:
            obs = self.final_obs(h)
            obs = rearrange(
                obs,
                "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=self.patch_size[-2],
                p2=self.patch_size[-1],
                h=self.img_size[0] // self.patch_size[-2],
                w=self.img_size[1] // self.patch_size[-1],
            )
            log_var = self.final_obserr(h)
            log_var = rearrange(
                log_var,
                "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=self.patch_size[-2],
                p2=self.patch_size[-1],
                h=self.img_size[0] // self.patch_size[-2],
                w=self.img_size[1] // self.patch_size[-1],
            )
            return obs, log_var

    def forward(self, state, sat, sat_mask, use_checkpoint=False):
        """端到端前向:encoder (state+sat) → latent → decoder → 上采样 → mask。

        Args:
            state (`torch.Tensor`): `[B, C_state, H, W]` ERA5 状态背景。
            sat (`torch.Tensor`): `[B, C_aux, H, W]` 卫星辅助场
                (cos(zenith) / azimuth / scan / fov / orbit / satellite_height)。
            sat_mask (`torch.Tensor`): `[B, C_sat, H, W]` 卫星通道扫描掩码。
            use_checkpoint (bool): SwinBlock 是否启用 gradient checkpointing。

        Returns:
            tuple: ``(out_sat, log_var, tgt_sat)``
            - ``out_sat``: `[B, C_sat, H, W]` 预测亮温;
            - ``log_var``: `[B, C_sat, H, W]` 观测误差 log-variance;
            - ``tgt_sat``: `[B, C_sat, H, W]` 真实亮温 (forward 内提取,便于 loss 计算)。
        """
        # 1) 状态编码：得到 latent-aware 的状态表示
        h_state = self.forward_encoder(state, use_checkpoint=use_checkpoint)  # B, L, D

        # 2) 卫星辅助场编码：得到与状态同维度的 conditioning token
        h_in_sat = self.forward_encoder_sat(sat, sat_mask, use_checkpoint=use_checkpoint)

        # 3) 瓶颈层 cross-attention：将卫星条件注入到状态 latent
        h_sat = self.forward_latent(h_state, h_in_sat, use_checkpoint=use_checkpoint)

        # 4) 解码回原分辨率的特征
        h_out_sat = self.forward_decoder(h_sat, use_checkpoint=use_checkpoint)

        # 5) 上采样到原始 1.0° 网格，得到亮温预测与误差 log-variance
        out_sat, log_var = self.up_forward(h_out_sat)  # B, L, V*p*p

        # 使用 sat_mask 屏蔽无效观测位置
        out_sat = out_sat * sat_mask
        # 双侧 softplus 截断，把 log_var 平滑夹到 [-10, 10]，避免数值爆炸
        log_var= -10 + F.softplus(log_var + 10)
        log_var = 10 - F.softplus(10 - log_var)
        log_var = log_var * sat_mask

        # 按 out_sat_vars 子集抽取对应的目标亮温，同样应用 sat_mask
        out_sat_var_ids = self.get_var_ids(self.sat_var_map, tuple(self.out_sat_vars), sat.device)
        tgt_sat = torch.index_select(sat, dim=1, index=out_sat_var_ids) * sat_mask

        return out_sat, log_var, tgt_sat