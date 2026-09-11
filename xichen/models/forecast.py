"""XiChen 预报模型架构 (forecast / arch.py)。

该模块定义了基于 Swin-Transformer V2 骨干的端到端中短期天气预报网络
:class:`XiChenForecast`,支持将 ERA5 多通道气象场映射到未来时刻的同分辨率
预报场,并以 ``(preds, log_var)`` 的高斯概率输出形式供 CRPS-Gaussian 损失
或对数似然训练使用。

设计要点:
    - 输入 69 通道 (4 表面 + 13 等压面层 × 5 变量),默认 1.0° 网格 ``181x360``。
    - ``PatchEmbed`` 将每个变量通道独立 tokenize,然后堆叠 Swin-V2 窗口自注意
      力层。
    - 三段式 ``encoder -> latent -> decoder``:encoder/decoder 为
      ``condition=False`` 的纯自注意力;latent 段使用 ``WindowCrossAttentionV2``
      注入预报时长 (``lead-time``) 嵌入。
    - 输出头同时给出均值预测 ``preds`` 与对数方差 ``log_var``,后者被 softplus
      双向裁剪到 ``[-10, 10]`` 的稳定区间。

历史:
    该文件已演进多版,``src.models.compression.arch`` 在 encoder/decoder/FPN/head
    结构上与本模块保持一致,但 latent 阶段任务分工不同: forecast 注入 lead-time
    跨注意力条件, compression 注入量化瓶颈。
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

class XiChenForecast(nn.Module):
    """基于 Swin-Transformer V2 的多通道气象预报模型。

    网络结构采用 encoder / latent / decoder 三段式,默认参数
    详见 ``__init__`` 签名（典型值 ``embed_dim=768``、``num_heads=12``、
    ``mlp_ratio=4``、``drop_path=0.2``）。checkpoint 是纯 state_dict（不含
    超参）,构造时必须传入随仓库分发的 config JSON 中的超参（patch_size 等）;
    直接以签名默认值构造后加载发布权重会因形状不符而失败。
    encoder 与 decoder 段不进行条件化,latent 段使用
    :class:`WindowCrossAttentionV2` 注入预报时长嵌入,以适配多步 AR 滚动训练。

    输出契约:
        ``forward`` 始终返回 ``(preds, log_var)`` 二元组,其中 ``log_var`` 由
        ``-10 + softplus(log_var + 10)`` 与 ``10 - softplus(10 - log_var)``
        双向软裁剪到 ``[-10, 10]``,为后续 CRPS-Gaussian 损失提供数值稳定
        的对数方差估计。

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

        # positional embedding and lead time embedding
        self.lead_time_embed = nn.Sequential(
            nn.Linear(1, embed_dim, bias=True),
            nn.GELU(),
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
        """初始化模型中所有可学习参数。

        流程:
            1. 对 ``PatchEmbed`` 的卷积权重按 ``std=0.02`` 做截断正态初始化;
            2. 调用 ``self.apply`` 触发 ``_init_weights`` 钩子,统一初始化所有
               :class:`nn.Linear` 与 :class:`nn.LayerNorm`。
        """
        # token embedding layer
        w = self.patch_embed.proj.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """模块级权重初始化钩子,由 ``self.apply`` 触发。

        Args:
            m (:class:`nn.Module`): 当前遍历到的子模块,仅处理
                :class:`nn.Linear` 与 :class:`nn.LayerNorm`。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_map(self):
        """构建变量名 → 通道索引的映射字典。

        Returns:
            dict: 键为变量名字符串、值为该变量在输入张量通道维上的索引。供
            :meth:`get_var_ids` 反查使用。
        """
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        """根据变量名列表解析出对应的通道索引,并缓存到目标设备上。

        使用 ``lru_cache`` 缓存已解析结果,可显著加速训练时同一变量集合的反
        复查询。

        Args:
            vars (tuple[str]): 变量名可迭代对象,会被 hash 后缓存。
            device (torch.device | str): 返回张量所在的目标设备。

        Returns:
            torch.Tensor: 长度为 ``len(vars)`` 的 ``int64`` 索引张量。
        """
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def forward_encoder(self, x: torch.Tensor, use_checkpoint=False):
        """编码前向过程。

        将多通道气象场 token 化后,堆叠 ``num_enc_layers`` 段 Swin-V2 窗口自
        注意力层,再经 ``enc_fpn`` 将各段归一化输出拼接融合,最后加回 ``residual``
        残差。

        Args:
            x (torch.Tensor): ``[B, V, H, W]`` 形状的输入气象场。
            use_checkpoint (bool): 是否对每段使用梯度检查点以节省显存。

        Returns:
            torch.Tensor: ``[B, L, D]`` 形状的潜在表示, ``L = h*w``,``D = embed_dim``。
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

    def forward_latent(self, h: torch.Tensor, lead_times: torch.Tensor, use_checkpoint=False):
        """带 lead-time 条件化的潜在空间前向过程。

        将预报时长标量映射为 ``embed_dim`` 维嵌入,沿 patch 维广播后,通过
        ``condition=True`` 的 Swin-V2 ``WindowCrossAttentionV2`` 注入到每一
        层 latent block。``latent_fpn`` 融合多段归一化输出并加回残差。

        Args:
            h (torch.Tensor): ``[B, L, D]`` 形状的编码器输出潜在表示。
            lead_times (torch.Tensor): ``[B]`` 形状的每个样本的预报时长
                (小时/步长,具体语义由上游 datamodule 决定)。
            use_checkpoint (bool): 是否对每段使用梯度检查点以节省显存。

        Returns:
            torch.Tensor: ``[B, L, D]`` 形状的潜在表示,已融入 lead-time 信息。
        """
        residual = h
            
        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times).unsqueeze(1)  # B, 1, D
        lead_time_emb = lead_time_emb.repeat(1, h.shape[1], 1)

        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.latent_layers):
            if use_checkpoint:
                h = checkpoint(blk, h, lead_time_emb, use_reentrant=False)
            else:
                h = blk(h, lead_time_emb)
            out = getattr(self, f"latent_norm{i}")(h)
            outs.append(out)
            
        if use_checkpoint:
            h = checkpoint(self.latent_fpn, torch.cat(outs, dim=-1), use_reentrant=False)
        else:
            h = self.latent_fpn(torch.cat(outs, dim=-1))

        h = h + residual

        return h

    def forward_decoder(self, h: torch.Tensor, use_checkpoint=False):
        """解码前向过程。

        堆叠 ``num_dec_layers`` 段不进行条件化的 Swin-V2 自注意力层,``dec_fpn``
        融合多段归一化输出,加回残差后经 ``out_norm`` 归一化,得到与输入空间
        同分辨率的解码特征。

        Args:
            h (torch.Tensor): ``[B, L, D]`` 形状的潜在表示。
            use_checkpoint (bool): 是否对每段使用梯度检查点以节省显存。

        Returns:
            torch.Tensor: ``[B, L, D]`` 形状的解码特征。
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

    def up_forward(self, x):
        """将潜在特征 ``[B, L, D]`` 上采样到 ``[B, V, H, W]`` 物理网格。

        根据 ``img_size[0]`` 的奇偶性分两套实现:
            - 奇数高度(例如 ``181``):使用 ``ConvTranspose2d`` 直接上采样。
            - 偶数高度:使用 :class:`nn.Linear` 沿 patch 维展开,再经
              :func:`einops.rearrange` 还原到 ``(B, V, H, W)``。

        Args:
            x (torch.Tensor): ``[B, L, D]`` 形状的解码特征。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 分别为 ``preds`` 与 ``log_var``,
            形状 ``[B, V, H, W]``。
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

    def forward(self, x, lead_time, variables, use_checkpoint=False):
        """端到端前向传播:encoder → latent (lead-time 条件化) → decoder → 上采样。

        ``log_var`` 在尾部使用 ``softplus`` 双向裁剪到 ``[-10, 10]`` 的稳定区
        间,以避免数值发散;最后根据 ``variables`` 列表裁剪到目标通道。

        Args:
            x (`torch.Tensor`): `[B, Vi, H, W]` 输入气象变量。
            lead_time (`torch.Tensor`): `[B]` 各样本的前置预报时长 (1/100 天为单位,
                由 ``XiChenForecast`` 内部 ``lead_time_embed`` 解析)。
            variables (`list[str]`): 输出变量名子集,沿通道维挑选输出。
            use_checkpoint (bool): 是否对 SwinBlock 使用 ``torch.utils.checkpoint``。

        Returns:
            tuple: ``(preds, log_var)``
            - ``preds`` (`torch.Tensor`): `[B, Vo, H, W]` 预测变量;
            - ``log_var`` (`torch.Tensor`): `[B, Vo, H, W]` 观测误差 log-variance
              (softplus clip 到 ``[-10, 10]``)。
        """
        h = self.forward_encoder(x, use_checkpoint)  # B, L, D

        h = self.forward_latent(h, lead_time, use_checkpoint)

        h = self.forward_decoder(h, use_checkpoint)

        preds, log_var = self.up_forward(h)  # B, L, V*p*p
        
        log_var= -10 + F.softplus(log_var + 10)
        log_var = 10 - F.softplus(10 - log_var)

        out_var_ids = self.get_var_ids(tuple(variables), preds.device)
        preds = preds[:, out_var_ids]
        log_var = log_var[:, out_var_ids]

        return preds, log_var
