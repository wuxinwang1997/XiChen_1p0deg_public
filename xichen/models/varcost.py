"""观测空间变分代价函数 (VarCost) 与观测算子 H(x) 模块。

本文件实现了数据同化 (DA) Solver 中两个核心组件:

1. ``Model_H`` — 观测算子 ``H(x)`` 的轻量包装,在原始残差上叠加 5·σ_obs 的
   质量控制 (QC) 掩码,输出``(dy, mask)``。``mask == 0`` 的位置既不参与 VarCost
   也不参与后续梯度计算。
2. ``Obs_WeighedL2Norm`` — 加权 L2 范数变分代价函数,以 ``R⁻¹ σ²`` 为权重。
   在 Solver 中,它接收 ``Model_H`` 输出的 ``dy`` 和标准化方差 ``std``,返回
   逐样本标量 loss,再由 ``loss.backward()`` 反向传播得到 ``∂J/∂xb`` 梯度,
   送入 DA 模型得到分析增量 ``xa``。

QC 阈值与权重约定的物理意义:
- 观测算子残差 ``OmB = |x - y| * std``(``std`` 来自 ``inference/utils/data_utils``
  预计算的 σ QC)与 ``5 * obs_err`` 比较:``OmB > 5·σ_obs`` 时掩码置 0(剔除离群观测)。
- ``Obs_WeighedL2Norm`` 中 ``R_inv = 1 / obs_err²`` 对应观测误差协方差 ``R`` 的逆,
  残差平方乘 ``R⁻¹ σ²`` 后对通道求和即得变分代价 ``J(x) = 0.5 * (H(x)-y)ᵀ R⁻¹ (H(x)-y)``。
"""
import numpy as np
import torch
import torch.nn as nn

variables = [
  "t2m", "u10", "v10", "msl",
  "z-50", "z-100", "z-150", "z-200", "z-250", "z-300", "z-400", "z-500", "z-600", "z-700", "z-850", "z-925", "z-1000",
  "u-50", "u-100", "u-150", "u-200", "u-250", "u-300", "u-400", "u-500", "u-600", "u-700", "u-850", "u-925", "u-1000",
  "v-50", "v-100", "v-150", "v-200", "v-250", "v-300", "v-400", "v-500", "v-600", "v-700", "v-850", "v-925", "v-1000",
  "t-50", "t-100", "t-150", "t-200", "t-250", "t-300", "t-400", "t-500", "t-600", "t-700", "t-850", "t-925", "t-1000",
  "q-50", "q-100", "q-150", "q-200", "q-250", "q-300", "q-400", "q-500", "q-600", "q-700", "q-850", "q-925", "q-1000",
]

class Model_Var_Cost(nn.Module):
    """变分代价函数 VarCost 的 nn.Module 包装。

    该类将具体的范数实现 (如 ``Obs_WeighedL2Norm``) 注入到 ``normObs`` 属性中,
    Solver 中按观测源 (atms / amsua / mhs / hrs4 / prepbufr / satwnd)
    各自实例化一个 ``Model_Var_Cost``, 互不共享权重。

    Attributes:
        normObs (nn.Module): 具体的变分范数模块,提供 ``forward(dy, std) -> loss``。
    """

    def __init__(self, m_NormObs):
        """初始化 VarCost 包装器。

        Args:
            m_NormObs (nn.Module): 任意实现了 ``forward(dy, std) -> Tensor[B]``
                接口的范数模块。``dy`` 形状为 ``(B, T, C, H, W)``,``std`` 形状为
                ``(1, C)``,返回 ``(B,)`` 标量代价。
        """
        super(Model_Var_Cost, self).__init__()
        self.normObs = m_NormObs

    def forward(self, dy, std):
        """计算变分代价损失。

        Args:
            dy (Tensor): 形状 ``(B, T, C, H, W)`` 的归一化残差,通常来自
                ``Model_H.forward`` 的 ``dyout``。
            std (Tensor): 形状 ``(1, C)`` 的标准化方差 ``σ``。

        Returns:
            Tensor: 形状 ``(B,)`` 的逐样本代价损失。
        """
        loss = self.normObs(dy, std)

        return loss

class Model_H(torch.nn.Module):
    """观测算子 ``H(x)`` 的轻量包装,带 5·σ_obs 质量控制 (QC) 阈值。

    实际计算中,Solver 通常直接调用 ``H_models[obs_name]`` (其底层就是 ``Model_H`` 实例),
    ``H(x)`` 在 ``x`` (模型状态) 与 ``y`` (观测) 之间逐元素做差, 然后用
    ``obs_err`` 阈值剔除离群观测:``OmB = |x - y| * std > 5 * obs_err`` 的位置
    掩码置 0(不参与 VarCost,也不贡献梯度)。

    Attributes:
        obs_err (Tensor): 注册为 buffer 的观测误差张量,形状 ``(1, 1, C, 1, 1)``。
            ``C`` 是当前观测源的通道数。
    """

    def __init__(self, obs_err):
        """初始化观测算子并注册 obs_err 为 buffer。

        Args:
            obs_err (np.ndarray): 一维数组, 长度 = 该观测源的通道数 ``C``。
                注册为 ``(1, 1, C, 1, 1)`` 形状的 ``Tensor`` buffer, 便于
                ``.to(device)`` 时随模型迁移而自动跟随。
        """
        super(Model_H, self).__init__()
        obs_err = np.nan_to_num(obs_err, nan=0, posinf=0, neginf=0).reshape(1, 1, -1, 1, 1)
        self.register_buffer('obs_err', torch.Tensor(obs_err))

    def forward(self, x, y, mask, std):
        """计算带 5·σ_obs QC 掩码的观测空间残差。

        QC 流程:
            1. 计算归一化偏差 ``omb = mask * |x - y| * std``(``std`` 来自
               ``std_dict``, 由 ``inference/utils/data_utils`` 预计算的 σ QC 提供);
            2. 与阈值 ``5 * obs_err`` 比较, ``omb > 5·σ_obs`` 的位置 qc_mask=0;
            3. 最终 ``mask = mask * qc_mask``, 离群观测被屏蔽(残差置 0)。

        Args:
            x (Tensor): 模型状态预测, 形状 ``(B, T, C, H, W)``。
            y (Tensor): 真实观测, 形状与 ``x`` 一致。
            mask (Tensor): 原始观测有效位掩码, 形状 ``(B, T, 1, H, W)``。
            std (Tensor): 标准化方差, 形状 ``(1, 1, C, 1, 1)``。

        Returns:
            tuple[Tensor, Tensor]: ``(dyout, mask)``,
                - ``dyout``: QC 后的残差 ``(x - y) * mask``, 形状 ``(B, T, C, H, W)``;
                - ``mask``: QC 后的掩码, 形状 ``(B, T, 1, H, W)``, 离群位置为 0。
        """
        omb = mask * torch.abs(x - y) * std.reshape(1, 1, -1, 1, 1).to(x.device, dtype=x.dtype)

        threshold = (5 * self.obs_err).to(x.device, dtype=x.dtype)
        qc_mask = (omb <= threshold).to(x.device, dtype=x.dtype)
        mask = mask * qc_mask
        dyout = (x - y) * mask

        return dyout, mask

        # dyout_plot = dyout * std.reshape(1, 1, -1, 1, 1).to(x.device, dtype=x.dtype)

        # for i in range(dyout.shape[-3]):
        #     dyout_flat = dyout_plot[:, :, i].double().flatten().detach().cpu().numpy()
        #     mask_flat = mask[:, :, i].double().flatten().detach().cpu().numpy()
        #     plot_omb_distribution(dyout_flat, mask_flat, variables[i], f"{output_dir}/figures/var_cost_grad/")

        # return dyout, mask
 
class Obs_WeighedL2Norm(torch.nn.Module):
    """以 ``R⁻¹ σ²`` 为权重的观测空间 L2 变分代价函数。

    数学形式 (单样本)::

        J(x) = 0.5 * sum_c  R_inv[c] * std[c]² * sum_{t,h,w} dy[b,t,c,h,w]²

    其中 ``R_inv = 1 / obs_err²`` 是观测误差协方差 ``R = diag(obs_err²)`` 的逆,
    ``std`` 是标准化方差 (``std_dict`` 提供的 σ QC);``dy`` 已经经过
    ``Model_H`` 的 5·σ_obs 掩码。

    内部按 (B, T, C, H, W) 依次在 (H, W)、T、C 三个维度求和, 最终得到
    ``(B,)`` 形状的逐样本代价, 与 Solver 期望的标量输出一致。

    Attributes:
        R_inv (Tensor): 注册为 buffer 的 ``R⁻¹`` 权重, 形状 ``(1, C)``。
    """

    def __init__(self, obs_err):
        """初始化并预计算 ``R⁻¹ = 1 / obs_err²``。

        Args:
            obs_err (np.ndarray): 一维数组, 长度 = 当前观测源的通道数 ``C``。
                ``np.nan_to_num`` 把 NaN/Inf 替换为 0, 防止除零产生 NaN 梯度。
        """
        super(Obs_WeighedL2Norm, self).__init__()
        R_inv = np.nan_to_num(1 / np.nan_to_num(obs_err) ** 2, nan=0, posinf=0, neginf=0).reshape(1, -1)
        self.register_buffer('R_inv', torch.Tensor(R_inv))

    def forward(self, x, std):
        """计算加权 L2 变分代价。

        Args:
            x (Tensor): ``Model_H`` QC 后的残差 ``dy``, 形状 ``(B, T, C, H, W)``。
            std (Tensor): 标准化方差, 形状 ``(1, C)``, 与 ``R_inv`` 形状 broadcast。

        Returns:
            Tensor: 形状 ``(B,)`` 的逐样本变分代价 ``J(x)``。
        """
        var = (std ** 2).reshape(1, -1)  # (1, C) σ², 与 R_inv 一起构成 R⁻¹ σ² 权重
        loss = torch.sum(x ** 2, dim=(-2, -1))  # (B, T, C) 先在 (H, W) 空间维求和
        loss = torch.sum(loss, dim=1)  # (B, C) 再在时间维 T 求和
        loss = loss * self.R_inv.to(x.device, dtype=x.dtype) * var  # (B, C) 乘以 R⁻¹ σ² 通道权重
        loss = torch.sum(loss, dim=-1)  # (B,) 最后在通道维 C 求和,得到每样本标量

        return loss