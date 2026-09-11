"""级联 (cascade) 数据同化 Solver。

本模块实现 ``Solver`` 类, 是 1.0° ERA5 + 多卫星辐射率观测的 cascade DA 推理主流程
（训练在独立训练仓完成, 本仓仅供推理）。
核心思想是 ``4D-Var`` 思想的轻量化近似: 对 ``obs_list`` 中的每一种观测源,
依次执行 ``AR 预报 → H(x) → VarCost → ∂J/∂xb → DA_model → xa``, 最后
对所有 ``xa`` 求平均得到当前步的分析场。

设计要点:
- **逐观测源串行** (``obs_list.keys()`` 顺序): 不同的 obs_name 使用各自的
  ``ObsOp_models`` / ``H_models`` / ``VarCost_models`` / ``DA_models`` 实例;
  cascade 模式下每个 obs_name 一个 DA 模型(per-obs DA)。
- **AR 多步长混合**: ``forecast`` 方法按 ``self.dt`` 步长把 ``1h/3h/6h/12h/24h``
  五个子预报模型串起来, 形成对整个同化窗口的累计预报轨迹。
- **梯度归一化**: ``var_cost_grad`` 通过 ``L2-norm`` 归一化后送入 DA 模型,
  避免不同 obs 的梯度尺度差异。
- **最终 ``mean(xas)``**: 对所有 per-obs ``xa`` 求平均, 即最终分析场。

注意: 旧版 (``old/`` 目录) 的 amsua/atms/hrs4/mhs/multimodal/prepbufr/satwnd
Solver 是早期单 obs 硬编码实现, 已废弃, 不要修改。
"""
import numpy as np
import torch
import torch.nn as nn
from xichen.models.ar import ar_forecast_trajectory
from xichen.utils import get_logger

log = get_logger("xichen.cascade")

class Solver(nn.Module):
    """级联数据同化主循环。

    该类没有可学习参数(``self.dt`` 只是时间步长配置), 训练时由 trainer 把
    forecast_model / ObsOp_models / H_models / VarCost_models / DA_models 作为
    ``forward`` 参数注入, 因此 ``Solver`` 自身不持有这些子模块。
    """

    def __init__(
        self,
        dt,
    ):
        """初始化 Solver。

        Args:
            dt (int): 预报步长(小时), 取值通常为 1 / 3 / 6。``forecast`` 方法按
                ``dt`` 把 ``1h/3h/6h/12h/24h`` 五个子预报模型对齐到时间网格。
        """
        super(Solver, self).__init__()
        self.dt = dt

    def forward(
        self,
        forecast_model,
        ObsOp_models,
        DA_models,
        H_models,
        VarCost_models,
        obs_list,
        xb,
        obs,
        obs_mask,
        obs_dict,
        std_dict,
        out_vars,
    ):
        """执行 cascade DA 主循环, 返回所有 obs 源 xa 的平均。

        流程(对每个 obs_name):
            1. 冻结 ``xb`` 为 leaf 且 ``requires_grad=True``;
            2. ``self.var_cost`` 计算 ``∂J/∂xb`` (含 5·σ_obs QC + L2 归一化);
            3. ``DA_models[obs_name](xb, grad)`` 输出 ``(xa, log_var)``;
            4. 若该 obs 不可训练, detach 切断梯度;
            5. 累加 ``xas`` 列表。

        最终 ``mean(xas)`` 即为多观测源联合分析场。

        Args:
            forecast_model (nn.Module): ``XiChenForecast`` 子模型, 用于 AR 预报。
            ObsOp_models (dict[str, nn.Module]): per-obs 观测算子(仅辐射率观测需要,
                如 atms/amsua/mhs/hrs4)。
            DA_models (dict[str, nn.Module]): per-obs DA 网络(``XiChenDA`` 实例),
                输入 ``(xb, grad)``, 输出 ``(xa, log_var)``。
            H_models (dict[str, nn.Module]): per-obs ``Model_H`` 观测算子实例。
            VarCost_models (dict[str, nn.Module]): per-obs ``Model_Var_Cost`` 实例。
            obs_list (dict[str, object]): 当前 batch 启用的观测源集合, key 是
                obs_name, value 含 ``trainable`` 标志。
            xb (Tensor): 背景场, 形状 ``(B, C, H, W)``。
            obs (dict[str, Tensor]): per-obs 观测张量字典。
            obs_mask (dict[str, Tensor]): per-obs 有效位掩码字典。
            obs_dict (dict): 观测元信息(通道名、``tmbrs_vars``、``vars`` 等)。
            std_dict (dict[str, Tensor]): per-obs 标准化方差 ``σ`` 字典。
            out_vars (list[str]): 状态变量名列表(69 通道)。

        Returns:
            tuple[Tensor, Tensor]: ``(mean_xa, log_var)``。
                - ``mean_xa``: 形状 ``(B, C, H, W)`` 的多 obs 源平均分析场;
                - ``log_var``: 最后一个 DA 模型的 log-variance 头输出。
        """
        # === F1: 设备判断(三后端兼容,避免 CPU 上触发 CUDA lazy-init) ===
        class _CpuNoOp:
            """CPU 后端的 no-op stub,使 dev_module.synchronize/empty_cache 调用安全。"""
            @staticmethod
            def synchronize(): pass
            @staticmethod
            def empty_cache(): pass

        if xb.device.type == "npu":
            dev_module = torch.npu
        elif xb.device.type == "cuda":
            dev_module = torch.cuda
        else:
            dev_module = _CpuNoOp

        # === F2: 释放 stream-sync 压力(在 NPU/GPU 上有效;CPU 上为 no-op) ===
        dev_module.synchronize()
        dev_module.empty_cache()

        # === F3: obs_list 为空防御 ===
        if not obs_list:
            raise ValueError(
                "Solver.forward: obs_list is empty; check training.obs_list config "
                "or datamodule obs loading."
            )

        xas = []
        try:
            for obs_name in obs_list.keys():
                # === F4: 循环内释放 stream-sync 压力 ===
                dev_module.synchronize()
                dev_module.empty_cache()

                # === F5: 每轮迭代重建 state,反映更新后的 xb ===
                # 旧写法 Variable(xb, requires_grad=True) 也是新建 leaf,语义一致;
                # 这里改用 detach().requires_grad_(True) 与 multimodal 风格对齐。
                state = xb.detach().requires_grad_(True)

                with torch.set_grad_enabled(True):
                    var_cost_grad = self.var_cost(
                        forecast_model,
                        ObsOp_models,
                        H_models,
                        VarCost_models,
                        obs_name,
                        state,
                        obs,
                        obs_mask,
                        obs_dict,
                        std_dict,
                        out_vars
                    )
                xa, log_var = DA_models[obs_name](
                    xb,
                    var_cost_grad,
                    out_vars,
                    use_checkpoint=True
                )
                if obs_list[obs_name].trainable == False:
                    xa, log_var = xa.detach(), log_var.detach()

                xas.append(xa)
                xb = torch.stack(xas, dim=0).mean(dim=0)
        except Exception as e:
            log.error(
                f"Solver.forward failed mid-loop (last obs={locals().get('obs_name', '?')}): "
                f"{type(e).__name__}: {e}; releasing xb.grad"
            )
            raise
        finally:
            # 收尾:清理 xb 的累积梯度(级联模式下 xb 跨 obs 共享)
            xb.grad = None

        dev_module.synchronize()
        dev_module.empty_cache()

        return torch.stack(xas, dim=0).mean(dim=0), log_var

    def var_cost(
        self,
        forecast_model,
        ObsOp_models,
        H_models,
        VarCost_models,
        obs_name,
        xb,
        obs,
        obs_mask,
        obs_dict,
        std_dict,
        out_vars
    ):
        """对单个 obs 源计算 ``∂J/∂xb`` 归一化梯度。

        流程:
            1. ``self.forecast`` 沿时间轴 AR 预报, 得到 ``preds`` 形状
               ``(B, T, C, H, W)``;
            2. 若该 obs 有 ObsOp: ``ObsOp_models[obs_name](preds, obs)`` 得到
               模拟辐射率,再 ``H_models[obs_name]`` 做差并施加 5·σ_obs QC;
            3. 若该 obs 是常规观测: 直接对 ``preds`` 的目标通道子集与 obs 做差;
            4. ``VarCost_models[obs_name]`` 算 ``J``, ``loss.backward()`` 反传;
            5. 取出 ``xb.grad``、nan_to_num 清洗、L2 归一化。

        Args:
            forecast_model (nn.Module): ``XiChenForecast`` 子模型。
            ObsOp_models (dict[str, nn.Module]): 辐射率观测算子字典。
            H_models (dict[str, nn.Module]): 观测算子 ``Model_H`` 字典。
            VarCost_models (dict[str, nn.Module]): 变分代价字典。
            obs_name (str): 当前处理的观测源名(如 ``atms``)。
            xb (Tensor): 背景场, ``requires_grad=True``, 形状 ``(B, C, H, W)``。
            obs (dict[str, Tensor]): per-obs 观测张量字典。
            obs_mask (dict[str, Tensor]): per-obs 有效位掩码。
            obs_dict (dict): 观测元信息(``microwave`` / ``conventional`` 子字典)。
            std_dict (dict[str, Tensor]): per-obs σ QC 字典。
            out_vars (list[str]): 状态变量名列表(69 通道)。

        Returns:
            Tensor: 形状 ``(B, C, H, W)`` 的 ``∂J/∂xb``, 已 L2 归一化且不含 NaN/Inf。
        """
        preds = ar_forecast_trajectory(
            forecast_model, 
            xb, 
            obs[obs_name], 
            out_vars, 
            self.dt,
            use_checkpoint=True
        )

        B, T, C, H, W = preds.shape
        B, T, Cs, H, W = obs[obs_name].shape

        if obs_name in ObsOp_models.keys():
            pred_obs, log_var, tgt_obs = ObsOp_models[obs_name](
                preds.view(B * T, C, H, W),
                obs[obs_name].view(B * T, Cs, H, W),
                obs_mask[obs_name].view(B * T, 1, H, W),
                use_checkpoint=True
            )
            tgt_sat_var_ids = np.array(
                [
                    obs_dict["microwave"][obs_name]["tmbrs_vars"].index(item)
                    for item in ObsOp_models[obs_name].out_sat_vars
                ]
            )
            tgt_sat_var_ids = torch.from_numpy(tgt_sat_var_ids).to(xb.device)
            std = std_dict[obs_name][tgt_sat_var_ids].reshape(1, 1, -1, 1, 1)

            dy, _ = H_models[obs_name](
                pred_obs.view(B, T, -1, H, W),
                tgt_obs.view(B, T, -1, H, W),
                obs_mask[obs_name].view(B, T, 1, H, W),
                std
            )
        else:
            tgt_sat_var_ids = np.array(
                [
                    out_vars.index(item)
                    for item in obs_dict["conventional"][obs_name]["vars"]
                ]
            )
            tgt_sat_var_ids = torch.from_numpy(tgt_sat_var_ids).to(xb.device)
            std = std_dict[obs_name].reshape(1, 1, -1, 1, 1)

            dy, _ = H_models[obs_name](
                preds[:, :, tgt_sat_var_ids],
                obs[obs_name],
                obs_mask[obs_name],
                std
            )

        # 关键:在传递给 VarCost 之前清理无效值
        dy = torch.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0)

        loss = VarCost_models[obs_name](dy, std.reshape(1, -1))

        loss = torch.where(torch.isnan(loss), 0, loss)
        loss = torch.where(torch.isinf(loss), 0, loss)
            
        # loss.backward(torch.ones_like(loss), retain_graph=True)
        loss.backward(torch.ones_like(loss))

        var_cost_grad = xb.grad.detach()
        # log.info(f"Norm Grad is {torch.sqrt(torch.mean(var_cost_grad ** 2, dim=(1, 2, 3), keepdim=True))}")
        xb.grad = None

        # 清理梯度
        var_cost_grad = torch.nan_to_num(var_cost_grad, nan=0.0, posinf=0.0, neginf=0.0)

        # 关键:对 ∂J/∂xb 做 L2 范数归一化,保证不同 obs 源(辐射率 / 常规)
        # 梯度的尺度一致,便于 DA 模型学习稳定的 (xb, grad) -> xa 映射;
        # +1e-8 防止 0 范数 sqrt(0) 反向传播炸梯度。
        normgrad_ = torch.sqrt(torch.mean(var_cost_grad ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8)
        normgrad_ = torch.nan_to_num(normgrad_, nan=1.0, posinf=1.0, neginf=1.0)
        normgrad_[normgrad_ == 0] = 1.0

        var_cost_grad = var_cost_grad / normgrad_

        del preds, dy, normgrad_, loss
        if obs_name in ObsOp_models.keys():
            del pred_obs, log_var, tgt_obs

        return var_cost_grad    