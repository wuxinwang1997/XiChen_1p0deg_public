"""AR 预报轨迹生成工具。

把 1h / 3h / 6h / 12h / 24h 五个子预报模型按 ``dt`` 步长串起来,沿同化窗口
长度 ``yobs.shape[1]`` 拼成完整 AR 轨迹。供 ``cascade.Solver`` 与
``multimodal.Solver`` 共享调用。

设计意图:单一职责的预报轨迹生成;不涉及观测算子、变分代价、DA 网络。
"""
import numpy as np
import torch
import torch.nn as nn

__all__ = ["ar_forecast_trajectory"]


def ar_forecast_trajectory(
    forecast_model: nn.Module,
    x0: torch.Tensor,
    yobs: torch.Tensor,
    out_vars: list[str],
    dt: int,
    *,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """把 1h/3h/6h/12h/24h 子模型串成完整 AR 预报轨迹。

    沿 ``yobs.shape[1]``(同化窗口长度)对时间步 ``i`` 推进:24h / 12h / 6h /
    3h / 1h 五个 lead_time 优先级触发对应子模型。lead-time 单位是 1/100 天
    (``lead / 100``),由 ``XiChenForecast`` 内部用 ``lead_time_embed`` 解析。
    ``lead_time`` 张量 dtype 使用 ``x0.dtype``,以便 bf16 训练时与 ``x0``
    对齐。

    Args:
        forecast_model (nn.Module): ``XiChenForecast`` 实例(支持 1/3/6/12/24h
            五个 lead_time)。
        x0 (Tensor): 初始背景场,形状 ``(B, C, H, W)``。
        yobs (Tensor): 观测张量,形状 ``(B, T, Cs, H, W)``,其中 ``T`` 是
            同化窗口的时间步数。
        out_vars (list[str]): 状态变量名列表(69 通道),作为
            ``forecast_model.forward`` 的输出键列表。
        dt (int): 预报步长(小时),取值通常为 1 / 3 / 6。
        use_checkpoint (bool): 是否对 ``forecast_model`` 调用启用
            ``use_checkpoint``。默认 ``True``(与历史行为一致)。

    Returns:
        Tensor: AR 预报轨迹,形状 ``(B, T, C, H, W)``,``T = yobs.shape[1]``。
    """
    preds = []
    preds.append(x0)
    for i in range(1, yobs.shape[1]):
        if ((24 // dt) > 0) and (i % (24 // dt)) == 0:
            preds.append(forecast_model(preds[i - 24 // dt],
                                    torch.from_numpy(24 * np.ones((x0.shape[0], 1))).to(x0.device, dtype=x0.dtype) / 100,
                                    out_vars,
                                    use_checkpoint=use_checkpoint)[0])
        elif ((12 // dt) > 0) and (i % (12 // dt)) == 0:
            preds.append(forecast_model(preds[i - 12 // dt],
                                    torch.from_numpy(12 * np.ones((x0.shape[0], 1))).to(x0.device, dtype=x0.dtype) / 100,
                                    out_vars,
                                    use_checkpoint=use_checkpoint)[0])
        elif ((6 // dt) > 0) and (i % (6 // dt)) == 0:
            preds.append(forecast_model(preds[i - 6 // dt],
                                    torch.from_numpy(6 * np.ones((x0.shape[0], 1))).to(x0.device, dtype=x0.dtype) / 100,
                                    out_vars,
                                    use_checkpoint=use_checkpoint)[0])
        elif ((3 // dt) > 0) and (i % (3 // dt)) == 0:
            preds.append(forecast_model(preds[i - 3 // dt],
                                    torch.from_numpy(3 * np.ones((x0.shape[0], 1))).to(x0.device, dtype=x0.dtype) / 100,
                                    out_vars,
                                    use_checkpoint=use_checkpoint)[0])
        elif ((1 // dt) > 0) and (i % (1 // dt)) == 0:
            preds.append(forecast_model(preds[i - 1 // dt],
                                    torch.from_numpy(1 * np.ones((x0.shape[0], 1))).to(x0.device, dtype=x0.dtype) / 100,
                                    out_vars,
                                    use_checkpoint=use_checkpoint)[0])

    preds = torch.stack(preds, dim=1)

    return preds
