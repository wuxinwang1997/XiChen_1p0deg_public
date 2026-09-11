# -*- coding: utf-8 -*-
"""确定性级联 DA 单周期（XiChen）。

单 12h 周期 = 12h 背景场预报 → 12h 窗口同化 → analysis_12h/<T>.nc；
T.hour ∈ trigger_6h_hours（默认 00/12 UTC）时再做 6h 窗口重分析 → ic_6h/<T>.nc。
跨周期主循环见 ``xichen.dacycle.common.run_dacycle_loop``。
"""
import time

import numpy as np
import torch

from xichen.dacycle.common import (
    VARIABLES,
    _run_solver,
    compute_metrics,
    get_normalize,
    load_era5_truth,
    load_obs_window,
    load_xb_initial,
    log,
    save_metrics,
    save_nc,
    save_npy,
    weighted_rmse_channels,
)


def assim_cycle_xichen(cfg, T, prev_xa, components, obs_dict, device):
    """单 12h 周期：
        A) 12h 窗口同化 → analysis_12h/<T>.nc
        B) 若 T.hour ∈ trigger_6h_hours → 6h 窗口重分析 → ic_6h/<T>.nc
    返回 (xa12, timing)，timing 含 t_forecast_12h_s / t_da_12h_s / t_da_6h_s（秒）。
    xa12 供下一周期作 prev_xa12；timing 写进 metrics JSON 与 main loop 累计均值。

    device 参数：从 main() 显式传入；Solver 模块本身无可学习参数，
    无法用 solver.device / next(solver.parameters()).device 反推。
    """
    forecast, obsops, da_models, h_models, varcost_models, solver, microwave_prep, conventional_prep = components

    # === 计时：12h 背景场 forecast ===
    t0 = time.time()
    if prev_xa is None:
        xb = load_xb_initial(cfg, T, device).to(device)
    else:
        # 修复 HIGH#1：prev_xa 链路 NaN/Inf 检测。cascade Solver 内部 nan_to_num 只
        # 保护当次迭代，跨周期的 prev_xa 在 forecast 输入端无任何拦截；若任一周期
        # 污染 NaN → 永久污染后续所有周期（最终 analysis_12h 全 NaN 且脚本静默 exit 0）。
        prev_xa = prev_xa.to(device)
        if torch.isnan(prev_xa).any() or torch.isinf(prev_xa).any():
            log.error("T=%s | prev_xa contains NaN/Inf, falling back to climatology", T.isoformat())
            xb = load_xb_initial(cfg, T, device).to(device)
        else:
            with torch.no_grad():
                preds, _ = forecast(
                    prev_xa,
                    torch.tensor([[12 * 0.01]], device=device),  # lead = 12h，单位 0.01 天
                    VARIABLES,
                    use_checkpoint=True,   # 推理：节省显存（gradient checkpointing）
                )
                # forecast 返回 (preds, log_var) 元组；preds.shape=(B=1, V=69, H, W)
                # 关键：不要 preds[0]（那是 (V, H, W) 缺 batch 维，喂给下一轮 Solver 会触发
                # Conv2d 把 V 当 batch，输出 (V, 768, h, w) → flatten+transpose → (768, 72, 36)，
                # LayerNorm(768) 报 "input of size [768, 72, 36] but expected [..., 768]"）
                #
                # 修复 MEDIUM#1：强制 fp32。autocast 下 forecast 可能输出 bf16；下一轮
                # forecast 的 Conv2d 权重若为 fp32 会抛 "Input type ... and weight type ... mismatch"。
                # 训练时 collate_fn 默认 fp32，所以推理也应锁定 fp32 跨周期。
                xb = preds.detach().float()
    t_forecast_12h_s = time.time() - t0

    # === 计时：12h 窗口同化 (load_obs + solver + save_nc + 评估) ===
    t0 = time.time()
    obs12, mask12, std12 = load_obs_window(
        cfg["obs_dir"], cfg["era5_lr_dir"], T,
        window_hours=12, dt_obs=3,
        obs_order=cfg["obs_order"], obs_dict=obs_dict,
        microwave_prep=microwave_prep,
        conventional_prep=conventional_prep,
    )
    xa12 = _run_solver(solver, components, xb, obs12, mask12, std12,
                       obs_dict, cfg["obs_order"])
    save_nc(xa12, T, cfg["output_dir"] + "/analysis_12h",
            cfg["scale_dir"], cfg["obs_order"])

    # 加载 ERA5 truth（无论成功与否，12h/6h 评估共用 — 修复 BUG #4）
    truth = None
    try:
        truth = load_era5_truth(cfg["era5_lr_dir"], T, cfg["scale_dir"], device)
    except FileNotFoundError as e:
        log.warning("T=%s | ERA5 truth not found, skip metrics: %s", T.isoformat(), e)

    # === 诊断日志：定位 xb / xa vs truth 的偏差来源 ===
    # 与 compute_metrics / forecast_eval._eval_one_init 完全一致：
    #   1) 反归一化到物理单位  2) weighted_rmse_channels（lat × cos 归一化）
    # 单位：z-/q-/t-/msl 为 m²/s² / kg/kg / K / Pa；u/v 风为 m/s。
    if truth is not None:
        with torch.no_grad():
            mean_np, std_np = get_normalize(cfg["scale_dir"], VARIABLES)  # (1, 69, 1, 1)

            def _denorm(x):
                # Solver 可能返回 (B=1, T=1, V, H, W)（dim=5）→ 压成 (1, V, H, W)
                if x.dim() == 5:
                    x = x.squeeze(1)
                return (x.detach().cpu().numpy() * std_np + mean_np).astype(np.float32)

            xb_phys = _denorm(xb)
            xa_phys = _denorm(xa12)
            truth_phys = _denorm(truth)

            # 纬度加权 RMSE（与 forecast_eval._eval_one_init 的 weighted_rmse 一致）
            # 输入 (N=1, C=69, H=181, W=360)；返回 (69, 1, 1, 1) → squeeze 得 (69,)
            rmse_xb = weighted_rmse_channels(xb_phys, truth_phys).squeeze()
            rmse_xa = weighted_rmse_channels(xa_phys, truth_phys).squeeze()

            Z500_IDX = 11
            # 关键变量集合（z-500 是位势高度核心评估，附带 z-300/z-850/t2m/u10/v10/msl 对照）
            KEY_IDX = {
                "z-300": 9, "z-500": 11, "z-850": 14,
                "t2m": 0, "u10": 1, "v10": 2, "msl": 3,
            }
            log.info(
                "T=%s | DIAG lat-weighted RMSE (physical units, 与 compute_metrics 一致):",
                T.isoformat(),
            )
            parts = [
                f"{v} xb={rmse_xb[i]:8.2f} xa={rmse_xa[i]:8.2f} Δ={rmse_xa[i] - rmse_xb[i]:+7.2f}"
                for v, i in KEY_IDX.items()
            ]
            log.info("  %s", " | ".join(parts))
            log.info(
                "T=%s | z-500 WRMSE: xb=%.2f xa=%.2f (m²/s²), improvement=%+.2f",
                T.isoformat(),
                rmse_xb[Z500_IDX], rmse_xa[Z500_IDX],
                rmse_xa[Z500_IDX] - rmse_xb[Z500_IDX],
            )

    # 评估 12h 分析
    if truth is not None:
        metrics_12h = compute_metrics(xa12, truth, cfg["scale_dir"])
        save_metrics(metrics_12h, T, "analysis_12h", cfg["output_dir"],
                     timing={"t_forecast_12h_s": t_forecast_12h_s,
                             "t_da_12h_s": time.time() - t0})
    t_da_12h_s = time.time() - t0

    # ---- 3) 6h 重分析（每日 00/12 UTC），共享同一 xb ----
    t_da_6h_s = 0.0
    if T.hour in cfg["trigger_6h_hours"]:
        t0 = time.time()
        obs6, mask6, std6 = load_obs_window(
            cfg["obs_dir"], cfg["era5_lr_dir"], T,
            window_hours=6, dt_obs=3,
            obs_order=cfg["obs_order"], obs_dict=obs_dict,
            microwave_prep=microwave_prep,
            conventional_prep=conventional_prep,
        )
        xa6 = _run_solver(solver, components, xb, obs6, mask6, std6,
                          obs_dict, cfg["obs_order"])
        # 6h 重分析写出：根据 cfg['save_format'] 决定格式
        #   nwp  → 只写 nc（默认，零影响）
        #   npy  → 只写 npy（forecast 微调直读）
        #   both → nc + npy 双写
        save_fmt = cfg.get("save_format", "nwp")
        if save_fmt in ("nwp", "both"):
            save_nc(xa6, T, cfg["output_dir"] + "/ic_6h",
                    cfg["scale_dir"], cfg["obs_order"])
        if save_fmt in ("npy", "both"):
            save_npy(xa6, T, cfg["output_dir"], cfg["scale_dir"],
                     subdir_name="ic_6h_npy")
        # 评估 6h 重分析（仅 truth 可用时）
        if truth is not None:
            metrics_6h = compute_metrics(xa6, truth, cfg["scale_dir"])
            save_metrics(metrics_6h, T, "ic_6h", cfg["output_dir"],
                         timing={"t_da_6h_s": time.time() - t0})
        t_da_6h_s = time.time() - t0
        log.info("T=%s | 6h IC saved in %.2fs", T.isoformat(), t_da_6h_s)

    return xa12, {
        "t_forecast_12h_s": t_forecast_12h_s,
        "t_da_12h_s": t_da_12h_s,
        "t_da_6h_s": t_da_6h_s,
    }
