"""Shared AR-forecast evaluation core.

Extracted from the near-duplicate pair `era5_lr_forecast.py` and
`era5_interp_forecast.py`. The two scripts differed only in how they
loaded the evaluation-time ERA5 truth + climatology (and how the
forecast slice was denormalised back to physical units). That step is
factored out behind a small `loader` protocol; everything else — model
construction, checkpoint loading, normalisation, the multi-resolution
AR rollout, the metric computation, plotting, and `*.csv` writing —
lives here.

Public API
----------
eval_forecast(loader, ckpt_path, config, output_dir) -> dict

* loader : Callable[[datetime], tuple(era5, clim)]
    Called once per evaluation lead-time. Receives the `eval_time`
    datetime for the current lead, and returns a 2-tuple:
        era5 : np.ndarray   shape (1, V, H, W), physical units
        clim : np.ndarray   same shape, physical units
* ckpt_path : str
    Directory passed to `load_forecast_ckpt`.
* config : dict
    Required keys: era5_lr_dir, era5_hr_dir, forecast_hours,
    start_year, end_year, decorrelation_hours, dt, forecast_name,
    device.
* output_dir : str
    Where the per-metric CSVs (`*_rmse.csv`, `*_acc.csv`,
    `*_activity.csv`, `*_pred_rmse.csv`) are written.

The wrapper scripts pass `resolution_tag` via `config["resolution_tag"]`
so figure / CSV filenames match the prior per-script conventions.

输出指标格式（CSV，rows=变量，cols=lead time）：
    ${output_dir}/${forecast_name}_${resolution_tag}_rmse.csv
    ${output_dir}/${forecast_name}_${resolution_tag}_acc.csv
    ${output_dir}/${forecast_name}_${resolution_tag}_activity.csv
    ${output_dir}/${forecast_name}_${resolution_tag}_pred_rmse.csv
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta

import json

from xichen.models.forecast import XiChenForecast
from xichen.metrics import weighted_rmse, weighted_acc, weighted_activity
from xichen.data import VARIABLES, get_era5, get_normalize, get_climatology
from xichen import nwp
from xichen.ckpt import load_forecast_ckpt
from xichen.plotting import (
    plot_forecast_metrics,
    save_forecast_plots,
)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("inference.era5_forecast_core")


_CONFIG_KEYS = (
    "era5_lr_dir",
    "era5_hr_dir",
    "forecast_hours",
    "dt",
    "forecast_name",
    "device",
)

# Only required when `config["init_times"]` is absent, i.e. the init times are
# derived from the built-in year-walk instead of being supplied explicitly.
_TIMEWALK_CONFIG_KEYS = (
    "start_year",
    "end_year",
    "decorrelation_hours",
)


def _build_model(forecast_config: str = "configs/xichen_forecast.json") -> XiChenForecast:
    """Construct the XiChenForecast from the hyper-parameter JSON.

    模型超参（patch_size=[6,5]、patch_stride=[5,5] 等）与训练时严格一致，
    全部来自随仓库分发的 JSON——checkpoint 是纯 state_dict，不含超参。
    """
    with open(forecast_config) as f:
        fc_cfg = json.load(f)
    fc_cfg = {k: v for k, v in fc_cfg.items() if k != "default_vars"}
    return XiChenForecast(VARIABLES, **fc_cfg)


def _autoregressive_rollout(
    forecast_model: XiChenForecast,
    era5_init: torch.Tensor,
    forecast_hours: int,
    dt: int,
    device,
):
    """Run the multi-resolution AR rollout; return (forecast, log_var)."""
    horizon = forecast_hours // dt + 1
    # dtype=float32 防止后续 torch.from_numpy() 转成 float64 tensor,与 Conv2d 权重不匹配
    seq_forecast = np.zeros((horizon, len(VARIABLES), 181, 360), dtype=np.float32)
    seq_log_var = np.zeros((horizon, len(VARIABLES), 181, 360), dtype=np.float32)

    with torch.no_grad():
        for i in range(horizon):
            if i == 0:
                # era5_init is (1, V, H, W) on `device` (cuda). Detach + cpu + squeeze
                # the batch dim so it matches seq_forecast[0] shape (V, H, W).
                seq_forecast[0] = era5_init.detach().cpu().numpy().squeeze(0)
                seq_log_var[0] = 0.0  # log-var of the IC is 0 by convention
                continue
            # Cascade through discrete sub-model horizons {24,12,6,3,1}h.
            for step in (24, 12, 6, 3, 1):
                if (step // dt) > 0 and (i % (step // dt)) == 0:
                    pred, log_var = forecast_model(
                        torch.from_numpy(
                            seq_forecast[i - step // dt : i - step // dt + 1]
                        ).to(device, dtype=torch.float32),
                        torch.from_numpy(
                            step * np.ones((1, 1))
                        ).to(device, dtype=torch.float32) / 100,
                        VARIABLES,
                        use_checkpoint=True,
                    )
                    seq_forecast[i : i + 1] = pred.detach().cpu().numpy()
                    seq_log_var[i : i + 1] = log_var.detach().cpu().numpy()
                    break
    return seq_forecast, seq_log_var


def _autoregressive_rollout_batched(
    forecast_model: XiChenForecast,
    era5_init_batch: torch.Tensor,
    forecast_hours: int,
    dt: int,
    device,
):
    """批量 AR rollout：一次 forward 处理 N 个独立 init time（数据并行）。

    与 _autoregressive_rollout 单 init 版本的区别：
      - 输入 era5_init_batch: (N, V, H, W) — N 个并行 init times
      - 输出 (seq_forecast, seq_log_var): (N, horizon, V, H, W) — N 个独立 AR 轨迹
      - lead-time tensor shape (N, 1) 而非 (1, 1)；forecast model 内部 broadcast 处理

    数学等价性：
      - AR 在 lead-time 维度上串行（每个 step 依赖前一步）
      - 但 N 个 init times 之间完全独立，可以 batch 内并行
      - 结果应与串行调用 N 次 _autoregressive_rollout 字节级一致（fp32 精度内）

    显存估算（forecast_hours=240, dt=6, N=batch_size）：
      - AR 状态张量：N × 41 × 69 × 181 × 360 × 4B ≈ N × 700MB（CPU offload 后）
      - GPU 激活（Swin-V2 with use_checkpoint=True）：N × ~1.5GB（gradient checkpoint 复用）
      - 模型本身（bf16）：~200MB
      - 推荐：单卡 24GB → N=4~8；单卡 80GB (A100) → N=16~32
    """
    horizon = forecast_hours // dt + 1
    V = len(VARIABLES)
    H, W = 181, 360
    N = era5_init_batch.shape[0]

    # 主体在 CPU 上累积（避免 GPU 显存爆炸），每 step 仅把当前输入搬到 GPU
    seq_forecast = torch.zeros(N, horizon, V, H, W, dtype=torch.float32)
    seq_log_var = torch.zeros(N, horizon, V, H, W, dtype=torch.float32)
    seq_forecast[:, 0] = era5_init_batch.cpu()
    seq_log_var[:, 0].zero_()

    with torch.no_grad():
        for i in range(1, horizon):
            for step in (24, 12, 6, 3, 1):
                if (step // dt) > 0 and (i % (step // dt)) == 0:
                    # 关键：N 个 init times 在 batch 维并行
                    x = seq_forecast[:, i - step // dt].to(device, dtype=torch.float32)  # (N, V, H, W)
                    lead = (step * torch.ones((N, 1)) / 100).to(device, dtype=torch.float32)
                    pred, log_var = forecast_model(
                        x, lead, VARIABLES, use_checkpoint=True,
                    )
                    # === 每步完整保存：与单 init 版本 _autoregressive_rollout 一致 ===
                    # seq_forecast[:, i] = pred.detach().cpu() 把第 i 步预测完整写入
                    # N 个 init times 全部保存，不丢任何 lead time 的数据
                    seq_forecast[:, i] = pred.detach().cpu()
                    seq_log_var[:, i] = log_var.detach().cpu()
                    # === 显存控制三层保险 ===
                    # 1) with torch.no_grad()：禁用 autograd graph（不保存 grad）
                    # 2) .detach()：即使 outside no_grad 也安全（防御性）
                    # 3) .cpu()：搬到 CPU 释放 GPU 显存（seq_forecast 在 CPU 累积）
                    # 显式 del 临时变量，离开 scope 后立刻释放引用（避免延迟到 GC）
                    del x, lead, pred, log_var
                    break
    return seq_forecast.numpy(), seq_log_var.numpy()


def _denorm(forecast_slice: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse normalise a (1, V, H, W) forecast slice with (V,) mean/std."""
    return forecast_slice * std + mean


def _eval_one_init(
    loader: Callable[[datetime], tuple],
    denorm: Callable[[np.ndarray], np.ndarray],
    seq_forecast: np.ndarray,
    seq_log_var: np.ndarray,
    current_time: datetime,
    forecast_hours: int,
    dt: int,
    pred_rmse_scale: np.ndarray,
):
    """Compute per-lead-time RMSE / ACC / activity / pred-RMSE for one init.

    `loader(eval_time)` returns `(era5, clim)` truth + climatology at the
    resolution used for the metric comparison. `denorm(forecast_slice)`
    takes the raw normalised forecast slice (1, V, H, W) and returns the
    physical-units slice at the metric-comparison resolution — e.g.
    identity for the LR (181x360) comparison, or
    `geographic_interpolate(lr2hr)` followed by HR denormalisation for
    the 0.25deg comparison. `pred_rmse_scale` is the high-res std used
    to scale the predicted standard deviation.
    """
    horizon = forecast_hours // dt + 1
    rmse_xf, acc_xf, activity_xf, pred_rmse_xf = [], [], [], []

    for i in range(horizon):
        eval_time = current_time + relativedelta(hours=i * dt)
        era5, clim = loader(eval_time)
        forecast_eval = denorm(seq_forecast[i : i + 1])

        rmse_xf.append(weighted_rmse(era5, forecast_eval))
        acc_xf.append(weighted_acc(era5 - clim, forecast_eval - clim))
        activity_xf.append(weighted_activity(forecast_eval, clim))
        pred_rmse = np.sqrt(np.exp(seq_log_var[i : i + 1])) * pred_rmse_scale
        pred_rmse_xf.append(np.sqrt(np.mean(pred_rmse ** 2, axis=(0, -2, -1))))

    return (
        np.stack(rmse_xf).squeeze(),
        np.stack(acc_xf).squeeze(),
        np.stack(activity_xf).squeeze(),
        np.stack(pred_rmse_xf).squeeze(),
    )


def _timewalk_init_times(config: dict) -> list:
    """The built-in init-time sequence: Jan 1 -> Dec 21 of `start_year`."""
    start_time = datetime(config["start_year"], 1, 1, 0, 0)
    end_time = datetime(config["start_year"], 12, 21, 0, 0)
    times, current_time = [], start_time
    while current_time < end_time:
        times.append(current_time)
        current_time = current_time + relativedelta(
            hours=config["decorrelation_hours"]
        )
    return times


def save_metrics_and_plots(
    seq_rmse_xf,
    seq_acc_xf,
    seq_activity_xf,
    seq_pred_rmse_xf,
    forecast_name: str,
    output_dir: str,
    resolution_tag: str,
    dt: int = 6,
) -> dict:
    """Stack, plot, save CSVs (rows=variable, cols=lead time); return metrics dict.

    输出（CSV，人类可读，pandas/excel 友好）：
      - ${output_dir}/${forecast_name}_${resolution_tag}_rmse.csv
      - ${output_dir}/${forecast_name}_${resolution_tag}_acc.csv
      - ${output_dir}/${forecast_name}_${resolution_tag}_activity.csv
      - ${output_dir}/${forecast_name}_${resolution_tag}_pred_rmse.csv

    `resolution_tag` distinguishes the two CLI variants, e.g. "1p0deg"
    or "interp_0p25deg"; it suffixes figure and CSV filenames.
    """
    seq_rmse_xf = np.stack(seq_rmse_xf, axis=0)        # (n_inits, lead_hours, V)
    q_level_idx = [i for i, v in enumerate(VARIABLES) if v.startswith("q-")]
    seq_rmse_xf[:, :, q_level_idx] *= 1000             # 湿度 q：kg/kg → g/kg（此前注误标 surface）
    seq_acc_xf = np.stack(seq_acc_xf, axis=0)
    seq_activity_xf = np.stack(seq_activity_xf, axis=0)
    seq_activity_xf[:, :, q_level_idx] *= 1000
    seq_pred_rmse_xf = np.stack(seq_pred_rmse_xf, axis=0)
    seq_pred_rmse_xf[:, :, q_level_idx] *= 1000

    # mean over init times: (lead_hours, V) → 转置为 (V, lead_hours) 给 CSV
    rmse_mean = np.mean(seq_rmse_xf, axis=0).T          # (V, lead_hours)
    acc_mean = np.mean(seq_acc_xf, axis=0).T
    activity_mean = np.mean(seq_activity_xf, axis=0).T
    pred_rmse_mean = np.mean(seq_pred_rmse_xf, axis=0).T

    horizon = rmse_mean.shape[1]
    lead_cols = [f"t+{i * dt:03d}h" for i in range(horizon)]

    # === Plot：plot_forecast_metrics 期望输入 (lead_hours, V) ===
    figures = plot_forecast_metrics(
        np.mean(seq_rmse_xf, axis=0),                  # (lead_hours, V)
        np.mean(seq_acc_xf, axis=0),
        np.mean(seq_activity_xf, axis=0),
        VARIABLES,
    )
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    save_forecast_plots(figures, output_dir=figures_dir)

    # === 写 CSV：rows=变量名（VARIABLES），cols=lead time ===
    os.makedirs(output_dir, exist_ok=True)

    def _save_csv(metric_array, name):
        df = pd.DataFrame(metric_array, index=VARIABLES, columns=lead_cols)
        df.index.name = "variable"
        path = f"{output_dir}/{forecast_name}_{resolution_tag}_{name}.csv"
        df.to_csv(path)
        return path

    rmse_csv = _save_csv(rmse_mean, "rmse")
    acc_csv = _save_csv(acc_mean, "acc")
    activity_csv = _save_csv(activity_mean, "activity")
    pred_rmse_csv = _save_csv(pred_rmse_mean, "pred_rmse")

    log.info("Saved CSVs:")
    log.info("  %s", rmse_csv)
    log.info("  %s", acc_csv)
    log.info("  %s", activity_csv)
    log.info("  %s", pred_rmse_csv)

    return {
        "variables": np.array(VARIABLES),
        "rmse": rmse_mean,                              # (V, lead_hours)
        "acc": acc_mean,
        "activity": activity_mean,
        "pred_rmse": pred_rmse_mean,
        "metrics_csv_dir": output_dir,
        "rmse_csv": rmse_csv,
        "acc_csv": acc_csv,
        "activity_csv": activity_csv,
        "pred_rmse_csv": pred_rmse_csv,
        "figures_dir": figures_dir,
        "lead_hours": lead_cols,
    }


# ---------------------------------------------------------------------------
# 预报场写出（NWP-Benchmark 格式：每 init time 一个子目录，每 lead time 一个 NC）
# ---------------------------------------------------------------------------
def save_forecast_nc(
    field_norm: np.ndarray,
    init_time: datetime,
    lead_hours: int,
    forecast_root: str,
    scale_dir: str,
    resolution: str = "1p0",
) -> str:
    """写出单个 (init_time, lead_hours) 的预报场到 NWP-Benchmark 格式 NetCDF。

    目录结构（NWP-Benchmark Saver 约定）：
      ${forecast_root}/<init:%Y%m%d%H>/<init:%Y%m%d>-<lead:02d>.nc
      （如 forecast/2023010100/20230101T00-06.nc；lead=0 即分析场）

    文件内容：
      - 按 GRIB short name 拆分：t2m/msl/u10/v10（地面）+ z/u/v/t/q（气压层）
      - dims: time / latitude / longitude（气压另有 plev_<short>）
      - data: 物理单位（反归一化后写盘）
      - attrs: initial_time, forecast_lead_time, generator 等

    Args:
        field_norm: (V, H, W) numpy，归一化空间（forecast 模型输出）
        init_time: 起报时刻 datetime
        lead_hours: 预报时长（小时）
        forecast_root: 预报场根目录（通常 ${output_dir}/forecast）
        scale_dir: 归一化参数目录（含 normalized_mean_std）
        resolution: "1p0"（默认，原生）或 "0p25"（LR 场插值到 721×1440）

    Returns:
        写出的 NC 文件路径
    """
    mean, std = get_normalize(scale_dir, VARIABLES)
    field_raw = field_norm * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)
    return nwp.save_field(
        field_raw, init_time, lead_hours, root=forecast_root,
        resolution=resolution, extra_attrs={"kind": "forecast"},
    )


def eval_forecast(
    loader: Callable[[datetime], tuple],
    ckpt_path: str,
    config: dict,
    output_dir: str,
    init_loader: Callable[[datetime], np.ndarray] | None = None,
) -> dict:
    """Run an AR forecast evaluation against ERA5 truth and write metrics/plots.

    Parameters
    ----------
    loader : Callable[[datetime], (era5, clim)]
        Per-lead-time data loader. Receives the evaluation `datetime` for
        the current lead, returns `(era5, clim)` arrays of shape
        `(1, V, H, W)` in physical units.
    ckpt_path : str
        Directory containing the forecast checkpoint passed to
        `load_forecast_ckpt`.
    config : dict
        Must contain: era5_lr_dir, era5_hr_dir, forecast_hours, dt,
        forecast_name, device. Additionally start_year, end_year and
        decorrelation_hours unless `init_times` is given. Optional:
        `init_times` (explicit list of init datetimes; defaults to the
        Jan 1 -> Dec 21 walk over `start_year`), resolution_tag (default
        "1p0deg"), and either `forecast_mean_key`/`forecast_std_key`
        ("lr" or "hr") to pick which normalisation pair denormalises the
        metric comparison (default "lr").
    output_dir : str
        Directory in which the RMSE/ACC/activity/pred-RMSE CSVs and
        the `figures/` sub-directory are written.
    init_loader : Callable[[datetime], np.ndarray], optional
        Supplies the initial condition for each init time as a
        `(1, V, H, W)` array in physical units — normalisation with the
        LR mean/std is applied here, same as for the built-in ERA5 path.
        Defaults to reading ERA5 from `config["era5_lr_dir"]`.

    Returns
    -------
    dict
        Metrics dict with keys: variables, rmse, acc, activity,
        pred_rmse, metrics_csv_dir, rmse_csv, acc_csv, activity_csv,
        pred_rmse_csv, figures_dir, lead_hours.
    """
    required = _CONFIG_KEYS
    if "init_times" not in config:
        required = required + _TIMEWALK_CONFIG_KEYS
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"eval_forecast config missing keys: {missing}")

    os.makedirs(output_dir, exist_ok=True)

    forecast_model = _build_model(
        config.get("forecast_config", "configs/xichen_forecast.json")
    )
    forecast_model = load_forecast_ckpt(
        ckpt_path, config["forecast_name"], forecast_model
    )
    forecast_model.to(config["device"], dtype=torch.float32)
    forecast_model.eval()

    era5_lr_mean, era5_lr_std = get_normalize(
        f"{config['era5_lr_dir']}/normalized_mean_std", VARIABLES
    )
    forecast_pair = config.get("forecast_pair", "lr")
    if forecast_pair == "lr":
        era5_hr_mean, era5_hr_std = None, None
    else:
        era5_hr_mean, era5_hr_std = get_normalize(
            f"{config['era5_hr_dir']}/normalized_mean_std", VARIABLES
        )
    if forecast_pair == "lr":
        forecast_mean, forecast_std = era5_lr_mean, era5_lr_std
    elif forecast_pair == "hr":
        forecast_mean, forecast_std = era5_hr_mean, era5_hr_std
    else:
        raise ValueError(f"forecast_pair must be 'lr' or 'hr', got {forecast_pair!r}")

    # Default denorm: simple inverse normalisation on the model's LR grid.
    # Wrappers may pass a richer callable (e.g. geographic interpolation +
    # HR denormalisation) via config["denorm_fn"] to match their grid.
    base_denorm = lambda s: _denorm(s, forecast_mean, forecast_std)
    denorm = config.get("denorm_fn", base_denorm)

    init_times = config.get("init_times")
    if init_times is None:
        init_times = _timewalk_init_times(config)

    seq_rmse_xf, seq_acc_xf, seq_activity_xf, seq_pred_rmse_xf = [], [], [], []
    seq_timing = []   # 2026-08-08 新增：每 init time 的 forecast + 评估总耗时

    # === 数据并行：按 eval_batch_size 切分 init times，每批一次 forward ===
    eval_batch_size = max(1, int(config.get("eval_batch_size", 1)))
    log.info(
        "Forecast eval: %d init times, eval_batch_size=%d (%d 批), forecast_dt=%dh",
        len(init_times), eval_batch_size,
        (len(init_times) + eval_batch_size - 1) // eval_batch_size,
        config["dt"],
    )

    for batch_start in range(0, len(init_times), eval_batch_size):
        batch_end = min(batch_start + eval_batch_size, len(init_times))
        batch_times = init_times[batch_start:batch_end]
        N = len(batch_times)

        # 1) 加载本批所有 IC
        era5_inits = []
        for current_time in batch_times:
            if init_loader is not None:
                era5 = init_loader(current_time)
            else:
                init_file = os.path.join(
                    config["era5_lr_dir"],
                    f"{current_time.year:04d}",
                    f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
                    f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}.npy",
                )
                if os.path.exists(init_file):
                    era5 = get_era5(init_file, (-1, 181, 360))
                else:
                    # 缺失初始场：显式报错（此前会 NameError 或静默复用上一时次 IC）。
                    raise FileNotFoundError(
                        f"[eval_forecast] missing ERA5 initial file: {init_file}"
                    )
            # get_era5 返回 (V, H, W)（reshape(-1,181,360) 恒 3 维）；init_loader 返回
            # (1, V, H, W)。统一 squeeze() 去掉所有 size-1 前导维 → (V, H, W)。
            era5_inits.append(np.asarray(era5).squeeze())
        # (N, V, H, W) physical units → 归一化
        era5_init_batch = torch.from_numpy(
            (np.stack(era5_inits, axis=0) - era5_lr_mean) / era5_lr_std
        ).to(torch.float32)

        # 2) 批量 AR rollout（向后兼容：N=1 时走 batched 路径，输出与单 init 等价）
        # === 计时 1/2：AR rollout 阶段 ===
        t_ar_start = time.time()
        seq_forecast_batch, seq_log_var_batch = _autoregressive_rollout_batched(
            forecast_model,
            era5_init_batch,
            config["forecast_hours"],
            config["dt"],
            config["device"],
        )
        t_ar_elapsed = time.time() - t_ar_start
        # seq_forecast_batch: (N, horizon, V, H, W)

        # 3) 对 batch 内每个 init time 独立评估（与原逐 init 评估字节级一致）
        forecast_root = os.path.join(output_dir, "forecast")
        for j, current_time in enumerate(batch_times):
            t_per_init = time.time()  # === 计时 2/2：单 init 评估（NC 写出 + metrics）===
            seq_forecast = seq_forecast_batch[j]      # (horizon, V, H, W)
            seq_log_var = seq_log_var_batch[j]        # (horizon, V, H, W)

            # === 写预报场：每 init time 一个子目录，每 lead time 一个 NC ===
            horizon = seq_forecast.shape[0]
            for i_lead in range(horizon):
                lead_h = i_lead * config["dt"]
                nc_path = save_forecast_nc(
                    field_norm=seq_forecast[i_lead],
                    init_time=current_time,
                    lead_hours=lead_h,
                    forecast_root=forecast_root,
                    scale_dir=f"{config['era5_lr_dir']}/normalized_mean_std",
                    resolution=config.get("output_resolution", "1p0"),
                )
            log.info(
                "Saved %d NC files for init %s → %s",
                horizon, current_time.isoformat(),
                os.path.join(forecast_root, current_time.strftime('%Y%m%d%H')),
            )

            rmse_xf, acc_xf, activity_xf, pred_rmse_xf = _eval_one_init(
                loader,
                denorm,
                seq_forecast,
                seq_log_var,
                current_time,
                config["forecast_hours"],
                config["dt"],
                era5_lr_std if forecast_pair == "lr" else era5_hr_std,
            )

            seq_rmse_xf.append(rmse_xf)
            seq_acc_xf.append(acc_xf)
            seq_activity_xf.append(activity_xf)
            seq_pred_rmse_xf.append(pred_rmse_xf)

            # === 记录单 init 耗时（AR rollout 时间均摊到本批 N 个 init；评估为单 init 独占）===
            t_per_init_elapsed = time.time() - t_per_init
            seq_timing.append({
                "init_time": current_time.isoformat(),
                "t_total_s": t_per_init_elapsed,
                "t_ar_shared_s_per_init": t_ar_elapsed / N,
                "horizon": seq_forecast.shape[0],
                "forecast_dt_h": config["dt"],
            })

            log.info(
                "Forecast Z500 [batch %d/%d, %d/%d] RMSE: %s at %s (took %.2fs)",
                batch_start // eval_batch_size + 1,
                (len(init_times) + eval_batch_size - 1) // eval_batch_size,
                j + 1, N,
                rmse_xf[:, 11], current_time, t_per_init_elapsed,
            )

    # === 2026-08-08 新增：写 timing CSV + log mean forecast time per init ===
    if seq_timing:
        timing_df = pd.DataFrame(seq_timing)
        timing_csv = f"{output_dir}/{config['forecast_name']}_{config.get('resolution_tag', '1p0deg')}_timing.csv"
        os.makedirs(output_dir, exist_ok=True)
        timing_df.to_csv(timing_csv, index=False)
        mean_t = float(timing_df["t_total_s"].mean())
        std_t = float(timing_df["t_total_s"].std())
        log.info("=" * 60)
        log.info(
            "10-DAY FORECAST TIMING: n_inits=%d, mean=%.2fs/init (std=%.2fs)",
            len(timing_df), mean_t, std_t,
        )
        log.info("  forecast_dt=%dh, horizon=%d lead times per init",
                 config["dt"], timing_df["horizon"].iloc[0])
        log.info("  timing CSV: %s", timing_csv)
        log.info("=" * 60)

    return save_metrics_and_plots(
        seq_rmse_xf,
        seq_acc_xf,
        seq_activity_xf,
        seq_pred_rmse_xf,
        config["forecast_name"],
        output_dir,
        config.get("resolution_tag", "1p0deg"),
        dt=config["dt"],
    )


# ---------------------------------------------------------------------------
# 数据加载器（lib 函数：CLI 与 notebook 共用）
# ---------------------------------------------------------------------------
def load_init_times(path: str) -> list:
    """Read ISO timestamps from JSON, sorted ascending.

    兼容两种格式：
      1) 直接 list：``["2023-01-05T00:00:00", ...]``
      2) dict payload：``{"init_times": [...], "details": [...], "n_kept": 87}``
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("init_times")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON list, or dict with 'init_times' key")
    return sorted(datetime.fromisoformat(t) for t in raw)


def make_lr_loader(era5_lr_dir: str):
    """Loader for the 1.0deg grid: truth + climatology at LR resolution."""
    def _loader(eval_time: datetime):
        file_path = os.path.join(
            era5_lr_dir,
            f"{eval_time.year:04d}",
            f"{eval_time.year:04d}-{eval_time.month:02d}-{eval_time.day:02d}",
            f"{eval_time.hour:02d}:{eval_time.minute:02d}:{eval_time.second:02d}.npy",
        )
        era5 = get_era5(file_path, (-1, 181, 360))
        clim = get_climatology(
            f"{era5_lr_dir}/climatology_np181x360_2010_2021",
            (-1, 181, 360), eval_time, VARIABLES,
        )
        return era5, clim
    return _loader


def ic_path(ic_dir: str, t: datetime) -> str:
    """DA 6h 分析场的文件名约定（``dacycle.common.save_nc`` → ``nwp``，lead=0）。"""
    return nwp.analysis_ic_path(ic_dir, t)


def check_all_present(ic_dir: str, init_times: list) -> None:
    """Fail fast if any init time has no analysis on disk."""
    missing = [t for t in init_times if not os.path.exists(ic_path(ic_dir, t))]
    if missing:
        listed = "\n  ".join(t.isoformat() for t in missing)
        raise FileNotFoundError(
            f"{len(missing)}/{len(init_times)} init times have no analysis in "
            f"{ic_dir}:\n  {listed}"
        )


def make_dacycle_init_loader(ic_dir: str):
    """Loader for the DA 6h analysis: (1, V, 181, 360) in physical units (fp32).

    从 NWP-Benchmark 格式 NetCDF（9 个按 GRIB short name 拆分的变量）经
    ``nwp.load_field`` 重组回 VARIABLES 顺序的 (V, H, W)，再补 batch 维。
    """
    def _loader(init_time: datetime):
        arr = nwp.load_field(ic_path(ic_dir, init_time))  # (V, 181, 360) fp32
        return arr[None, ...]  # (1, V, 181, 360)
    return _loader
