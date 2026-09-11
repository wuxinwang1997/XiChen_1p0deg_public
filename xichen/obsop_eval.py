# -*- coding: utf-8 -*-
"""观测算子（ObsOp）OMB 评估：逐 3h 时次对 1 年（或 debug 2 天）亮温做
out-vs-tgt 统计，输出 ``{save_dir}/{obs_name}/avg_obs_error.npz``——
该文件即级联 DA 中 ``Model_H`` 5σ QC 与 ``Obs_WeighedL2Norm`` R⁻¹ 所用的观测误差估计。
"""
import logging
import os
import sys
from datetime import datetime

import json
import numpy as np
import torch
from dateutil.relativedelta import relativedelta

from xichen.models.obsoperator import XiChenObsOp
from xichen.data import (
    VARIABLES,
    get_era5,
    get_normalize,
    get_sat,
    prepare_sat,
    sat_auxiliary_vars,
    sat_tmbrs_vars,
)
from xichen.ckpt import load_obsop_ckpt
from xichen.plotting import plot_obsop_omb

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(name)s - %(levelname)s - %(message)s')


def eval_obsoperator(
    era5_dir,
    obs_name,
    obs_dir,
    save_dir,
    start_year,
    end_year,
    model_name,
    debug,
    device,
    obsop_config=None,
):
    """对 ``obs_name``（atms/amsua/mhs/hrs4 之一）跑全年 OMB 评估。

    Args:
        era5_dir: ERA5 1.0° npy 根目录。
        obs_name: 观测源名。
        obs_dir: 观测 npy 根目录（含 ``1b<obs>_merged_npy_1.0deg/``）。
        save_dir: 输出根目录（``{save_dir}/{obs_name}/avg_obs_error.npz``）。
        start_year / end_year: 评估年份区间 [start_year, end_year)；debug=True 时只跑
            start_year 年 1 月 1–3 日。
        model_name: 扁平 ``.ckpt`` 文件路径（推荐），或旧版 logs 目录下的 run 名。
        debug: True 时只评估 2 天。
        device: ``torch.device``。
        obsop_config: 观测算子超参 JSON；默认 ``configs/{obs_name}_obsop.json``。
    """
    if obsop_config is None:
        obsop_config = f"configs/{obs_name}_obsop.json"
    with open(obsop_config) as f:
        model_params = json.load(f)

    obsop_model = XiChenObsOp(**model_params)
    # model_name 兼容两种写法：
    #   - run 名（旧，如 train_atms_obsop_20260608）→ load_obsop_ckpt 拼 runs/checkpoints/best.ckpt
    #   - 扁平 .ckpt 文件路径（DCU 服务器 xichen_obsop_atms.ckpt）→ 直接加载该文件
    is_flat_file = model_name.endswith(".ckpt")
    obsop_model = load_obsop_ckpt(
        model_name if is_flat_file else "logs",
        "" if is_flat_file else model_name,
        obsop_model,
    )
    obsop_model.to(device, dtype=torch.float32)

    obs_dict = prepare_sat[obs_name](
        obs_dir,
        obsop_model.out_sat_vars,
        sat_tmbrs_vars[obs_name]
    )

    era5_mean, era5_std = get_normalize(f"{era5_dir}/normalized_mean_std", VARIABLES)

    if debug:
        start_time = datetime(start_year, 1, 1, 0, 0)
        end_time = datetime(start_year, 1, 3, 0, 0)
    else:
        start_time = datetime(start_year, 1, 1, 0, 0)
        end_time = datetime(end_year, 1, 1, 0, 0)

    # Initialize dictionaries to store all data for each channel
    tgt_tmbrs_data = {var: [] for var in obsop_model.out_sat_vars}
    out_tmbrs_data = {var: [] for var in obsop_model.out_sat_vars}
    val_obserr_data = {var: [] for var in obsop_model.out_sat_vars}
    total_mse, total_var = 0, 0

    current_time = start_time
    num_samples = 0

    while current_time < end_time:
        era5_path = os.path.join(
            era5_dir,
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}.npy",
        )
        if os.path.exists(era5_path):
            era5 = get_era5(era5_path, (-1, 181, 360))
        else:
            # 缺失 ERA5 时次：显式跳过该样本，禁止静默复用上一时次的 era5 污染 OMB
            # 统计（此前会 NameError 或在后续缺失时复用上一时次）。
            logging.warning(f"skip missing ERA5 slot @ {current_time} for {obs_name}")
            current_time = current_time + relativedelta(hours=3)
            continue

        era5 = torch.from_numpy((era5 - era5_mean) / era5_std)

        np_tmbrs_data, np_auxiliary_data, np_mask = get_sat[obs_name](
            obs_dir=obs_dir,
            obs_time=current_time,
            auxiliary_vars=sat_auxiliary_vars[obs_name],
            tmbrs_vars=sat_tmbrs_vars[obs_name],
            obs_dict=obs_dict,
            num_lat=181,
            num_lon=360,
        )

        tmbrs_tensor = torch.as_tensor((np_tmbrs_data - obs_dict["tmbrs_mean"]) / obs_dict["tmbrs_std"])  # (n_tmbrs, H, W)
        auxiliary_tensor = torch.as_tensor(np_auxiliary_data, dtype=torch.float32)                        # (n_aux, H, W)
        mask_tensor = torch.as_tensor(np_mask, dtype=torch.float32)                                       # (H, W)
        # XiChenObsOp.forward 期望 sat 为 [B, C_total, H, W]（aux 在前、tmbrs 在后，见 load_obs_window）。
        # 修复 BUG: auxiliary 此前被多 unsqueeze 一维，导致与 3-D tmbrs 无法沿通道维拼接。
        sat_tensor = torch.concat([auxiliary_tensor, tmbrs_tensor], dim=0)  # (n_aux + n_tmbrs, H, W)
        sat_tensor = sat_tensor.unsqueeze(0) * mask_tensor[None, None]       # (1, C_total, H, W)

        with torch.no_grad():
            out_tmbrs, log_var, tgt_tmbrs = obsop_model(
                era5.to(device, dtype=torch.float32),
                sat_tensor.to(device, dtype=torch.float32),
                mask_tensor[None, None].to(device, dtype=torch.float32),   # (1,1,H,W) → 跨通道广播
                use_checkpoint=True   # 推理：节省显存（gradient checkpointing）
            )

        out_tmbrs = mask_tensor.detach().cpu().numpy() * out_tmbrs.detach().cpu().numpy()
        log_var = log_var.detach().cpu().numpy()
        tgt_tmbrs = mask_tensor.detach().cpu().numpy() * tgt_tmbrs.detach().cpu().numpy()
        sat_mask = mask_tensor.detach().cpu().numpy()
        var = np.exp(log_var) * sat_mask
        tgt_sat_var_ids = np.array([sat_tmbrs_vars[obs_name].index(item) for item in obsop_model.out_sat_vars])

        out_tmbrs = (obs_dict["tmbrs_std"][:, tgt_sat_var_ids] * out_tmbrs + obs_dict["tmbrs_mean"][:, tgt_sat_var_ids]) * sat_mask
        tgt_tmbrs = (obs_dict["tmbrs_std"][:, tgt_sat_var_ids] * tgt_tmbrs + obs_dict["tmbrs_mean"][:, tgt_sat_var_ids]) * sat_mask
        var = sat_mask * (obs_dict["tmbrs_std"][:, tgt_sat_var_ids] * np.sqrt(var)) ** 2

        val_rmse = sat_mask * np.sqrt((out_tmbrs - tgt_tmbrs) ** 2)
        val_obserr = sat_mask * np.sqrt(var)

        # Process each channel (level) separately
        for channel in range(len(obsop_model.out_sat_vars)):
            # Extract data for current channel
            tgt_tmbrs_channel = tgt_tmbrs[0, channel, :, :]
            out_tmbrs_channel = out_tmbrs[0, channel, :, :]
            val_obserr_channel = val_obserr[0, channel, :, :]
            mask_channel = np_mask

            # Get channel name
            channel_name = obsop_model.out_sat_vars[channel]

            # Get indices where mask is 1
            valid_indices = np.where(mask_channel == 1)

            # Extract masked data
            tgt_tmbrs_masked = tgt_tmbrs_channel[valid_indices]
            out_tmbrs_masked = out_tmbrs_channel[valid_indices]
            val_obserr_masked = val_obserr_channel[valid_indices]

            # Store data for this channel
            tgt_tmbrs_data[channel_name].extend(tgt_tmbrs_masked)
            out_tmbrs_data[channel_name].extend(out_tmbrs_masked)
            val_obserr_data[channel_name].extend(val_obserr_masked)

        total_mse += np.sum(sat_mask * ((out_tmbrs - tgt_tmbrs) ** 2), axis=(0, -2, -1)) / (sat_mask.sum(axis=(0, -2, -1)) + 1e-6)
        total_var += np.sum(sat_mask * var, axis=(0, -2, -1)) / (sat_mask.sum(axis=(0, -2, -1)) + 1e-6)
        num_samples += 1

        current_time = current_time + relativedelta(hours=3)

    total_rmse = (total_mse / num_samples) ** 0.5
    total_obserr = (total_var / num_samples) ** 0.5

    obs_sigma = {var: [] for var in obsop_model.out_sat_vars}
    for j in range(val_rmse.shape[1]):
        logging.info(f"{obs_name} ObsOp RMSE of {obsop_model.out_sat_vars[j]} is: {total_rmse[j]}")
        logging.info(f"{obs_name} ObsOp predict error of {obsop_model.out_sat_vars[j]} is: {total_obserr[j]}")
        obs_sigma[obsop_model.out_sat_vars[j]].append(total_rmse[j])

    np.savez(
        f"{save_dir}/{obs_name}/avg_obs_error.npz",
        **obs_sigma,
    )

    # After processing all time steps, create plots for each variable
    for channel_name in obsop_model.out_sat_vars:
        tgt_tmbrs_values = np.array(tgt_tmbrs_data[channel_name])
        out_tmbrs_values = np.array(out_tmbrs_data[channel_name])

        if len(tgt_tmbrs_values) > 0:
            logging.info(f"Creating plot for {channel_name} with {len(tgt_tmbrs_values)} total data points")
            # Create a mask of all ones since we've already filtered the data
            mask_ = np.ones_like(tgt_tmbrs_values)
            plot_obsop_omb(
                tgt_tmbrs_values=tgt_tmbrs_values,
                out_tmbrs_values=out_tmbrs_values,
                mask=mask_,
                variable_name=channel_name,
                plot_dir=f"{save_dir}/{obs_name}",
            )
        else:
            logging.info(f"Warning: No valid data for {channel_name} across all time steps")
