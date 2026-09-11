"""Cascade DA 一年同化循环推理 — 公共共享模块（XiChen）。

拆分自 ``inference/dacycle.py``（2026-08-10）。本模块承载两个入口脚本
（``dacycle_det.py`` 确定性 / ``dacycle_ens.py`` 集合）共用的：
  - 模型 / obs / 指标 加载与计算函数
  - main() 的公共骨架（build_common_parser / load_config_and_env / run_dacycle_loop）

函数体与拆分前快照 ``inference/archive/dacycle.py.2026-08-10.pre-split`` 逐字一致。
"""
import argparse
import functools
import json
import logging
import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
from dateutil.relativedelta import relativedelta

# 让顶层目录的 src/ inference/ 可被 import

from xichen.data import (
    VARIABLES,
    get_climatology,
    get_era5,
    get_normalize,
    get_sat,
    prepare_atms,
    prepare_amsua,
    prepare_mhs,
    prepare_hrs4,
    prepare_prepbufr,
    prepare_satwnd,
    sat_auxiliary_vars,
    sat_tmbrs_vars,
)
from xichen import nwp
from xichen.ckpt import (
    load_forecast_ckpt,
    load_obsop_ckpt,
)
from xichen.metrics import (
    weighted_acc_channels,
    weighted_bias_torch_channels,
    weighted_rmse_channels,
)
from xichen.models.cascade import Solver
from xichen.models.varcost import (
    Model_H,
    Model_Var_Cost,
    Obs_WeighedL2Norm,
)
from xichen.models.da import XiChenDA
from xichen.models.forecast import XiChenForecast
from xichen.models.obsoperator import XiChenObsOp

# ---------------------------------------------------------------------------
# 日志（对齐参考脚本风格：stdout + 时间戳）
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("xichen.dacycle")

# 微波 obs（需要 ObsOp 子模型），与训练 cascade 一致
SAT_LIST = ["atms", "amsua", "mhs", "hrs4"]
# 完整 cascade 6 obs（含常规 obs prepbufr / satwnd，无 ObsOp）
OBS_LIST = SAT_LIST + ["prepbufr", "satwnd"]

# low-res 网格尺寸（181 × 360）
H, W = 181, 360


def load_models(cfg, cascade_cfg, obs_dict, device):
    """一次性加载所有组件到 ``device``，全部 ``eval()`` 模式。

    关键修复（critical-review R2/R4/R5/R6/R7）：
      - XiChenForecast / XiChenDA 第一位置参数 ``default_vars`` 显式传 VARIABLES
      - obs_err 路径按微波 vs 常规分派（avg_obs_error.npz vs obs_sigma.npz）
      - Model_H / Obs_WeighedL2Norm 接受 1D np.ndarray
      - Model_Var_Cost 嵌套 Obs_WeighedL2Norm

    返回：forecast, obsops, da_models, h_models, varcost_models, solver, microwave_prep, conventional_prep
    microwave_prep 是 {sat: prepare_<sat> dict}，包含 get_atms 等需要的
    meta_data + satellite_height_scaler + *_quality_flags_scaler 等字段
    （obs_dict_6obs.json 只存 vars 列表，不含这些 schema）。
    """
    # 1) Forecast（fc_cfg 已含 default_vars；此处不传避免重复）
    fc_cfg_path = cfg["forecast_config"]
    fc_cfg = json.load(open(fc_cfg_path))
    fc_cfg_no_dv = {k: v for k, v in fc_cfg.items() if k != "default_vars"}
    forecast = XiChenForecast(VARIABLES, **fc_cfg_no_dv)
    forecast = load_forecast_ckpt(cfg["ckpt_forecast"], "", forecast).to(device).eval()

    # 2) 4 ObsOps
    obsops = torch.nn.ModuleDict()
    for sat in SAT_LIST:
        cfg_path = cfg["obsop_configs"][sat]
        oc = json.load(open(cfg_path))
        # 4 个位置参数（default_vars / all_sat_vars / in_sat_vars / out_sat_vars）
        # 都由 JSON 直接提供，其余 Swin kwargs 经 kwargs 传入
        oc_kwargs = {
            k: v for k, v in oc.items()
            if k not in ("default_vars", "all_sat_vars", "in_sat_vars", "out_sat_vars")
        }
        m = XiChenObsOp(VARIABLES, oc["all_sat_vars"], oc["in_sat_vars"], oc["out_sat_vars"], **oc_kwargs)
        m = load_obsop_ckpt(cfg["ckpt_obsop"][sat], "", m).to(device).eval()
        obsops[sat] = m

    # 3) 6 DAs（从 cascade ckpt 的 da_model_state_dict 切分子字典 — 模板见
    #    src/pipeline/assimilate/cascade/random_bg_trainer.py:400-402）
    # 支持两种布局：cfg["ckpt_cascade_da"] 可以是旧目录（追加 runs/checkpoints/best.ckpt），
    # 也可以是扁平 .ckpt 文件（DCU 服务器 xichen_cascade_da_....ckpt）。
    if os.path.isfile(cfg["ckpt_cascade_da"]):
        cascade_ckpt_path = cfg["ckpt_cascade_da"]
    else:
        cascade_ckpt_path = os.path.join(
            cfg["ckpt_cascade_da"], "runs", "checkpoints", "best.ckpt"
        )
    cascade_ckpt = torch.load(cascade_ckpt_path, map_location="cpu")
    da_state = cascade_ckpt["da_model_state_dict"]
    da_models = torch.nn.ModuleDict()
    for obs in OBS_LIST:
        da_kwargs = {k: v for k, v in cascade_cfg["da_models"][obs].items() if k != "default_vars"}
        da = XiChenDA(VARIABLES, **da_kwargs)
        sub = {k[len(f"{obs}."):]: v for k, v in da_state.items() if k.startswith(f"{obs}.")}
        if not sub:
            raise RuntimeError(
                f"No DA weights found for obs={obs!r} in cascade ckpt; check prefix."
            )
        da.load_state_dict(sub, strict=True)
        da_models[obs] = da.to(device).eval()

    # 4) H / VarCost — 每个 obs 加载 obs σ；按 obs 类型分派路径
    h_models = torch.nn.ModuleDict()
    varcost_models = torch.nn.ModuleDict()
    for obs in OBS_LIST:
        obs_err = _load_obs_err(obs, cfg["obs_dir"], obs_dict)  # 1D np.ndarray, 长度 = C_obs（avg_obs_error/obs_sigma）
        h_models[obs] = Model_H(obs_err).to(device)
        varcost_models[obs] = Model_Var_Cost(Obs_WeighedL2Norm(obs_err)).to(device)

    # 5) Solver
    solver = Solver(dt=cascade_cfg["solver"]["dt"]).to(device).eval()

    # 6) Microwave prepare cache — get_atms 等函数内部访问
    #    obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"] 等 schema/scaler 字段。
    #    必须在启动时构建一次，传给 load_obs_window 用作 get_<sat> 的 5th arg。
    microwave_prep = {}
    for sat, prep_fn in zip(
        SAT_LIST, (prepare_atms, prepare_amsua, prepare_mhs, prepare_hrs4)
    ):
        # prepare_<sat>(obs_dir, out_<sat>_vars, <sat>_tmbrs_vars)
        # out_<sat>_vars：仅 metadata 列表（atms:15; amsua/mhs/hrs4:10）— 即 sat_auxiliary_vars
        # <sat>_tmbrs_vars：必须用 sat_tmbrs_vars[sat]（完整集：atms:22, amsua:15, mhs:5, hrs4:19）
        # 这样 prepare_* 构造的 tmbrs_mean/tmbrs_std 长度 == npy 文件中 tmbrs 通道数，
        # load_obs_window 才能对 get_sat 返回的全部 tmbrs 做归一化（Bug A 修复）
        microwave_prep[sat] = prep_fn(
            cfg["obs_dir"],
            sat_auxiliary_vars[sat],
            sat_tmbrs_vars[sat],          # 全部 tmbrs 名（numerical order），匹配 npy 通道
        )

    # 6) Conventional obs prepare cache（Bug C 修复）：
    #    prepbufr/satwnd 是物理单位（t/u/v/z/msl 量级），但训练时 Solver 直接消费 normalize 空间；
    #    必须用 prepare_prepbufr/prepare_satwnd（内部用 normalize_mean_std npz）归一化
    conventional_prep = {}
    for obs in ("prepbufr", "satwnd"):
        prep_fn = prepare_prepbufr if obs == "prepbufr" else prepare_satwnd
        conventional_prep[obs] = prep_fn(
            cfg["obs_dir"], cfg["era5_lr_dir"],
            obs_dict["conventional"][obs]["vars"],
        )

    return (
        forecast, obsops, da_models, h_models, varcost_models, solver,
        microwave_prep, conventional_prep,
    )


def _load_obs_err(obs_name, obs_dir, obs_dict):
    """加载 obs σ 1D 数组（用于 ``Model_H`` 5σ QC + ``Obs_WeighedL2Norm`` 的 R⁻¹）。

    路径对齐 ``src/pipeline/assimilate/cascade/random_bg_trainer.py:253-260``：
      - 微波 (atms/amsua/mhs/hrs4) → ``1b<sat>_merged_npy_1.0deg/avg_obs_error.npz``
      - prepbufr                  → ``GDAS_prepbufr_merged_npy_1.0deg/obs_sigma.npz``（GDAS_ 前缀）
      - satwnd                    → ``satwnd_merged_npy_1.0deg/obs_sigma.npz``

    通道顺序以消费端通道名（``obs_dict`` 的 ``tmbrs_vars`` / ``vars``）为准：
    优先按名称逐一取 σ —— 写入端 obsop_eval.py 以 ``out_sat_vars`` 为 key 落盘，
    名字即通道本身，对任何存储顺序都免疫；若名称与 key 不匹配，则回退为镜像训练侧
    random_bg_trainer.py:263 的 ``obs_err.keys()`` 落盘序迭代（**不得 sorted**——
    字母序会把 tmbrs_10..22 排在 tmbrs_2..9 之前，与消费通道序错位，让 5σ QC 与
    R⁻¹ 权重挂到错误通道）。
    """
    if obs_name in SAT_LIST:
        path = os.path.join(obs_dir, f"1b{obs_name}_merged_npy_1.0deg", "avg_obs_error.npz")
        expected_vars = obs_dict["microwave"][obs_name]["tmbrs_vars"]
    elif obs_name == "prepbufr":
        path = os.path.join(obs_dir, "GDAS_prepbufr_merged_npy_1.0deg", "obs_sigma.npz")
        expected_vars = obs_dict["conventional"][obs_name]["vars"]
    elif obs_name == "satwnd":
        path = os.path.join(obs_dir, "satwnd_merged_npy_1.0deg", "obs_sigma.npz")
        expected_vars = obs_dict["conventional"][obs_name]["vars"]
    else:
        raise ValueError(f"Unknown obs_name: {obs_name}")

    npz = np.load(path)
    if all(v in npz for v in expected_vars):
        # 名称键值对（推荐路径）：通道顺序 == expected_vars，天然正确。
        sigmas = [np.nan_to_num(npz[v], nan=0.0, posinf=0.0, neginf=0.0) for v in expected_vars]
    else:
        # 回退：镜像训练侧按落盘序迭代；长度由下方断言兜底。
        sigmas = [np.nan_to_num(npz[k], nan=0.0, posinf=0.0, neginf=0.0) for k in npz.keys()]
    sigmas = np.stack(sigmas, axis=0).astype(np.float32)
    if sigmas.shape[0] != len(expected_vars):
        raise ValueError(
            f"obs sigma for {obs_name}: got {sigmas.shape[0]} channels, "
            f"expected {len(expected_vars)} <{expected_vars}>; "
            f"<avg_obs_error|obs_sigma>.npz does not match the stream schema."
        )
    return sigmas


def _load_normalize_std(obs_name, obs_dir, obs_dict, era5_lr_dir):
    """加载归一化 std 1D 数组（用于 cascade ``std_dict`` —— 给 Solver.var_cost 的
    Model_H QC omb 乘子 + Obs_WeighedL2Norm 的 ``var = std²`` 权重）。

    关键修复 — 训练 vs 推理路径不一致：

      - **训练**（``random_bg_trainer.py:716``）：
        ``std_dict[name] = microwave_transforms[name]["std"]``（来自
        ``1b<sat>_merged_npy_1.0deg/normalize_std.npz`` 按 tmbrs_vars 顺序）。
        VarCost 权重 = ``R⁻¹ · σ_data_norm²`` = ``(1/σ_obs²) · σ_data²``。
      - **推理（旧 bug）**：cascade 收到 ``_load_obs_sigma``（avg_obs_error.npz），
        VarCost 权重 = ``(1/σ_obs²) · σ_obs² = 1``，与训练差 ``(σ_data/σ_obs)²`` 倍。

    修复后推理 std_dict 改用本函数加载 normalize_std.npz，与训练对齐。

    路径（与训练 npydatamodule.py:183-186 对齐）：
      - 微波：``1b<sat>_merged_npy_1.0deg/normalize_std.npz``，按
        ``obs_dict["microwave"][obs]["tmbrs_vars"]`` 顺序
      - 常规 (prepbufr / satwnd)：``era5_lr_dir/normalized_mean_std/normalize_std.npz``，
        按 ``obs_dict["conventional"][obs]["vars"]`` 顺序
    """
    if obs_name in SAT_LIST:
        std_dir = os.path.join(obs_dir, f"1b{obs_name}_merged_npy_1.0deg")
        npz = np.load(os.path.join(std_dir, "normalize_std.npz"))
        var_names = obs_dict["microwave"][obs_name]["tmbrs_vars"]
    elif obs_name in ("prepbufr", "satwnd"):
        # 常规观测走 ERA5 normalize_std.npz 路径，
        # var 顺序由 obs_dict.conventional.<obs>.vars 决定。
        std_dir = os.path.join(era5_lr_dir, "normalized_mean_std")
        npz = np.load(os.path.join(std_dir, "normalize_std.npz"))
        var_names = obs_dict["conventional"][obs_name]["vars"]
    else:
        raise ValueError(f"Unknown obs_name: {obs_name}")

    sigmas = [
        np.nan_to_num(npz[k], nan=0.0, posinf=0.0, neginf=0.0).reshape(1)
        for k in var_names
    ]
    return np.concatenate(sigmas).astype(np.float32)


def load_xb_initial(cfg, valid_time, device):
    """加载 valid_time 对应的 climatology 作首周期 xb。

    climatology 是**物理单位**，必须归一化到训练空间才能喂给 forecast 模型。
    路径固定为 ``{era5_lr_dir}/climatology_np181x360_2010_2021/<MM-DD>/<var>.npy``，
    内部调 ``get_climatology``（data_utils.py:195）。
    normalize 用 ``get_normalize(scale_dir, VARIABLES)``（per-variable dict schema）。

    修复 Bug B：原 docstring 说"已归一化"，实际未归一化 → 首周期 xb 量级
    错误（climatology 物理单位 ≈ 几百 hPa 偏差），导致 cycle 1 forecast 失效、
    全年累计漂移。修改为 (clim - mean) / std。

    修复 HIGH#2：启动期 fail-fast 校验 climatology 目录存在 + 每个 MM-DD 子目录
    至少一个 var.npy。69 个变量 × 366 天 ≈ 25,254 个文件；缺一个文件若不早报错，
    全年 730 个周期反复抛 FileNotFound，被 main() 静默 except 吞掉，最终零输出。
    """
    clim_dir = os.path.join(cfg["era5_lr_dir"], "climatology_np181x360_2010_2021")
    if not os.path.isdir(clim_dir):
        raise FileNotFoundError(
            f"climatology directory missing: {clim_dir}\n"
            f"（HIGH#2 — 全年 730 周期会反复抛 FileNotFound，零 NetCDF 输出）"
        )
    month_day = f"{valid_time.month:02d}-{valid_time.day:02d}"
    first_var = VARIABLES[0]
    first_file = os.path.join(clim_dir, month_day, f"{first_var}.npy")
    if not os.path.exists(first_file):
        raise FileNotFoundError(
            f"climatology file missing: {first_file}（HIGH#2 — get_climatology 内部会继续读 {len(VARIABLES)-1} 个文件，全部抛错）"
        )

    scale_dir = cfg["scale_dir"]
    mean, std = get_normalize(scale_dir, VARIABLES)             # (1, 69, 1, 1), (1, 69, 1, 1)
    clim = get_climatology(clim_dir, (-1, H, W), valid_time, VARIABLES)  # (1, 69, 181, 360)
    # Bug B 修复：clim 是物理单位，必须 (clim - mean) / std 才能与 forecast 输入空间匹配
    normed = ((clim - mean) / std).astype(np.float32)
    # 统一约定：load_* 返回 CPU；进入模型前调用方负责 .to(device)
    return torch.from_numpy(normed)


def _filter_active_obs(obs_order, obs_mask):
    """剔除 mask 全为 0 的 obs（窗口内无任何有效观测 → 跳过该 obs 的同化步骤）。

    判断在 CPU 上进行（load_obs_window 返回 CPU tensor，设备搬运前过滤，
    避免无谓的 GPU 往返）。返回活跃 obs 列表；被剔除的 obs 打印 warning。
    """
    active = []
    for name in obs_order:
        m = obs_mask[name]
        # mask 为 0/1；(m != 0).any() 对 NaN 安全（NaN != 0 → True，不会被误判全 0）
        if bool((m != 0).any()):
            active.append(name)
        else:
            log.warning("obs %s: mask all zero over window — skipping its assimilation step", name)
    return active


def _run_solver(solver, components, xb, obs, obs_mask, std_dict, obs_dict, obs_order):
    forecast, obsops, da_models, h_models, varcost_models, _, _, _ = components  # 8-tuple; 末尾 microwave_prep/conventional_prep 在 load_obs_window 用

    # 跳过 mask 全 0 的观测源（CPU 上判断、设备搬运前；剔除后不传给 Solver）
    active = _filter_active_obs(obs_order, obs_mask)
    if not active:
        # 全部观测源均无有效观测 → 分析场 = 背景场
        log.warning("all obs masks zero — returning background xb as analysis")
        return xb.cpu() if xb.device.type != "cpu" else xb

    obs_list = {n: SimpleNamespace(trainable=False) for n in active}
    # 设备对齐：load_obs_window 返回 CPU tensors；ObsOp/VarCost/Demo 模型已 .to(device)。
    # Solver.forward 内部把 obs[obs_name] 直接喂给 ObsOp（patch_embed Conv2d 在 GPU 上），
    # 设备不一致会抛 "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)"。
    dev = xb.device
    obs = {k: v.to(dev) for k, v in obs.items() if k in active}
    obs_mask = {k: v.to(dev) for k, v in obs_mask.items() if k in active}
    std_dict = {k: v.to(dev) for k, v in std_dict.items() if k in active}
    with torch.no_grad():
        xa, _ = solver(
            forecast_model=forecast,
            ObsOp_models=obsops,
            DA_models=da_models,
            H_models=h_models,
            VarCost_models=varcost_models,
            obs_list=obs_list,
            xb=xb,
            obs=obs,
            obs_mask=obs_mask,
            obs_dict=obs_dict,
            std_dict=std_dict,
            out_vars=VARIABLES,
        )
    # 模型输出统一搬到 CPU（约定：进入模型前 .to(device)，输出 .cpu()）
    return xa.cpu()


def load_obs_window(obs_dir, era5_lr_dir, valid_time, window_hours, dt_obs,
                    obs_order, obs_dict, microwave_prep, conventional_prep):
    """加载 [T, T+window_hours) 内每 dt_obs 步的所有 obs，返回 (obs_data, obs_mask, std_dict)。

    返回：
      obs_data[obs]: Tensor[1, n_steps, C_obs, H, W]  (B=1, T=n_steps)
      obs_mask[obs]: Tensor[1, n_steps, 1, H, W]
      std_dict[obs]: Tensor[C_obs]  (用于 Solver.var_cost 内的 R⁻¹·σ² 加权)

    微波 obs_data 通道合约（按 all_sat_vars 顺序）：
      - 前 n_aux 通道 = metadata（按 sat_auxiliary_vars[obs] 顺序，等于 all_sat_vars 前段）
      - 后 n_tmbrs 通道 = 全部 tmbrs（按 numerical order，等于 all_sat_vars 后段）
      - 总通道数 = len(all_sat_vars) = atms:37, amsua:25, mhs:15, hrs4:29
      - ObsOp 内部用 sat_var_map（按 all_sat_vars 构造）按名字查 in_sat_vars/out_sat_vars
        子集，因此 obs_data 必须覆盖 all_sat_vars 全集
      - std_dict 长度 = len(tmbrs_vars) = len(out_sat_vars) = atms:11, amsua:6, mhs:3, hrs4:8
        （avg_obs_error.npz 只存 out_sat_vars 的 σ），cascade.py 的 tgt_sat_var_ids
        基于 obs_dict["microwave"][obs]["tmbrs_vars"] 查 out_sat_vars 索引

    修复 BUG #5 + Bug A（微波 obs_dict 接口）：
      - auxiliary_vars 用 sat_auxiliary_vars[obs]（仅 metadata 列表，与 all_sat_vars 前段一致）
      - get_<sat> 第 4 参数传 all_sat_vars 的 tmbrs 段（全部 tmbrs 名，匹配 npy 文件通道数）
      - 5th arg 用 prepare_<sat> 缓存（含 meta_data + scalers）
    """
    n_steps = window_hours // dt_obs
    times = [valid_time + relativedelta(hours=i * dt_obs) for i in range(n_steps)]

    obs_data, obs_mask = {}, {}
    std_dict = {}

    for obs in obs_order:
        # 关键修复：cascade std_dict 必须加载归一化 std（normalize_std.npz），与训练
        # random_bg_trainer.py:716 ``std_dict[name] = microwave_transforms[name]["std"]`` 一致，
        # 而不是加载 obs σ（avg_obs_error.npz / obs_sigma.npz）。
        # 旧 bug 让 VarCost 加权 = R⁻¹·σ_obs² 而训练是 R⁻¹·σ_data²，差 (σ_data/σ_obs)² 倍。
        std_dict[obs] = torch.from_numpy(
            _load_normalize_std(obs, obs_dir, obs_dict, era5_lr_dir)
        )

        if obs in SAT_LIST:
            # 微波：obs_data 按 all_sat_vars 顺序，让 ObsOp 能从 sat_var_map 查表
            auxiliary_vars = sat_auxiliary_vars[obs]   # metadata-only（与 all_sat_vars 前段一致）
            n_aux = len(auxiliary_vars)
            tmbrs_names_full = obs_dict["microwave"][obs]["all_sat_vars"][n_aux:]  # 全部 tmbrs 名
            n_tmbrs = len(tmbrs_names_full)
            obs_data[obs] = torch.zeros(1, n_steps, n_aux + n_tmbrs, H, W)  # = len(all_sat_vars)
            obs_mask[obs] = torch.zeros(1, n_steps, 1, H, W)
            # Bug A 修复：tmbrs 必须归一化（K 亮温，~240-300K）才能与 ObsOp 训练空间匹配；
            # 用 prepare_<sat> 缓存里的 tmbrs_mean/tmbrs_std（已按 sat_tmbrs_vars 全集构造）
            tmbrs_mean = torch.from_numpy(
                microwave_prep[obs]["tmbrs_mean"].squeeze()  # (n_tmbrs,)
            ).float()
            tmbrs_std = torch.from_numpy(
                microwave_prep[obs]["tmbrs_std"].squeeze()    # (n_tmbrs,)
            ).float()
            for i, t in enumerate(times):
                # 修复 BUG #A: 5th arg 用 prepare_<sat> 缓存（含 meta_data + scalers）
                # 修复 in_sat_vars 处理：第 4 参数传全部 tmbrs 名（匹配 npy 通道数）
                tmbrs, aux, mask = get_sat[obs](
                    obs_dir, t,
                    auxiliary_vars,                                 # metadata 列表
                    tmbrs_names_full,                               # 全部 tmbrs 名（numerical order）
                    microwave_prep[obs],                            # prepare_<sat> 缓存
                )
                # tmbrs (n_tmbrs, H, W), aux (n_aux, H, W), mask (H, W)
                # Bug A 修复：归一化 tmbrs 到训练空间（mask=0 区域填 0，不参与归一化分母）
                tmbrs_t = torch.from_numpy(tmbrs).float()          # (n_tmbrs, H, W)
                mask_t = torch.from_numpy(mask).float()            # (H, W)
                tmbrs_norm = (tmbrs_t - tmbrs_mean[:, None, None]) / tmbrs_std[:, None, None]
                tmbrs_norm = tmbrs_norm * mask_t[None]              # mask 外强制为 0
                # 前 n_aux 通道填 metadata，后 n_tmbrs 通道填归一化后的 tmbrs
                obs_data[obs][0, i, :n_aux] = torch.from_numpy(aux)
                obs_data[obs][0, i, n_aux:n_aux + n_tmbrs] = tmbrs_norm
                obs_mask[obs][0, i, 0] = mask_t
        else:
            # 常规 obs：daw/dt 一次加载整个窗口
            vars_ = obs_dict["conventional"][obs]["vars"]
            data, mask = get_sat[obs](
                obs_dir, valid_time, window_hours, dt_obs, vars_,
            )
            # Bug C 修复：prepbufr/satwnd 是物理单位（t/u/v/z/msl 量级），
            # 必须归一化到 ERA5 state 同一训练空间；prepare_<obs> 返回的 mean/std shape (1, C, 1, 1)
            if obs == "prepbufr":
                mean_key, std_key = "prepbufr_mean", "prepbufr_std"
            elif obs == "satwnd":
                mean_key, std_key = "satwnd_mean", "satwnd_std"
            else:
                raise ValueError(
                    f"load_obs_window: unknown conventional obs {obs!r}; "
                    "expected one of {'prepbufr','satwnd'}."
                )
            mean = torch.from_numpy(conventional_prep[obs][mean_key].squeeze()).float()  # (C,)
            std = torch.from_numpy(conventional_prep[obs][std_key].squeeze()).float()    # (C,)
            data = (data.float() - mean[None, :, None, None]) / std[None, :, None, None]
            # Bug #2 修复：归一化后再乘 mask（mask=0 处强制清零），与训练 npydataset.py:558 对齐
            # 训练侧: returns transforms(prepbufrs) * prepbufr_masks
            # dacycle 旧版: transforms(x*mask) → mask=0 处 = -mean/std（非零，污染 VarCost）
            data = data * mask
            # data / mask shape = (n_steps, C_obs, H, W) — prepend B=1
            obs_data[obs] = data.unsqueeze(0)  # (1, n_steps, C, H, W)
            # Bug #1 修复：去掉 channel-mean，保留 per-channel mask
            # 训练侧 mask 是 per-channel (B, T, C, H, W)，dacycle 不应压缩 channel 维
            # 旧版 mean(dim=2) 会让 mask=0 的通道影响 broadcast 到所有 C 通道，污染 VarCost
            obs_mask[obs] = mask.unsqueeze(0)  # (1, n_steps, C, H, W) — 与训练对齐

    return obs_data, obs_mask, std_dict


def save_nc(xa_norm, T, out_dir, scale_dir, obs_order):
    """把 12h/6h 分析场写成 NWP-Benchmark 格式（lead=0 的"预报"）。

    容器重构（2026-08-08）：分析场恒为 1.0°（DA 周期按 1.0° 设计），
    内部结构对齐 NWP-Benchmark Saver——按 GRIB short name 拆分变量、
    latitude/longitude 坐标、time 维=有效时刻 T。
    """
    # Bug D 修复：normalize_*.npz 是 per-variable dict schema（每个 var 一个 key），
    # 不是单 "mean"/"std" key；用 get_normalize 与训练/反归一化逻辑统一。
    mean, std = get_normalize(scale_dir, VARIABLES)
    xa_raw = (xa_norm.cpu().numpy() * std + mean)  # (1, 69, 181, 360)
    return nwp.save_field(
        xa_raw, T, lead_hours=0, root=out_dir, resolution="1p0",
        extra_attrs={
            "kind": os.path.basename(out_dir),
            "obs_order": obs_order,
            "scale_dir": scale_dir,  # LOW#2 fix: 记录归一化上下文复现信息
        },
    )


def save_npy(xa_norm, T, out_dir, scale_dir, subdir_name="ic_6h_npy"):
    """把 6h/12h 分析场另存为 .npy（**与 ERA5 lr 训练 npy 严格同 shape**），
    供 forecast 微调直接走 NpyDataset._get_era5（无 reshape，依赖 3D 落盘）。

    路径：{out_dir}/{subdir_name}/{YYYYMMDDHH}/{HH:MM:SS}.npy
          （与 nwp_saver 子目录风格平行：YYYYMMDDHH 子目录，文件名只带时间）
    形状：(69, 181, 360) fp32，物理单位（与 ERA5 lr 完全一致）

    关键：必须 3D 不是 4D。
      - 训练侧 transforms = Normalize(mean[69], std[69])，
        mean/std 是 [C] 一维，会广播到 (C, H, W)；
        若落盘是 4D (1, C, H, W) → 广播不匹配，训练立即报错。
      - 落盘前 squeeze batch 维。
    """
    mean, std = get_normalize(scale_dir, VARIABLES)
    xa_raw = (xa_norm.cpu().numpy() * std + mean).astype(np.float32)
    # Solver 可能返回 5D (1, 1, 69, 181, 360) 或 4D (1, 69, 181, 360)；
    # 训练侧 Normalize 期望 3D (C, H, W)，需要逐步压到 3D。
    if xa_raw.ndim == 5:
        xa_raw = xa_raw.squeeze(1)                                    # → (1, 69, 181, 360)
    if xa_raw.ndim == 4 and xa_raw.shape[0] == 1:
        xa_raw = xa_raw[0]                                            # → (69, 181, 360)
    if xa_raw.ndim != 3:
        raise ValueError(
            f"save_npy 期望落盘为 3D (C, H, W)，实际拿到 ndim={xa_raw.ndim}, shape={xa_raw.shape}"
        )
    sub = T.strftime("%Y%m%d%H")
    fname = f"{T.hour:02d}:{T.minute:02d}:{T.second:02d}.npy"
    npy_dir = os.path.join(out_dir, subdir_name, sub)
    os.makedirs(npy_dir, exist_ok=True)
    npy_path = os.path.join(npy_dir, fname)
    np.save(npy_path, xa_raw)
    return npy_path


def load_era5_truth(era5_lr_dir, T, scale_dir, device):
    """加载 ``T`` 时刻 ERA5 真值（已归一化到训练空间），shape ``(1, 69, 181, 360)``。

    路径约定（参见 ``inference/era5_lr_forecast.py:13-28``）：
    ``{era5_lr_dir}/{YYYY}/{YYYY-MM-DD}/{HH:MM:SS}.npy``。
    反归一化 → 再归一化到训练空间，等价于 ``(raw - mean) / std``。
    """
    era5_path = os.path.join(
        era5_lr_dir,
        f"{T.year:04d}",
        f"{T.year:04d}-{T.month:02d}-{T.day:02d}",
        f"{T.hour:02d}:{T.minute:02d}:{T.second:02d}.npy",
    )
    raw = np.load(era5_path).reshape(1, -1, H, W)         # 物理单位 (1, 69, 181, 360)
    # Bug D 修复：npz 是 per-variable dict schema；用 get_normalize 与 save_nc/load_xb_initial 一致
    mean, std = get_normalize(scale_dir, VARIABLES)
    normed = ((raw - mean) / std).astype(np.float32)
    return torch.from_numpy(normed).to(device)


# ---------------------------------------------------------------------------
# D1 修复（2026-08-10）：climatology 加载 + 缓存（避免每周期 69 文件 I/O）
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _climatology_cached(clim_dir: str, month: int, day: int):
    """按 (clim_dir, MM-DD) 缓存每日 climatology（物理单位 (1, V, H, W)）。

    性能说明：get_climatology 每次调用读 len(VARIABLES) 个 npy 文件（无内部缓存）。
    全年 DA 每周期直接调用 = 730 × 69 ≈ 5 万次文件 I/O。这里按天缓存
    —— 每个 DCU 进程对当日 climatology 仅一次 69 文件 I/O。多进程 DDP 各进程
    各自一份缓存（每进程 ≤18 MB：69×4×181×360 ×366 天 ≈ 18 MB，可接受）。

    只服务 DA 路径；forecast 路径（era5_lr_forecast / era5_interp_forecast）仍直接调
    get_climatology 以保留各自缓存语义。
    """
    # get_climatology 只用 month/day；构造虚拟 datetime 供调用
    t_dummy = datetime(2000, month, day)
    return get_climatology(clim_dir, (-1, H, W), t_dummy, VARIABLES)


def _load_climatology(cfg, T, state=None, key="clim_cache"):
    """step_fn 内加载 T 对应日期 climatology（物理单位 (1, 69, 181, 360)）。

    双层缓存：
      1) 模块级 ``_climatology_cached``（lru_cache，clim_dir+MM-DD）—— 一进程一天一次 I/O；
      2) ``state[key]`` dict（det/ens/daw 路径）—— 跨周期 O(1) dict 取。

    timelagged 路径 state 是 deque（装 lagged analysis），不是 dict —— 调用方传
    ``state=None`` 仅依赖 lru_cache（已足够，timelagged week 7 天 = 7 次 I/O）。

    Args:
        cfg: 推理配置；需 ``cfg["era5_lr_dir"]``（clim_dir 由此派生）。
        T: 当前周期 datetime。
        state: 可选；为 dict 时按 mmdd 做第二层缓存（state 必须支持 ``state.setdefault``）。
        key: state 中缓存子键名。

    Returns:
        np.ndarray ``(1, 69, 181, 360)`` float32，物理单位（K / m/s / m²/s² / Pa）。
    """
    clim_dir = os.path.join(cfg["era5_lr_dir"], "climatology_np181x360_2010_2021")
    mmdd = f"{T.month:02d}-{T.day:02d}"
    if state is not None:
        cache = state.setdefault(key, {})
        if mmdd in cache:
            return cache[mmdd]
    clim = _climatology_cached(clim_dir, T.month, T.day)
    if state is not None:
        cache[mmdd] = clim
    return clim


def compute_metrics(xa_norm, truth_norm, scale_dir, var_subset=None, climatology=None):
    """逐通道计算纬度加权 RMSE / ACC / Bias（**物理单位空间**）。

    关键修复：之前在归一化空间（z-score）算指标没气象学意义 —— z-500（~5000 m²/s²）
    和 t2m（~280 K）都被拉到 ~N(0, 1)，丢失了"哪类变量偏大"的物理语义。
    反归一化到物理单位后才可比、与文献对齐（z-500 RMSE ~几十 m²/s²、t2m RMSE ~2 K）。

    默认评全部 69 个变量（与 forecast10d 同口径；eval_forecast 单变量分析时传
    var_subset=['z-500'] 等子集）。

    ACC 距平（D1 修复 2026-08-10）：
      - ``climatology is not None``：用真 climatology（物理单位 (1, V, H, W)，路径
        {era5_lr_dir}/climatology_np181x360_2010_2021/<MM-DD>/<var>.npy，daily mean
        2010-2021）减距平，与 forecast 评估 ``weighted_acc(era5 - clim,
        forecast_eval - clim)``（inference/era5_forecast_core.py:251）口径一致。
      - ``climatology is None``（旧行为兼容）：用场空间均值近似
        ``pred - pred.mean(axis=(-1, -2))``（archive 快照 + 消融 A/B 不变）。

    Args:
        xa_norm: 分析场，``(1, 69, 181, 360)`` torch.Tensor（归一化空间）。
        truth_norm: ERA5 真值，同 shape（归一化空间）。
        scale_dir: 标准化 npz 目录（含 per-variable mean/std）。
        var_subset: 要评估的变量名子集（如 ['z-500']）；None 表示全部 69 个变量。
        climatology: 物理单位 climatology，shape ``(1, V, H, W)`` 或 ``(1, V, 1, 1)``。
            None 时回退场均值近似（向后兼容）。

    Returns:
        ``{"rmse": {var: float}, "acc": {...}, "bias": {...}, "summary": {...}}``
        summary 含 var_subset 字段，方便审计。
    """
    if var_subset is None:
        var_subset = list(VARIABLES)
    var_subset_set = set(var_subset)

    # 统一 CPU：确定性 + 集合评估指标统一在 CPU 上算（per-variable 标量，tensor 体量小）。
    # load_era5_truth 默认 device=cuda，xa12 在 chunked ensemble 路径末尾 .cpu()，
    # 显式 .cpu() 避免 device 不一致隐患（与 compute_ensemble_metrics 一致）。
    xa_norm = xa_norm.cpu() if xa_norm.device.type != "cpu" else xa_norm
    truth_norm = truth_norm.cpu() if truth_norm.device.type != "cpu" else truth_norm

    mean, std = get_normalize(scale_dir, VARIABLES)   # (1, 69, 1, 1)
    # 反归一化到物理单位：x_raw = x_norm * std + mean
    pred = (xa_norm.detach().numpy() * std + mean).astype(np.float32)
    tgt = (truth_norm.detach().numpy() * std + mean).astype(np.float32)

    # WRMSE：物理单位空间纬度加权（输入 (N, C, H, W)）
    rmse = weighted_rmse_channels(pred, tgt).squeeze()      # (C,)
    # ACC：D1 修复 — 真 climatology 距平 vs 旧场均值近似
    if climatology is not None:
        # climatology 物理单位 (1, V, H, W) 或 (1, V, 1, 1)；与 pred/tgt (1, V, H, W) 广播减
        clim = np.asarray(climatology, dtype=np.float32)
        if clim.ndim == 4 and clim.shape[1] != pred.shape[1]:
            raise ValueError(
                f"climatology 变量数 {clim.shape[1]} 与 pred 变量数 {pred.shape[1]} "
                f"不一致；请确认 climatology 是按 VARIABLES 顺序加载"
            )
        pred_anom = pred - clim
        tgt_anom = tgt - clim
    else:
        # 向后兼容：旧"场均值近似"，archive 快照与消融 A/B 行为不变
        pred_anom = pred - pred.mean(axis=(-1, -2), keepdims=True)
        tgt_anom = tgt - tgt.mean(axis=(-1, -2), keepdims=True)
    acc = weighted_acc_channels(pred_anom, tgt_anom).squeeze()    # (C,)

    # Bias：转 torch 用 torch 版（无 NumPy 版）
    bias = weighted_bias_torch_channels(
        torch.from_numpy(pred), torch.from_numpy(tgt)
    ).squeeze().numpy()                                          # (C,)

    metrics = {"rmse": {}, "acc": {}, "bias": {}}
    for i, var in enumerate(VARIABLES):
        if var in var_subset_set:
            metrics["rmse"][var] = float(rmse[i])
            metrics["acc"][var] = float(acc[i])
            metrics["bias"][var] = float(bias[i])

    # summary 仅对 var_subset 求平均（空列表防御）
    n = max(len(metrics["rmse"]), 1)
    metrics["summary"] = {
        "rmse_mean": float(np.mean(list(metrics["rmse"].values()))),
        "acc_mean":  float(np.mean(list(metrics["acc"].values()))),
        "bias_mean": float(np.mean(list(metrics["bias"].values()))),
        "var_subset": list(var_subset),
    }
    return metrics


def save_metrics(metrics, T, kind, output_dir, timing=None, meta=None):
    """将每周期评估指标写为 JSON + 日志逐变量打印 RMSE / Bias（4 列紧凑格式）。

    日志输出：
      1) summary 行（rmse_mean / acc_mean / bias_mean）
      2) Per-var header（变量数）
      3) 逐变量紧凑打印（每 4 个变量一行；物理单位 + 符号化 bias）

    Args:
        metrics: 由 compute_metrics 返回的 dict，含 rmse/acc/bias/summary
        T: 当前周期时间
        kind: 'analysis_12h' / 'ic_6h'
        output_dir: 输出根目录
        timing: 可选 dict，含 t_forecast_12h_s / t_da_12h_s / t_da_6h_s（秒）
        meta: 可选 dict，原样写入 JSON 的 ``"meta"`` 子键（如 P1 的
            ``{"n_active_obs": int, "da_skipped": bool, "active_obs": [...]}``，
            用于标记“观测缺失→分析=背景”的周期；缺省不写该键，向后兼容）。
    """
    out_dir = os.path.join(output_dir, kind + "_metrics")
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{T.strftime('%Y%m%dT%H%M%S')}.json")
    payload = {
        "time": T.isoformat(),
        "kind": kind,
        "rmse": metrics["rmse"],
        "acc": metrics["acc"],
        "bias": metrics["bias"],
        # 复用 compute_metrics 已算好的 summary（含 var_subset 信息）
        "summary": metrics.get("summary", {
            "rmse_mean": float(np.mean(list(metrics["rmse"].values()))),
            "acc_mean":  float(np.mean(list(metrics["acc"].values()))),
            "bias_mean": float(np.mean(list(metrics["bias"].values()))),
        }),
        # === 集合同化专属字段（仅 N>=3 时存在；由 assim_cycle_ensemble_xichen 写入）===
        "ensemble_metrics": metrics.get("ensemble_metrics", {}),
        # === 周期元信息（2026-08-26 新增，P1）：观测活性/同化是否跳过 ===
        # meta 为空时省略该键，保持旧 JSON schema 的向后兼容。
        "meta": meta or {},
        # === 计时（2026-08-08 新增）：每周期各阶段耗时 ===
        "timing": timing or {},
    }
    with open(fname, "w") as f:
        json.dump(payload, f, indent=2)

    summary = payload["summary"]
    log.info("Metrics saved %s | summary rmse=%.3f acc=%.3f bias=%.3f",
             fname, summary["rmse_mean"], summary["acc_mean"], summary["bias_mean"])

    # ---- 逐变量打印 RMSE / Bias（4 列紧凑；快速审计每个变量的偏差与系统偏差）----
    items = list(metrics["rmse"].keys())
    log.info("Per-var metrics [%s] T=%s (%d vars, 4 per line):", kind, T.isoformat(), len(items))
    cols = 4
    for i in range(0, len(items), cols):
        chunk = items[i:i + cols]
        parts = [f"{v:8s} rmse={metrics['rmse'][v]:9.3f} bias={metrics['bias'][v]:+9.3f}"
                 for v in chunk]
        log.info("  %s", " | ".join(parts))


# ---------------------------------------------------------------------------
# main() 公共骨架（det / ens 两个入口脚本共用）
# ---------------------------------------------------------------------------
def build_common_parser():
    """创建基础 parser 并添加共享参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dacycle_oneyear.json")
    parser.add_argument("--cascade_config",
                        default="configs/cascade_da_6obs.json")
    parser.add_argument("--ablation", default=None,
                        help="可选：消融 JSON（覆盖 obs_order + output_dir_suffix）")

    # === 输出根目录（覆盖 cfg["output_dir"]）===
    parser.add_argument("--output_root", type=str,
                        default=None,
                        help="所有分析场/指标的输出根目录；task_name 作为子目录。"
                             "不指定时使用 cfg['output_dir']。")
    parser.add_argument("--task_name_out", type=str, default=None,
                        help="输出子目录名；默认用 cfg['task_name']，没有则用 'default'。")
    return parser


def _expand_env_vars(obj):
    """递归对配置树中的字符串值做 ``os.path.expandvars``（支持 ``${VAR}`` 占位）。"""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


def load_config_and_env(args):
    """加载配置 + 模型，返回 (cfg, cascade_cfg, components, obs_dict, device)。

    复刻拆分前 dacycle.main() 的步骤（json 加载 → ablation 覆盖 → output_dir 三级
    优先级 → path fail-fast → device → obs_dict → load_models）。
    """
    cfg = _expand_env_vars(json.load(open(args.config)))
    cascade_cfg = json.load(open(args.cascade_config))

    # 消融覆盖
    if args.ablation:
        abl = json.load(open(args.ablation))
        cfg.update({k: v for k, v in abl.items() if not k.startswith("_")})

    # output_dir 优先级（从高到低）：
    #   1) --output_root + --task_name_out（CLI 显式指定）
    #   2) cfg["output_dir"] + ablation suffix（消融 JSON 提供）
    #   3) cfg["output_dir"] 原值（JSON 配置默认）
    if args.output_root is not None:
        task_name_out = args.task_name_out or cfg.get("task_name", "default")
        cfg["output_dir"] = os.path.join(args.output_root, task_name_out)
        log.info("output_dir overridden by CLI: %s", cfg["output_dir"])
    else:
        # 消融后缀（向后兼容）
        suffix = cfg.pop("output_dir_suffix", None)
        if suffix:
            cfg["output_dir"] = os.path.join(cfg["output_dir"], suffix)
    # 确保输出目录存在（save_nc/save_metrics 内部也建子目录，但根目录要预建）
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # 启动期 path 检查（fail fast）
    must_exist = [
        cfg["ckpt_forecast"],
        cfg["ckpt_cascade_da"],
        cfg["obs_dir"],
        cfg["era5_lr_dir"],
        cfg["scale_dir"],
        cfg["forecast_config"],
        cfg["cascade_config"] if "cascade_config" in cfg else args.cascade_config,
        # HIGH#2 修复：climatology 目录也加入启动期校验（之前漏检，导致首周期
        # 缺文件 → 全年 730 次静默 FileNotFound → 零输出）。
        os.path.join(cfg["era5_lr_dir"], "climatology_np181x360_2010_2021"),
    ]
    must_exist += [cfg["ckpt_obsop"][sat] for sat in SAT_LIST]
    must_exist += [cfg["obsop_configs"][sat] for sat in SAT_LIST]
    for p in must_exist:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path missing: {p}")

    device = torch.device(cfg.get("device", "cuda"))

    # 加载 Solver 所需的 obs_dict（microwave + conventional schemas）
    # 直接复用 obs_dict_6obs.json，避免重复维护 prepare_* 缓存
    obs_dict_cfg_path = cfg.get("obs_dict_config",
                                "configs/obs_dict_6obs.json")
    obs_dict = json.load(open(obs_dict_cfg_path))["obs_dict"]
    log.info("obs_dict loaded from %s", obs_dict_cfg_path)

    # 模型加载（依赖 obs_dict 来构造 microwave prepare 缓存）
    components = load_models(cfg, cascade_cfg, obs_dict, device)
    log.info("Models loaded: forecast + 4 obsop + 6 DA + Solver + microwave_prep + conventional_prep")
    return cfg, cascade_cfg, components, obs_dict, device


def run_dacycle_loop(cfg, components, obs_dict, device, step_fn, initial_state=None):
    """主循环骨架（复刻拆分前 dacycle.main() 的 while 循环 + 异常处理）。

    step_fn(cfg, T, prev_xa, components, obs_dict, device, state)
        -> (new_prev_xa, timing_or_None, new_state)
    timing 非 None 时累计计时（确定性路径）；None 不累计（集合路径）。
    state 为跨周期状态 dict（集合路径存 prev_xa_plus/minus 列表）。
    ``initial_state``（2026-08-10 新增）：可选外部初始 state，传入路径与
    step_fn 返回的 ``new_state`` 形成闭环；不传时退化为 ``{}``（保持
    dacycle_det / dacycle_ens 调用语义不变，向后兼容）。

    返回累计 timing dict（确定性摘要打印用；集合路径忽略）：
      n_cycles / n_6h_cycles / total_t_12h_cycle / total_t_forecast_12h /
      total_t_da_12h / total_t_da_6h
    """
    T = datetime.fromisoformat(cfg["cycle_start"])
    T_end = datetime.fromisoformat(cfg["cycle_end"])
    interval = cfg["cycle_interval_hours"]
    prev_xa = None
    state = initial_state if initial_state is not None else {}

    # === 计时累计（2026-08-08 新增）：用于打印 6h DA / 12h DA 平均 cycle 时间 ===
    n_cycles = 0
    n_6h_cycles = 0
    n_obs_missing_cycles = 0
    total_t_12h_cycle = 0.0
    total_t_forecast_12h = 0.0
    total_t_da_12h = 0.0
    total_t_da_6h = 0.0

    while T <= T_end:
        t0 = time.time()
        log.info("==== T=%s ====", T.isoformat())
        try:
            prev_xa, cycle_timing, state = step_fn(
                cfg, T, prev_xa, components, obs_dict, device, state,
            )
            if cycle_timing is not None:
                # 累计均值（不依赖 metrics JSON，直接统计）
                n_cycles += 1
                total_t_12h_cycle += time.time() - t0
                total_t_forecast_12h += cycle_timing["t_forecast_12h_s"]
                total_t_da_12h += cycle_timing["t_da_12h_s"]
                if cycle_timing["t_da_6h_s"] > 0:
                    n_6h_cycles += 1
                    total_t_da_6h += cycle_timing["t_da_6h_s"]
        except FileNotFoundError as e:
            # HIGH#3 修复：首周期 FileNotFound 不可恢复（climatology / 模型 checkpoint 缺失
            # 时 prev_xa 仍为 None，下一周期再次走 climatology 分支，全年反复抛错零输出）。
            # 必须显式 raise 中断，避免静默失败。
            if prev_xa is None:
                log.error(
                    "Cycle T=%s | First-cycle FileNotFound, aborting (HIGH#3): %s",
                    T.isoformat(), e,
                )
                raise
            # 非首周期 FileNotFound：保留 prev_xa 推进（P1：累计观测缺失周期数，
            # 供摘要打印，警示全年“分析=背景”的静默退化；不中断主循环以保持兼容）
            n_obs_missing_cycles += 1
            log.warning("Cycle T=%s | FileNotFound (use prev_xa): %s", T.isoformat(), e)
        except Exception as e:
            # 真实 bug：立即抛错中断，避免静默失败（修复 BUG #6）
            log.error("Cycle T=%s | FATAL: %s", T.isoformat(), e)
            raise
        log.info("T=%s done in %.1fs", T.isoformat(), time.time() - t0)
        T += relativedelta(hours=interval)

    return {
        "n_cycles": n_cycles,
        "n_6h_cycles": n_6h_cycles,
        "n_obs_missing_cycles": n_obs_missing_cycles,
        "total_t_12h_cycle": total_t_12h_cycle,
        "total_t_forecast_12h": total_t_forecast_12h,
        "total_t_da_12h": total_t_da_12h,
        "total_t_da_6h": total_t_da_6h,
    }


# ---------------------------------------------------------------------------
# 全年运行结束：打印 12h / 6h cycle 平均时间
# ---------------------------------------------------------------------------
# 从 dacycle_det.py:197-229 移入（2026-08-13；multimodal/ensemble 共享）。
def print_timing_summary(cfg, timing):
    """打印整年 cycle timing 总览（12h / 6h / forecast / DA）。

    Args:
        cfg: dacycle 配置；用 ``cfg.get('trigger_6h_hours')`` 校准 6h 触发逻辑。
        timing: ``run_dacycle_loop`` 返回 dict，含
          ``n_cycles / n_6h_cycles / total_t_12h_cycle / total_t_forecast_12h /
          total_t_da_12h / total_t_da_6h``。
    """
    n_cycles = timing["n_cycles"]
    n_6h_cycles = timing["n_6h_cycles"]
    n_obs_missing_cycles = timing.get("n_obs_missing_cycles", 0)
    total_t_12h_cycle = timing["total_t_12h_cycle"]
    total_t_forecast_12h = timing["total_t_forecast_12h"]
    total_t_da_12h = timing["total_t_da_12h"]
    total_t_da_6h = timing["total_t_da_6h"]
    log.info("=" * 60)
    log.info("CYCLE TIMING SUMMARY (deterministic DA)")
    if n_obs_missing_cycles > 0:
        log.warning(
            "OBS-MISSING: %d cycles hit FileNotFound mid-run (analysis kept prev_xa "
            "forecast). If this is large, check obs_dir data availability — those "
            "cycles' metrics reflect background, not an assimilated analysis (P1).",
            n_obs_missing_cycles,
        )
    if n_cycles > 0:
        log.info(
            "12h cycle total : n=%d  mean=%.2fs/cycle  total=%.1fs",
            n_cycles, total_t_12h_cycle / n_cycles, total_t_12h_cycle,
        )
        log.info(
            "  forecast 12h  : mean=%.2fs  (%.1f%% of cycle)",
            total_t_forecast_12h / n_cycles,
            100.0 * total_t_forecast_12h / max(total_t_12h_cycle, 1e-9),
        )
        log.info(
            "  DA 12h        : mean=%.2fs  (%.1f%% of cycle)",
            total_t_da_12h / n_cycles,
            100.0 * total_t_da_12h / max(total_t_12h_cycle, 1e-9),
        )
    if n_6h_cycles > 0:
        log.info(
            "6h DA cycle     : n=%d  mean=%.2fs/cycle  total=%.1fs",
            n_6h_cycles, total_t_da_6h / n_6h_cycles, total_t_da_6h,
        )
    else:
        log.warning("6h DA cycle: no cycles triggered (check trigger_6h_hours=%s)",
                    cfg.get("trigger_6h_hours"))
    log.info("=" * 60)
