"""NWP-Benchmark 兼容的 XiChen 分析场/预报场 NetCDF 读写器。

对齐 https://github.com/lixruize-del/NWP-Benchmark 的 ``src/common/saver.py``
(``Saver.save()``) 保存约定，使 XiChen 写出的场文件在结构上与基准一致，
可被其下游评估工具（station / TC / heatwave / metrics-backfill）消费。

目标格式（每文件一个 NetCDF，9 个变量）：
  - 地面（4）：t2m, u10, v10, msl      dims ["time","latitude","longitude"]
    （顺序与 inference/configs/xichen_forecast.json 的 default_vars / data_utils.VARIABLES 一致）
  - 气压层（5）：z, u, v, t, q          dims ["time","plev_<short>","latitude","longitude"]
    plev_* = [50,100,150,200,250,300,400,500,600,700,850,925,1000] hPa（升序），
    plev 坐标 attrs: units=hPa, long_name=isobaric level；气压变量 attr positive:down
  - 坐标：time=[valid_time]（起报+lead，datetime64[ns]）、
          latitude=linspace(90,-90,H)、longitude=linspace(0,360,W,endpoint=False)
  - 变量 attrs：units / long_name / GRIB_shortName
  - 数据集 attrs：initial_time(iso)、forecast_lead_time("N hours")、
    generator="NWPBench Saver"，外加 XiChen 附加（kind/obs_order/scale_dir）与 grid/resolution
    （刻意省略 NWP 的 creation_date：now() 会让同 seed 两次跑的文件 attrs 块不再字节级
    一致，破坏 CLAUDE.md 的复现承诺；下游消费者不读该属性）
  - 文件：{root}/{init:%Y%m%d%H}/{init:%Y%m%dT%H}-{lead:02d}.nc（lead=0 即分析场；
    文件名含完整起报时刻 %Y%m%dT%H，如 20230101T00-00.nc，避免同日多次起报无法区分）

编码：每变量 zlib complevel=4（延续 XiChen 现状；读者对编码无感，属刻意的无害偏离）。
原子写：tempfile.mkstemp 于目标子目录 → to_netcdf(engine="netcdf4") → os.replace。

集合 member 后缀 ``_{member}`` 仅预留不用：XiChen 写集合平均分析场而非逐成员文件。

0.25° 输出（resolution="0p25"）：对 1.0° 物理场做 geographic_interpolate("lr2hr")
线性插值到 721×1440（分变量块插值以控内存；输入已是 721×1440 时跳过）。
注意：该插值线性且非守恒，0.25° 产物的指标与 1.0° 产物不可直接比较。
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from xichen.data import VARIABLES, geographic_interpolate

log = logging.getLogger("inference.nwp_saver")

# --- GRIB short-name 元数据（与 NWP-Benchmark src/common/saver.py 一致）---
# 顺序与 inference/configs/xichen_forecast.json default_vars / data_utils.VARIABLES 一致
# （surface: t2m, u10, v10, msl）。通道映射按名字不按顺序，此顺序只决定输出文件变量排列。
SURFACE_SHORTS = ["t2m", "u10", "v10", "msl"]
PRESSURE_SHORTS = ["z", "u", "v", "t", "q"]
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

CHANNEL_META = {
    "z":   {"Name": "Geopotential",              "Unit": "m^2 s^-2"},
    "t":   {"Name": "Temperature",               "Unit": "K"},
    "u":   {"Name": "U component of wind",       "Unit": "m s^-1"},
    "v":   {"Name": "V component of wind",       "Unit": "m s^-1"},
    "q":   {"Name": "Specific humidity",         "Unit": "kg kg^-1"},
    "t2m": {"Name": "2 metre temperature",       "Unit": "K"},
    "msl": {"Name": "Mean sea level pressure",   "Unit": "Pa"},
    "u10": {"Name": "10 metre U wind component", "Unit": "m s^-1"},
    "v10": {"Name": "10 metre V wind component", "Unit": "m s^-1"},
}


def channel_map(variables: list = VARIABLES) -> list:
    """XiChen 通道名 → (GRIB short, 层或 None) 的映射（与 variables 顺序对齐）。"""
    out = []
    for var in variables:
        if var in SURFACE_SHORTS:
            out.append((var, None))
        else:
            base, lvl = var.rsplit("-", 1)
            out.append((base, int(lvl)))
    return out


def _split(field: np.ndarray, variables: list = VARIABLES):
    """把 (V,H,W) 物理场按 GRIB short name 分组。

    返回 (surface, pressure)：
      surface: {short: (H,W) 数组}
      pressure: {short: [(level, (H,W) 数组), ...]}，层序按 level 数值升序
    （防御性，不依赖 VARIABLES 的层序）。
    """
    surface, pressure = {}, {}
    for (short, lev), arr in zip(channel_map(variables), field):
        if lev is None:
            surface[short] = arr
        else:
            pressure.setdefault(short, []).append((lev, arr))
    for short in pressure:
        pressure[short] = sorted(pressure[short], key=lambda x: x[0])
    return surface, pressure


def _regrid_lr2hr_block(block: np.ndarray) -> np.ndarray:
    """单个 (L,181,360) 块插值到 (L,721,1440)（按块调用以控内存）。"""
    return geographic_interpolate(np.asarray(block, dtype=np.float32), interp_direction="lr2hr")


def field_path(root: str, init_time: datetime, lead_hours: int, member=None) -> str:
    """NWP 文件路径约定：{root}/{init:%Y%m%d%H}/{init:%Y%m%dT%H}-{lead:02d}.nc。

    子目录含小时（%Y%m%d%H）以区分同日 00/12 UTC 等多次起报；文件名也含完整起报时刻
    （%Y%m%dT%H，如 20230101T00-00.nc），避免同名日期无法区分时刻。lead=0 即分析场。
    """
    sub = init_time.strftime("%Y%m%d%H")
    date = init_time.strftime("%Y%m%dT%H")
    name = f"{date}-{int(lead_hours):02d}.nc"
    if member is not None:
        name = f"{date}-{int(lead_hours):02d}_{member}.nc"
    return os.path.join(root, sub, name)


def analysis_ic_path(root: str, t: datetime) -> str:
    """分析/IC 文件路径（lead=0）。"""
    return field_path(root, t, 0)


def _to_netcdf_fallback(ds: xr.Dataset, path: str, encoding: dict) -> None:
    """写 NC，引擎回退链：netcdf4 → h5netcdf（若已装）→ scipy。

    某些文件系统对 netCDF4/HDF5 写入有限制（即使属主也 PermissionError，而 touch 正常——
    已在 NPU 集群实测：netcdf4 引擎 PermissionError、scipy 引擎 OK）。scipy 产出
    netCDF3-classic（无压缩、~2GB 上限），xarray 与下游 NWP 工具仍可读。
    """
    engines = ["netcdf4"]
    try:
        import h5netcdf  # noqa: F401
        engines.append("h5netcdf")
    except ImportError:
        pass
    engines.append("scipy")
    last_err = None
    for eng in engines:
        try:
            if eng == "scipy":
                ds.to_netcdf(path, engine=eng)
            else:
                ds.to_netcdf(path, engine=eng, encoding=encoding)
            log.info("to_netcdf(engine=%s) -> %s", eng, path)
            return
        except Exception as e:
            last_err = e
            log.warning("to_netcdf(engine=%s) failed: %s; try next engine", eng, e)
    raise last_err


def save_field(
    field_phys,
    init_time: datetime,
    lead_hours: int,
    root: str,
    resolution: str = "1p0",
    member=None,
    extra_attrs: dict | None = None,
    path_time: datetime | None = None,
) -> str:
    """写出单个 (init_time, lead_hours) 的场（分析或预报）到 NWP-Benchmark 格式。

    Args:
        field_phys: 物理单位场，(1,V,H,W) 或 (V,H,W)（fp32 或可转 numpy）；V=69。
        init_time: 起报/分析时刻 datetime。
        lead_hours: 预报时效（小时）；分析场传 0。
        root: 输出根目录（如 analysis_12h / ic_6h / forecast），其下自动建 {init:%Y%m%d%H}/ 子目录。
        resolution: "1p0"（默认，原生）或 "0p25"（LR 场插值到 721×1440；输入已是 HR 则跳过）。
        member: 集合成员号（预留；XiChen 写集合平均分析场，默认 None 不带后缀）。
        extra_attrs: 附加数据集属性（如 kind/obs_order/scale_dir）。
        path_time: 可选；若给定时**路径用该时刻**（目录/文件名按 ``{path_time}`` 组织），但文件内
            ``time`` 坐标仍为 ``init_time + lead_hours``、attrs 仍记录 ``init_time``/``lead_hours``。
            用于"预报的验证时刻（目标时刻）作为目录名"的场景（如 TC 预报网格：文件
            ``{root}/{t:%Y%m%d%H}/{t:%Y%m%dT%H}-{lead:02d}.nc``，其中 init_time=t-lead、
            path_time=t），保持旧调用方（path_time=None）行为完全不变。

    Returns:
        写出的 NC 文件路径。
    """
    field = np.asarray(field_phys, dtype=np.float32)
    while field.ndim > 3 and field.shape[0] == 1:
        field = field[0]                      # (1,1,V,H,W) 或 (1,V,H,W) → (V,H,W)
    V, H, W = field.shape
    if V != len(VARIABLES):
        raise ValueError(f"channel dim {V} != {len(VARIABLES)}")

    surface, pressure = _split(field)
    if resolution == "0p25" and H == 181:
        H, W = 721, 1440
        surface = {s: _regrid_lr2hr_block(a[None])[0] for s, a in surface.items()}
        pressure = {
            s: list(zip(
                [lv for lv, _ in grp],
                _regrid_lr2hr_block(np.stack([a for _, a in grp], axis=0)),
            ))
            for s, grp in pressure.items()
        }
    # resolution=="0p25" 且 H==721：输入已是 HR，跳过插值。

    valid_time = init_time + timedelta(hours=int(lead_hours))
    lat = np.linspace(90.0, -90.0, H)
    lon = np.linspace(0.0, 360.0, W, endpoint=False)
    time_coord = [np.datetime64(valid_time, "ns")]

    ds_list = []
    for short in SURFACE_SHORTS:
        if short not in surface:
            continue
        meta = CHANNEL_META.get(short, {"Name": short, "Unit": "unknown"})
        da = xr.DataArray(
            data=surface[short][None, :, :],
            dims=["time", "latitude", "longitude"],
            coords={"time": time_coord, "latitude": lat, "longitude": lon},
            attrs={
                "units": meta["Unit"],
                "long_name": meta["Name"],
                "GRIB_shortName": short,
            },
        )
        ds_list.append(da.to_dataset(name=short))

    for short in PRESSURE_SHORTS:
        if short not in pressure:
            continue
        grp = pressure[short]
        levels = np.array([lv for lv, _ in grp], dtype=np.float64)
        stack = np.stack([a for _, a in grp], axis=0)[None, ...]  # (1, L, H, W)
        plev_dim = f"plev_{short}"
        meta = CHANNEL_META.get(short, {"Name": short, "Unit": "unknown"})
        da = xr.DataArray(
            data=stack,
            dims=["time", plev_dim, "latitude", "longitude"],
            coords={
                "time": time_coord,
                plev_dim: xr.DataArray(
                    levels,
                    dims=[plev_dim],
                    attrs={"units": "hPa", "long_name": "isobaric level"},
                ),
                "latitude": lat,
                "longitude": lon,
            },
            attrs={
                "units": meta["Unit"],
                "long_name": meta["Name"],
                "GRIB_shortName": short,
                "positive": "down",
            },
        )
        ds_list.append(da.to_dataset(name=short))

    if not ds_list:
        raise ValueError("no variables to save")

    final_ds = xr.merge(ds_list)
    final_ds.attrs["initial_time"] = init_time.isoformat()
    final_ds.attrs["forecast_lead_time"] = f"{int(lead_hours)} hours"
    # 注意：刻意不写 creation_date=now()——它会让同 seed 两次跑产生的 analysis_12h/*.nc
    # 在 attrs 块上不再字节级一致（CLAUDE.md 承诺的复现不变量）。下游消费者不读该属性。
    final_ds.attrs["generator"] = "NWPBench Saver"
    final_ds.attrs["grid"] = "0p25" if (H, W) == (721, 1440) else "1p0"
    final_ds.attrs["resolution"] = resolution
    if extra_attrs:
        final_ds.attrs.update(extra_attrs)

    # path_time 非 None 时路径按该时刻组织（文件内 time/attrs 仍用 init_time+lead）
    _path_time = path_time if path_time is not None else init_time
    out_path = field_path(root, _path_time, lead_hours, member=member)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    # 启动期写权限预检：很多 NPU/NFS/Lustre 集群对 ``touch`` 通过但对
    # netCDF4/scipy 的临时文件 + rename 模式 PermissionError；先实测一次写
    # + unlink，让 PermissionError 提前抛在清晰位置（而不是钻进 xarray 内部）。
    try:
        probe = os.path.join(out_dir, ".xichen_write_probe")
        with open(probe, "w") as f:
            f.write("probe")
        os.unlink(probe)
    except OSError as e:
        raise OSError(
            f"save_field: output_dir {out_dir!r} not writable "
            f"({type(e).__name__}: {e}). If on NPU/NFS cluster, check "
            f"`ls -ld {out_dir}` and try a different OUTPUT_ROOT (e.g. "
            f"/tmp/$USER/xichen_runs or /work2/$USER/xichen_runs)."
        ) from e
    encoding = {name: {"zlib": True, "complevel": 4} for name in final_ds.data_vars}
    fd, tmp_path = tempfile.mkstemp(suffix=".nc", dir=out_dir)
    os.close(fd)
    # mkstemp 默认 0600；先还原为 0644 再让 netCDF4 写（部分文件系统对 0600 临时文件的
    # netCDF4 写入有限制）。chmod 本身受限则容忍，继续按 0600 写。
    try:
        os.chmod(tmp_path, 0o644)
    except OSError as e:
        log.warning("chmod %s -> 0644 failed (%s); continue with current mode", tmp_path, e)
    try:
        _to_netcdf_fallback(final_ds, tmp_path, encoding)
        os.replace(tmp_path, out_path)
    except OSError as e:
        # 某些集群文件系统对"临时文件 + os.replace"原子写有限制（写临时文件或 replace
        # 均可能 PermissionError）。回退直写最终路径，仍失败则原样抛出。
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        log.warning("atomic write to %s failed (%s); fall back to direct write", out_path, e)
        _to_netcdf_fallback(final_ds, out_path, encoding)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    log.info("Saved %s", out_path)
    return out_path


def load_field(path: str, variables: list = VARIABLES) -> np.ndarray:
    """从 NWP-Benchmark 格式 NetCDF 精确重建 (V,H,W) 物理场（fp32，VARIABLES 顺序）。

    地面读 ``ds[short]``；气压读 ``ds[short].isel({plev_dim: 层索引})``。层索引按
    坐标值（isclose）定位而非位置，对层序不敏感；time 维取 ``isel(time=0)``。
    """
    cm = channel_map(variables)
    with xr.open_dataset(path) as ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
        H, W = lat.size, lon.size
        out = np.zeros((len(variables), H, W), dtype=np.float32)
        for i, var in enumerate(variables):
            short, lev = cm[i]
            da = ds[short]
            if "time" in da.dims:
                da = da.isel(time=0)
            if lev is None:
                out[i] = da.values
            else:
                plev_dim = next(
                    d for d in da.dims if d == "isobaricInhPa" or d.startswith("plev_")
                )
                levels = np.asarray(da.coords[plev_dim].values, dtype=np.float64)
                idx = int(np.flatnonzero(np.isclose(levels, float(lev)))[0])
                out[i] = da.isel({plev_dim: idx}).values
        return out.astype(np.float32)
