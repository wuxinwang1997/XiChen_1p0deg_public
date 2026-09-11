import numpy as np
import time
import os
import glob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import torch
import json
import sys
from scipy.interpolate import RegularGridInterpolator
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(name)s - %(levelname)s - %(message)s')

VARIABLES = [
  "t2m", "u10", "v10", "msl",
  "z-50", "z-100", "z-150", "z-200", "z-250", "z-300", "z-400", "z-500", "z-600", "z-700", "z-850", "z-925", "z-1000",
  "u-50", "u-100", "u-150", "u-200", "u-250", "u-300", "u-400", "u-500", "u-600", "u-700", "u-850", "u-925", "u-1000",
  "v-50", "v-100", "v-150", "v-200", "v-250", "v-300", "v-400", "v-500", "v-600", "v-700", "v-850", "v-925", "v-1000",
  "t-50", "t-100", "t-150", "t-200", "t-250", "t-300", "t-400", "t-500", "t-600", "t-700", "t-850", "t-925", "t-1000",
  "q-50", "q-100", "q-150", "q-200", "q-250", "q-300", "q-400", "q-500", "q-600", "q-700", "q-850", "q-925", "q-1000",
]

conv_vars = {
    "prepbufr": [
        "t2m", "u10", "v10", "msl", #"tp",
        "z-50", "z-100", "z-150", "z-200", "z-250", "z-300", "z-400", "z-500", "z-600", "z-700", "z-850", "z-925", "z-1000",
        "u-50", "u-100", "u-150", "u-200", "u-250", "u-300", "u-400", "u-500", "u-600", "u-700", "u-850", "u-925", "u-1000",
        "v-50", "v-100", "v-150", "v-200", "v-250", "v-300", "v-400", "v-500", "v-600", "v-700", "v-850", "v-925", "v-1000",
        "t-50", "t-100", "t-150", "t-200", "t-250", "t-300", "t-400", "t-500", "t-600", "t-700", "t-850", "t-925", "t-1000",
        "q-50", "q-100", "q-150", "q-200", "q-250", "q-300", "q-400", "q-500", "q-600", "q-700", "q-850", "q-925", "q-1000",
    ],
    "satwnd": [
        "u-50", "u-100", "u-150", "u-200", "u-250", "u-300", "u-400", "u-500", "u-600", "u-700", "u-850", "u-925", "u-1000",
        "v-50", "v-100", "v-150", "v-200", "v-250", "v-300", "v-400", "v-500", "v-600", "v-700", "v-850", "v-925", "v-1000",
    ],
}

sat_auxiliary_vars = {
    "atms": [
        'obs_time', 'obs_latitude', 'obs_longitude', 'sat_id', 
        'scanline', 'fov', 'orbit_number', 'satellite_zenith_angle', 
        'satellite_azimuth_angle', 'solar_zenith_angle', 'solar_azimuth_angle', 
        'satellite_height', 'geolocation_quality_flags', 'scan_quality_flags', 'granule_quality_flags'
    ],
    "amsua": [
        "said", "siid", "fovn", "lsql", "saza", "soza", "hols", "hmsl", "solazi", "bearaz",
    ],
    "mhs": [
        "said", "siid", "fovn", "lsql", "saza", "soza", "hols", "hmsl", "solazi", "bearaz",
    ],
    "hrs4": [
        "said", "siid", "fovn", "lsql", "saza", "soza", "hols", "hmsl", "solazi", "bearaz",
    ]
}

sat_tmbrs_vars = {
    "atms": [
        "tmbrs_1", "tmbrs_2", "tmbrs_3", "tmbrs_4", "tmbrs_5", 
        "tmbrs_6", "tmbrs_7", "tmbrs_8", "tmbrs_9", "tmbrs_10", 
        "tmbrs_11", "tmbrs_12", "tmbrs_13", "tmbrs_14", "tmbrs_15", 
        "tmbrs_16", "tmbrs_17", "tmbrs_18", "tmbrs_19", "tmbrs_20",
        "tmbrs_21", "tmbrs_22" 
    ],
    "amsua": [
        "tmbrs_1", "tmbrs_2", "tmbrs_3", "tmbrs_4", "tmbrs_5", 
        "tmbrs_6", "tmbrs_7", "tmbrs_8", "tmbrs_9", "tmbrs_10", 
        "tmbrs_11", "tmbrs_12", "tmbrs_13", "tmbrs_14", "tmbrs_15", 
    ],
    "mhs": [
        "tmbrs_1", "tmbrs_2", "tmbrs_3", "tmbrs_4", "tmbrs_5", 
    ],
    "hrs4": [
        "tmbrs_1", "tmbrs_2", "tmbrs_3", "tmbrs_4", "tmbrs_5", 
        "tmbrs_6", "tmbrs_7", "tmbrs_8", "tmbrs_9", "tmbrs_10", 
        "tmbrs_11", "tmbrs_12", "tmbrs_13", "tmbrs_14", "tmbrs_15", 
        "tmbrs_16", "tmbrs_17", "tmbrs_18", "tmbrs_19", 
    ]
}

high_resolution = (721, 1440)
low_resolution = (181, 360)

def geographic_interpolate(data, interp_direction="hr2lr"):
    """
    基于经纬度坐标进行插值

    Parameters:
    data: 输入数据，形状为 [..., 721, 1440]
    interp_direction: "hr2lr" 或 "lr2hr"

    Returns:
    插值后的数据，形状根据方向决定
    """
    original_shape = data.shape
    spatial_dims = original_shape[-2:] # 最后两个维度是空间维度

    extra_dims = original_shape[:-2]
    total_extra = np.prod(extra_dims)

    # 修正：生成与数据第一个维度匹配的坐标点
    vars_coords = np.arange(total_extra)

    reshaped_data = data.reshape(total_extra, spatial_dims[0], spatial_dims[1])

    if interp_direction == "hr2lr":
        hr_lats = np.linspace(90, -90, high_resolution[0])
        hr_lons = np.linspace(0, 359.75, high_resolution[1])

        lr_lats = np.linspace(90, -90, low_resolution[0])
        lr_lons = np.linspace(0, 359, low_resolution[1])

        interpolated_data = interpolate_3d(reshaped_data, vars_coords, hr_lats, hr_lons, lr_lats, lr_lons)
        output_shape = extra_dims + low_resolution
    else:
        hr_lats = np.linspace(90, -90, high_resolution[0])
        hr_lons = np.linspace(0, 359.75, high_resolution[1])

        lr_lats = np.linspace(90, -90, low_resolution[0])
        lr_lons = np.linspace(0, 359, low_resolution[1])

        # 处理周期边界条件
        reshaped_data_extended = np.concatenate([reshaped_data, reshaped_data[:, :, 0:1]], axis=-1)
        lr_lons_extended = np.linspace(0, 360, low_resolution[1] + 1)

        interpolated_data = interpolate_3d(reshaped_data_extended, vars_coords, lr_lats, lr_lons_extended, hr_lats, hr_lons)
        output_shape = extra_dims + high_resolution

    interpolated_data = interpolated_data.reshape(output_shape)

    return interpolated_data

def interpolate_3d(data_3d, vars_coords, original_lats, original_lons, target_lats, target_lons):
    """
    对三维数据进行经纬度插值
    """
    # 转置数据以适应RegularGridInterpolator的输入格式
    data_3d_transpose = np.transpose(data_3d, axes=(1, 2, 0))

    # 创建插值器
    interpolator = RegularGridInterpolator(
        (original_lats, original_lons, vars_coords),
        data_3d_transpose,
        method='linear',
        bounds_error=False,
        fill_value=np.nan # 使用NaN填充超出范围的值
    )

    # 创建目标网格
    target_lat_grid, target_lon_grid, target_var_grid = np.meshgrid(
        target_lats, target_lons, vars_coords, indexing='ij'
    )

    # 构建插值点
    target_points = np.stack([
        target_lat_grid.ravel(),
        target_lon_grid.ravel(),
        target_var_grid.ravel()
    ], axis=-1)

    # 执行插值
    interpolated_values = interpolator(target_points)

    # 重塑结果
    target_resolution = (len(target_lats), len(target_lons), len(vars_coords))
    interpolated_values = interpolated_values.reshape(target_resolution)

    # 转置回原始维度顺序
    return np.transpose(interpolated_values, axes=(2, 0, 1))

def get_era5(file_path, shape):
    data = np.load(file_path).reshape(shape)

    return data

def get_normalize(
    scale_dir: str, 
    variables
):
    normalize_mean = dict(np.load(os.path.join(scale_dir, "normalize_mean.npz")))
    mean = []
    for var in variables:
        if var != "tp":
            mean.append(normalize_mean[var].reshape(1))
        else:
            mean.append(np.array([0.0]).reshape(1))
    normalize_mean = np.concatenate(mean)
    normalize_std = dict(np.load(os.path.join(scale_dir, "normalize_std.npz")))
    normalize_std = np.concatenate([normalize_std[var].reshape(1) for var in variables])

    return normalize_mean.reshape(1, -1, 1, 1), normalize_std.reshape(1, -1, 1, 1)

def get_climatology(clim_dir, shape, current_time, vairables):
    data = []
    for var in vairables:
        file_path = os.path.join(
            clim_dir,
            f"{current_time.month:02d}-{current_time.day:02d}",
            f"{var}.npy",
        )
        data.append(get_era5(file_path, shape))

    data = np.stack(data, axis=1)

    return data

def prepare_atms(
    obs_dir,
    metadata_vars,        # LOW#4 fix: 重命名（原 out_atms_vars — 实参是 metadata，非 out）
    atms_tmbrs_vars
):
    tmbrs_mean, tmbrs_std = get_normalize(f"{obs_dir}/1batms_merged_npy_1.0deg", atms_tmbrs_vars)
    satellite_height_scaler = dict(np.load(os.path.join(f"{obs_dir}/1batms_merged_npy_1.0deg", "satellite_height_scaler.npz")))
    scan_quality_flags_scaler = dict(np.load(os.path.join(f"{obs_dir}/1batms_merged_npy_1.0deg", "scan_quality_flags_scaler.npz")))
    geolocation_quality_flags_scaler = dict(np.load(os.path.join(f"{obs_dir}/1batms_merged_npy_1.0deg", "geolocation_quality_flags_scaler.npz")))
    granule_quality_flags_scaler = dict(np.load(os.path.join(f"{obs_dir}/1batms_merged_npy_1.0deg", "granule_quality_flags_scaler.npz")))
    with open(f"{obs_dir}/1batms_merged_npy_1.0deg/atms_1.0deg_schema.json", "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    atms_dict = {
        "tmbrs_mean": tmbrs_mean,
        "tmbrs_std": tmbrs_std,
        "satellite_height_scaler": satellite_height_scaler,
        "scan_quality_flags_scaler": scan_quality_flags_scaler,
        "geolocation_quality_flags_scaler": geolocation_quality_flags_scaler,
        "granule_quality_flags_scaler": granule_quality_flags_scaler,
        "meta_data": meta_data,
        "metadata_vars": metadata_vars,  # LOW#4 fix: 原 out_tmbrs_vars 命名误导（实测无人读）
        "tmbrs_vars": atms_tmbrs_vars
    }

    return atms_dict

def get_atms(
    obs_dir, 
    obs_time, 
    auxiliary_vars,
    tmbrs_vars,
    obs_dict,
    num_lat: int = 181,
    num_lon: int = 360,
):     
    tmbrs_shape = (len(tmbrs_vars), num_lat, num_lon)
    auxiliary_shape = (len(auxiliary_vars), num_lat, num_lon)
    auxiliarty_path = os.path.join(
        obs_dir,
        "1batms_merged_npy_1.0deg",
        f"{obs_time.year:04d}",
        f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
        f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-auxiliary_value.npy",
    )
    tmbrs_path = os.path.join(
        obs_dir,
        "1batms_merged_npy_1.0deg",
        f"{obs_time.year:04d}",
        f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
        f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-brightness_temperature_value.npy",
    )
    mask_path = os.path.join(
        obs_dir,
        "1batms_merged_npy_1.0deg",
        f"{obs_time.year:04d}",
        f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
        f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-mask.npy",
    )
    if os.path.exists(mask_path):
        auxiliary_value = np.load(auxiliarty_path)
        tmbrs_value = np.load(tmbrs_path)
        np_mask = np.load(mask_path)
        np_tmbrs_data = tmbrs_value.astype(np.float32)
        np_auxiliary_data = auxiliary_value.astype(np.float32)
        scanline_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("scanline")
        np_auxiliary_data[scanline_idx] = np_auxiliary_data[scanline_idx] / 12
        fovn_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("fov")
        np_auxiliary_data[fovn_idx] = np_auxiliary_data[fovn_idx] / 96
        orbit_number_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("orbit_number")
        np_auxiliary_data[orbit_number_idx] = np_auxiliary_data[orbit_number_idx]
        saza_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("satellite_zenith_angle")
        np_auxiliary_data[saza_idx] = np.cos(np.deg2rad(np_auxiliary_data[saza_idx]))
        saaa_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("satellite_azimuth_angle")
        np_auxiliary_data[saaa_idx] = np.cos(np.deg2rad(np_auxiliary_data[saaa_idx] / 2))
        soza_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("solar_zenith_angle")
        np_auxiliary_data[soza_idx] = np.cos(np.deg2rad(np_auxiliary_data[soza_idx]))
        soaa_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("solar_azimuth_angle")
        np_auxiliary_data[soaa_idx] = np.cos(np.deg2rad(np_auxiliary_data[soaa_idx] / 2))
        satellite_height_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("satellite_height")
        np_auxiliary_data[satellite_height_idx] = (np_auxiliary_data[satellite_height_idx] - obs_dict["satellite_height_scaler"]["satellite_height_min"]) / (obs_dict["satellite_height_scaler"]["satellite_height_max"] - obs_dict["satellite_height_scaler"]["satellite_height_min"])
        geolocation_quality_flags_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("geolocation_quality_flags")
        np_auxiliary_data[geolocation_quality_flags_idx] = (np_auxiliary_data[geolocation_quality_flags_idx] - obs_dict["geolocation_quality_flags_scaler"]["geolocation_quality_flags_min"]) / (obs_dict["geolocation_quality_flags_scaler"]["geolocation_quality_flags_max"] - obs_dict["geolocation_quality_flags_scaler"]["geolocation_quality_flags_min"])
        scan_quality_flags_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("scan_quality_flags")
        np_auxiliary_data[scan_quality_flags_idx] = (np_auxiliary_data[scan_quality_flags_idx] - obs_dict["scan_quality_flags_scaler"]["scan_quality_flags_min"]) / (obs_dict["scan_quality_flags_scaler"]["scan_quality_flags_max"] - obs_dict["scan_quality_flags_scaler"]["scan_quality_flags_min"])
        granule_quality_flags_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("granule_quality_flags")
        np_auxiliary_data[granule_quality_flags_idx] = (np_auxiliary_data[granule_quality_flags_idx] - obs_dict["granule_quality_flags_scaler"]["granule_quality_flags_min"]) / (obs_dict["granule_quality_flags_scaler"]["granule_quality_flags_max"] - obs_dict["granule_quality_flags_scaler"]["granule_quality_flags_min"])
    else:
        np_tmbrs_data = (np.ones(tmbrs_shape) * np.nan).astype(np.float32)
        np_auxiliary_data = (np.ones(auxiliary_shape) * np.nan).astype(np.float32)
        np_mask = (np.zeros(tmbrs_shape[-2:])).astype(np.int32)

    # 对齐训练侧 src/datamodules/assimilate/random_bg/npydataset.py:267-273：
    # tmbrs 与 auxiliary 都显式乘 mask（mask=0 区域强制清零，np.nan_to_num 仅替换 NaN 不够）
    nan_mask = np.nan_to_num(np_mask)[None]  # (1, H, W) 广播到 (C, H, W)
    return (np.nan_to_num(np_tmbrs_data) * nan_mask,
            np.nan_to_num(np_auxiliary_data) * nan_mask,
            np.nan_to_num(np_mask))

def prepare_amsua(
    obs_dir,
    out_amsua_vars,
    amsua_tmbrs_vars
):
    tmbrs_mean, tmbrs_std = get_normalize(f"{obs_dir}/1bamsua_merged_npy_1.0deg", amsua_tmbrs_vars)
    amsua_hols_scaler = dict(np.load(os.path.join(f"{obs_dir}/1bamsua_merged_npy_1.0deg", "hols_scaler.npz")))
    amsua_hmsl_scaler = dict(np.load(os.path.join(f"{obs_dir}/1bamsua_merged_npy_1.0deg", "hmsl_scaler.npz")))
    with open(f"{obs_dir}/1bamsua_merged_npy_1.0deg/amsua_1.0deg_schema.json", "r", encoding="utf-8") as f:
        amsua_meta_data = json.load(f)

    amsua_dict = {
        "tmbrs_mean": tmbrs_mean,
        "tmbrs_std": tmbrs_std,
        "hols_scaler": amsua_hols_scaler,
        "hmsl_scaler": amsua_hmsl_scaler,
        "meta_data": amsua_meta_data,
        "out_tmbrs_vars": out_amsua_vars,
        "tmbrs_vars": amsua_tmbrs_vars
    }

    return amsua_dict

def get_amsua(
    obs_dir, 
    obs_time, 
    auxiliary_vars,
    tmbrs_vars,
    obs_dict,
    num_lat: int = 181,
    num_lon: int = 360,
):     
    tmbrs_shape = (len(tmbrs_vars), num_lat, num_lon)
    auxiliary_shape = (len(auxiliary_vars), num_lat, num_lon)
    auxiliarty_path = os.path.join(
        obs_dir,
        "1bamsua_merged_npy_1.0deg",
        f"{obs_time.year:04d}",
        f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
        f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-auxiliary_value.npy",
    )
    tmbrs_path = os.path.join(
        obs_dir,
        "1bamsua_merged_npy_1.0deg",
        f"{obs_time.year:04d}",
        f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
        f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-tmbrs_value.npy",
    )
    mask_path = os.path.join(
        obs_dir,
        "1bamsua_merged_npy_1.0deg",
        f"{obs_time.year:04d}",
        f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
        f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-mask.npy",
    )
    if os.path.exists(mask_path):
        auxiliary_value = np.load(auxiliarty_path)
        tmbrs_value = np.load(tmbrs_path)
        np_mask = np.load(mask_path)
        np_tmbrs_data = tmbrs_value.astype(np.float32)
        np_auxiliary_data = auxiliary_value.astype(np.float32)
        fovn_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("fovn")
        np_auxiliary_data[fovn_idx] = np_auxiliary_data[fovn_idx] / 30
        lsql_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("lsql")
        np_auxiliary_data[lsql_idx] = np_auxiliary_data[lsql_idx] / 2
        saza_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("saza")
        np_auxiliary_data[saza_idx] = np.cos(np.deg2rad(np_auxiliary_data[saza_idx]))
        soza_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("soza")
        np_auxiliary_data[soza_idx] = np.cos(np.deg2rad(np_auxiliary_data[soza_idx]))
        hols_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("hols")
        np_auxiliary_data[hols_idx] = (np_auxiliary_data[hols_idx] - obs_dict["hols_scaler"]["hols_min"]) / (obs_dict["hols_scaler"]["hols_max"] - obs_dict["hols_scaler"]["hols_min"])
        hmsl_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("hmsl")
        np_auxiliary_data[hmsl_idx] = (np_auxiliary_data[hmsl_idx] - obs_dict["hmsl_scaler"]["hmsl_min"]) / (obs_dict["hmsl_scaler"]["hmsl_max"] - obs_dict["hmsl_scaler"]["hmsl_min"])
        solazi_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("solazi")
        np_auxiliary_data[solazi_idx] = np.cos(np.deg2rad(np_auxiliary_data[solazi_idx]) / 2) 
        bearaz_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("bearaz")
        np_auxiliary_data[bearaz_idx] = np.cos(np.deg2rad(np_auxiliary_data[bearaz_idx]) / 2) 
    else:
        np_tmbrs_data = (np.ones(tmbrs_shape) * np.nan).astype(np.float32)
        np_auxiliary_data = (np.ones(auxiliary_shape) * np.nan).astype(np.float32)
        np_mask = (np.zeros(tmbrs_shape[-2:])).astype(np.int32)

    # 对齐训练侧 src/datamodules/assimilate/random_bg/npydataset.py:267-273：
    # tmbrs 与 auxiliary 都显式乘 mask（mask=0 区域强制清零，np.nan_to_num 仅替换 NaN 不够）
    nan_mask = np.nan_to_num(np_mask)[None]  # (1, H, W) 广播到 (C, H, W)
    return (np.nan_to_num(np_tmbrs_data) * nan_mask,
            np.nan_to_num(np_auxiliary_data) * nan_mask,
            np.nan_to_num(np_mask))

def prepare_mhs(
    obs_dir,
    out_mhs_vars,
    mhs_tmbrs_vars
):
    tmbrs_mean, tmbrs_std = get_normalize(f"{obs_dir}/1bmhs_merged_npy_1.0deg", mhs_tmbrs_vars)
    mhs_hols_scaler = dict(np.load(os.path.join(f"{obs_dir}/1bmhs_merged_npy_1.0deg", "hols_scaler.npz")))
    mhs_hmsl_scaler = dict(np.load(os.path.join(f"{obs_dir}/1bmhs_merged_npy_1.0deg", "hmsl_scaler.npz")))
    with open(f"{obs_dir}/1bmhs_merged_npy_1.0deg/mhs_1.0deg_schema.json", "r", encoding="utf-8") as f:
        mhs_meta_data = json.load(f)

    mhs_dict = {
        "tmbrs_mean": tmbrs_mean,
        "tmbrs_std": tmbrs_std,
        "hols_scaler": mhs_hols_scaler,
        "hmsl_scaler": mhs_hmsl_scaler,
        "meta_data": mhs_meta_data,
        "out_vars": out_mhs_vars,
        "tmbrs_vars": mhs_tmbrs_vars
    }

    return mhs_dict

def get_mhs(
    obs_dir, 
    obs_time, 
    auxiliary_vars,
    tmbrs_vars,
    obs_dict,
    num_lat: int = 181,
    num_lon: int = 360,
):     
    tmbrs_shape = (len(tmbrs_vars), num_lat, num_lon)
    auxiliary_shape = (len(auxiliary_vars), num_lat, num_lon)
    auxiliarty_path = os.path.join(
        obs_dir,
        "1bmhs_merged_npy_1.0deg",
        f"{obs_time.year:04d}",
        f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
        f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-auxiliary_value.npy",
    )
    tmbrs_path = os.path.join(
        obs_dir,
        "1bmhs_merged_npy_1.0deg",
        f"{obs_time.year:04d}",
        f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
        f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-tmbrs_value.npy",
    )
    mask_path = os.path.join(
        obs_dir,
        "1bmhs_merged_npy_1.0deg",
        f"{obs_time.year:04d}",
        f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
        f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-mask.npy",
    )
    if os.path.exists(mask_path):
        auxiliary_value = np.load(auxiliarty_path)
        tmbrs_value = np.load(tmbrs_path)
        np_mask = np.load(mask_path)
        np_tmbrs_data = tmbrs_value.astype(np.float32)
        np_auxiliary_data = auxiliary_value.astype(np.float32)
        fovn_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("fovn")
        # 对齐训练侧 src/datamodules/assimilate/random_bg/npydataset.py:399 (mhs fovn/90)
        np_auxiliary_data[fovn_idx] = np_auxiliary_data[fovn_idx] / 90
        lsql_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("lsql")
        np_auxiliary_data[lsql_idx] = np_auxiliary_data[lsql_idx] / 2
        saza_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("saza")
        np_auxiliary_data[saza_idx] = np.cos(np.deg2rad(np_auxiliary_data[saza_idx]))
        soza_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("soza")
        np_auxiliary_data[soza_idx] = np.cos(np.deg2rad(np_auxiliary_data[soza_idx]))
        hols_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("hols")
        np_auxiliary_data[hols_idx] = (np_auxiliary_data[hols_idx] - obs_dict["hols_scaler"]["hols_min"]) / (obs_dict["hols_scaler"]["hols_max"] - obs_dict["hols_scaler"]["hols_min"])
        hmsl_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("hmsl")
        np_auxiliary_data[hmsl_idx] = (np_auxiliary_data[hmsl_idx] - obs_dict["hmsl_scaler"]["hmsl_min"]) / (obs_dict["hmsl_scaler"]["hmsl_max"] - obs_dict["hmsl_scaler"]["hmsl_min"])
        solazi_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("solazi")
        np_auxiliary_data[solazi_idx] = np.cos(np.deg2rad(np_auxiliary_data[solazi_idx]) / 2) 
        bearaz_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("bearaz")
        np_auxiliary_data[bearaz_idx] = np.cos(np.deg2rad(np_auxiliary_data[bearaz_idx]) / 2) 
    else:
        np_tmbrs_data = (np.ones(tmbrs_shape) * np.nan).astype(np.float32)
        np_auxiliary_data = (np.ones(auxiliary_shape) * np.nan).astype(np.float32)
        np_mask = (np.zeros(tmbrs_shape[-2:])).astype(np.int32)

    # 对齐训练侧 src/datamodules/assimilate/random_bg/npydataset.py:267-273：
    # tmbrs 与 auxiliary 都显式乘 mask（mask=0 区域强制清零，np.nan_to_num 仅替换 NaN 不够）
    nan_mask = np.nan_to_num(np_mask)[None]  # (1, H, W) 广播到 (C, H, W)
    return (np.nan_to_num(np_tmbrs_data) * nan_mask,
            np.nan_to_num(np_auxiliary_data) * nan_mask,
            np.nan_to_num(np_mask))

def prepare_hrs4(
    obs_dir,
    out_hrs4_vars,
    hrs4_tmbrs_vars
):
    tmbrs_mean, tmbrs_std = get_normalize(f"{obs_dir}/1bhrs4_merged_npy_1.0deg", hrs4_tmbrs_vars)
    hrs4_hols_scaler = dict(np.load(os.path.join(f"{obs_dir}/1bhrs4_merged_npy_1.0deg", "hols_scaler.npz")))
    hrs4_hmsl_scaler = dict(np.load(os.path.join(f"{obs_dir}/1bhrs4_merged_npy_1.0deg", "hmsl_scaler.npz")))
    with open(f"{obs_dir}/1bhrs4_merged_npy_1.0deg/hrs4_1.0deg_schema.json", "r", encoding="utf-8") as f:
        hrs4_meta_data = json.load(f)

    hrs4_dict = {
        "tmbrs_mean": tmbrs_mean,
        "tmbrs_std": tmbrs_std,
        "hols_scaler": hrs4_hols_scaler,
        "hmsl_scaler": hrs4_hmsl_scaler,
        "meta_data": hrs4_meta_data,
        "out_vars": out_hrs4_vars,
        "tmbrs_vars": hrs4_tmbrs_vars
    }

    return hrs4_dict

def get_hrs4(
    obs_dir, 
    obs_time, 
    auxiliary_vars,
    tmbrs_vars,
    obs_dict,
    num_lat: int = 181,
    num_lon: int = 360,
):     
    tmbrs_shape = (len(tmbrs_vars), num_lat, num_lon)
    auxiliary_shape = (len(auxiliary_vars), num_lat, num_lon)
    auxiliarty_path = os.path.join(
        obs_dir,
        "1bhrs4_merged_npy_1.0deg",
        f"{obs_time.year:04d}",
        f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
        f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-auxiliary_value.npy",
    )
    tmbrs_path = os.path.join(
        obs_dir,
        "1bhrs4_merged_npy_1.0deg",
        f"{obs_time.year:04d}",
        f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
        f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-tmbrs_value.npy",
    )
    mask_path = os.path.join(
        obs_dir,
        "1bhrs4_merged_npy_1.0deg",
        f"{obs_time.year:04d}",
        f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
        f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-mask.npy",
    )
    if os.path.exists(mask_path):
        auxiliary_value = np.load(auxiliarty_path)
        tmbrs_value_full = np.load(tmbrs_path)
        # 防御性切片（MEDIUM#4）：旧代码硬编码 `[:-1]` 假设 npy 通道数 == len(tmbrs_vars) + 1。
        # 若上游数据预处理脚本变更 npy 通道数，旧的固定 slice 会静默错位或崩溃。
        # 新逻辑：expected = len(tmbrs_vars)，若 full == expected + 1 则 drop last，
        # 否则等于 expected 直接用，否则 raise。
        expected_channels = len(tmbrs_vars)
        actual_channels = tmbrs_value_full.shape[0]
        if actual_channels == expected_channels + 1:
            tmbrs_value = tmbrs_value_full[:-1]
        elif actual_channels == expected_channels:
            tmbrs_value = tmbrs_value_full
        else:
            raise ValueError(
                f"hrs4 tmbrs npy has {actual_channels} channels, "
                f"expected {expected_channels} or {expected_channels + 1}"
            )
        np_mask = np.load(mask_path)
        np_tmbrs_data = tmbrs_value.astype(np.float32)
        np_auxiliary_data = auxiliary_value.astype(np.float32)
        fovn_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("fovn")
        np_auxiliary_data[fovn_idx] = np_auxiliary_data[fovn_idx] / 56
        lsql_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("lsql")
        np_auxiliary_data[lsql_idx] = np_auxiliary_data[lsql_idx] / 2
        saza_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("saza")
        np_auxiliary_data[saza_idx] = np.cos(np.deg2rad(np_auxiliary_data[saza_idx]))
        soza_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("soza")
        np_auxiliary_data[soza_idx] = np.cos(np.deg2rad(np_auxiliary_data[soza_idx]))
        hols_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("hols")
        np_auxiliary_data[hols_idx] = (np_auxiliary_data[hols_idx] - obs_dict["hols_scaler"]["hols_min"]) / (obs_dict["hols_scaler"]["hols_max"] - obs_dict["hols_scaler"]["hols_min"])
        hmsl_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("hmsl")
        np_auxiliary_data[hmsl_idx] = (np_auxiliary_data[hmsl_idx] - obs_dict["hmsl_scaler"]["hmsl_min"]) / (obs_dict["hmsl_scaler"]["hmsl_max"] - obs_dict["hmsl_scaler"]["hmsl_min"])
        solazi_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("solazi")
        np_auxiliary_data[solazi_idx] = np.cos(np.deg2rad(np_auxiliary_data[solazi_idx]) / 2) 
        bearaz_idx = obs_dict["meta_data"]["auxiliary_value"]["fields_in_order"].index("bearaz")
        np_auxiliary_data[bearaz_idx] = np.cos(np.deg2rad(np_auxiliary_data[bearaz_idx]) / 2) 
    else:
        np_tmbrs_data = (np.ones(tmbrs_shape) * np.nan).astype(np.float32)
        np_auxiliary_data = (np.ones(auxiliary_shape) * np.nan).astype(np.float32)
        np_mask = (np.zeros(tmbrs_shape[-2:])).astype(np.int32)

    # 对齐训练侧 src/datamodules/assimilate/random_bg/npydataset.py:267-273：
    # tmbrs 与 auxiliary 都显式乘 mask（mask=0 区域强制清零，np.nan_to_num 仅替换 NaN 不够）
    nan_mask = np.nan_to_num(np_mask)[None]  # (1, H, W) 广播到 (C, H, W)
    return (np.nan_to_num(np_tmbrs_data) * nan_mask,
            np.nan_to_num(np_auxiliary_data) * nan_mask,
            np.nan_to_num(np_mask))

def prepare_prepbufr(
    obs_dir,
    era5_lr_dir,
    prepbufr_vars,
):
    prepbufr_mean, prepbufr_std = get_normalize(f"{era5_lr_dir}/normalized_mean_std", prepbufr_vars)

    # LOW#3 fix: 删除死代码 `mult_prepbufr`（仅乘 std 后丢弃，无人读取）
    prepbufr_dict = {
        "prepbufr_mean": prepbufr_mean,
        "prepbufr_std": prepbufr_std,
        "prepbufr_vars": prepbufr_vars,
    }

    return prepbufr_dict

def get_prepbufr(
    obs_dir,
    current_time,
    daw,
    dt,
    prepbufr_vars,
    num_lat: int = 181,
    num_lon: int = 360,
):
    """加载 [current_time, current_time + daw) 内每 dt 步的 prepbufr 数据。

    返回：
      obs:  Tensor[n_steps, C, H, W]，缺失时刻填 0（已被 mask 屏蔽）
      mask: Tensor[n_steps, C, H, W]，缺失时刻填 0

    Mask 来源（与训练 npydataset.py:536-547 完全一致）：
      - 当 obs 文件存在时，直接加载同目录 ``_qcmask.npy`` 作为掩码（由
        ``data_factory/_sigma_utils.py`` 生成，定义为
        ``mask=1 if ~isnan(obs) AND |obs-era5| < 0.5*24h_deviation``）；
        该 mask 已包含背景场合理性 QC，远比 ``~np.isnan`` 严格。
      - 若 ``_qcmask.npy`` 不存在（data_factory 未跑过该时刻），该时次视为
        无有效观测（掩码全 0）——与训练侧语义一致（训练侧 qcmask 缺失即
        ``np.load`` 崩溃，观测不可用）。不再回退 ``~np.isnan``（训练从未见过的
        分布外掩码，会造成训练/推理分布失配）。
    """
    prepbufr_shape = (len(prepbufr_vars), num_lat, num_lon)
    obs_times = [current_time + relativedelta(hours=i) for i in range(0, daw, dt)]
    np_obs_data, np_obs_mask = [], []
    for obs_time in obs_times:
        obs_path = os.path.join(
            obs_dir,
            "GDAS_prepbufr_merged_npy_1.0deg",
            f"{obs_time.year:04d}",
            f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
            f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-obs_value.npy",
        )
        # 对齐训练侧 npydataset.py:540-547 加载的外部 _qcmask.npy 文件
        mask_path = os.path.join(
            obs_dir,
            "GDAS_prepbufr_merged_npy_1.0deg",
            f"{obs_time.year:04d}",
            f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
            f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}_qcmask.npy",
        )
        if os.path.exists(obs_path):
            obs_data = np.load(obs_path).squeeze()
            np_obs_data.append(obs_data)
            if os.path.exists(mask_path):
                # 与训练侧 npydataset.py:546 一致：直接加载外部 _qcmask.npy
                obs_mask = np.load(mask_path).squeeze()
            else:
                # 与训练侧对齐：qcmask 缺失则该时次观测不可用（训练侧 np.load 会
                # 直接崩溃）；推理侧置全 0 掩码（训练见过的缺测分布），
                # 不再静默回退 ~isnan（分布外掩码）。
                logging.getLogger(__name__).warning(
                    f"prepbufr qcmask missing, treating as no valid obs: {mask_path}"
                )
                obs_mask = np.zeros(prepbufr_shape)
            np_obs_mask.append(obs_mask.astype(np.float32))
        else:
            np_obs_data.append(np.ones(prepbufr_shape) * np.nan)
            np_obs_mask.append(np.zeros(prepbufr_shape))

    np_obs_data = np.nan_to_num(
        np.stack(np_obs_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    np_obs_mask = np.nan_to_num(
        np.stack(np_obs_mask, axis=0), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    prepbufrs = torch.from_numpy(np_obs_data)
    prepbufr_masks = torch.from_numpy(np_obs_mask)

    return prepbufrs * prepbufr_masks, prepbufr_masks

def prepare_satwnd(
    obs_dir,
    era5_lr_dir,
    satwnd_vars,
):
    satwnd_mean, satwnd_std = get_normalize(f"{era5_lr_dir}/normalized_mean_std", satwnd_vars)

    # LOW#3 fix: 删除死代码 `mult_satwnd`（仅乘 std 后丢弃，无人读取）
    satwnd_dict = {
        "satwnd_mean": satwnd_mean,
        "satwnd_std": satwnd_std,
        "satwnd_vars": satwnd_vars,
    }

    return satwnd_dict

def get_satwnd(
    obs_dir,
    current_time,
    daw,
    dt,
    satwnd_vars,
    num_lat: int = 181,
    num_lon: int = 360,
):
    """加载 satwnd 观测 + 掩码。

    Mask 来源（与训练 npydataset.py:583-594 完全一致）：
      - 当 obs 文件存在时，直接加载同目录 ``_qcmask.npy``（与 prepbufr 同样的
        ``0.5*24h_deviation`` 背景 QC）；
      - 若缺失则该时次视为无有效观测（掩码全 0），不再回退 ``~np.isnan``。
    """
    satwnd_shape = (len(satwnd_vars), num_lat, num_lon)
    obs_times = [current_time + relativedelta(hours=i) for i in range(0, daw, dt)]
    np_obs_data, np_obs_mask = [], []
    for obs_time in obs_times:
        obs_path = os.path.join(
            obs_dir,
            "satwnd_merged_npy_1.0deg",
            f"{obs_time.year:04d}",
            f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
            f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}-obs_value.npy",
        )
        # 对齐训练侧 npydataset.py:583-588 加载的外部 _qcmask.npy 文件
        mask_path = os.path.join(
            obs_dir,
            "satwnd_merged_npy_1.0deg",
            f"{obs_time.year:04d}",
            f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}",
            f"{obs_time.hour:02d}:{obs_time.minute:02d}:{obs_time.second:02d}_qcmask.npy",
        )
        if os.path.exists(obs_path):
            obs_data = np.load(obs_path).squeeze()
            np_obs_data.append(obs_data)
            if os.path.exists(mask_path):
                # 与训练侧 npydataset.py:593 一致：直接加载外部 _qcmask.npy
                obs_mask = np.load(mask_path).squeeze()
            else:
                # 与训练侧对齐：qcmask 缺失则该时次观测不可用，置全 0 掩码
                logging.getLogger(__name__).warning(
                    f"satwnd qcmask missing, treating as no valid obs: {mask_path}"
                )
                obs_mask = np.zeros(satwnd_shape)
            np_obs_mask.append(obs_mask.astype(np.float32))
        else:
            np_obs_data.append(np.ones(satwnd_shape) * np.nan)
            np_obs_mask.append(np.zeros(satwnd_shape))

    np_obs_data = np.nan_to_num(
        np.stack(np_obs_data, axis=0), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    np_obs_mask = np.nan_to_num(
        np.stack(np_obs_mask, axis=0), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    satwnds = torch.from_numpy(np_obs_data)
    satwnd_masks = torch.from_numpy(np_obs_mask)

    return satwnds * satwnd_masks, satwnd_masks

prepare_sat = {
    "atms": prepare_atms,
    "amsua": prepare_amsua,
    "mhs": prepare_mhs,
    "hrs4": prepare_hrs4,
    "prepbufr": prepare_prepbufr,
    "satwnd": prepare_satwnd,
}

get_sat = {
    "atms": get_atms,
    "amsua": get_amsua,
    "mhs": get_mhs,
    "hrs4": get_hrs4,
    "prepbufr": get_prepbufr,
    "satwnd": get_satwnd,
}