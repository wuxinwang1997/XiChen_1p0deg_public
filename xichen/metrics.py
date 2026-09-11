"""
纬度加权 ACC / RMSE / MAE / Bias / Activity 指标

为 ERA5 类全球网格预报（``(N, C, H, W)``）提供球面纬度加权评估。
同时提供两套实现：

- **NumPy 版**（``*_channels`` / ``weighted_*``）：CPU 友好，多用于
  评估脚本与可视化后处理。
- **PyTorch 版**（``weighted_*_torch*``）：与训练在同一 device 上运行，
  无须 ``.cpu().numpy()`` 来回切换，可直接放进 validation loop。

权重定义：``weight[j] = num_lat * cos(lat_j) / Σ_k cos(lat_k)``，
等价于"按面积归一化的纬度 cos 权重"。

上游依赖：被 ``src/pipeline/*/trainer.py`` 在 validation step 中调用；
被 ``plots/plot_forecast_metrics.py`` 用于画图。
下游调用：纯函数 / ``np.ndarray`` / ``torch.Tensor``，无副作用。
"""

import numpy as np
import torch


# NumPy 版本
def lat_np(j: np.ndarray, num_lat: int) -> np.ndarray:
    """把网格索引 ``j``（0=北极）转换为纬度（度）。

    Args:
        j: 纬度索引数组，``0`` 对应 ``90°N``。
        num_lat: 纬度总格点数。

    Returns:
        纬度（度）的 ndarray，``90 - j * 180 / (num_lat - 1)``。
    """
    return 90 - j * 180/float(num_lat-1)


def latitude_weighting_factor(j: np.ndarray, num_lat: int, s: np.ndarray) -> np.ndarray:
    """返回纬度加权因子 ``num_lat * cos(lat_j) / s``，``s`` 是 ``Σ cos``。

    Args:
        j: 纬度索引数组。
        num_lat: 纬度总格点数。
        s: ``Σ cos(lat)`` 的全局归一化常数（保证 ``Σ weight = num_lat``）。

    Returns:
        加权因子 ndarray，shape 与 ``j`` 同。
    """
    return num_lat * np.cos(3.1416/180. * lat_np(j, num_lat))/s


def weighted_acc_channels(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """每个通道的纬度加权异常相关系数（ACC）。

    输入形状 ``(N, C, H, W)``；返回 ``(C, 1, 1, 1)`` 的逐通道 ACC。

    ACC 定义：``Σ weight * pred * target /
    sqrt( Σ weight * pred² * Σ weight * target² )``，
    与纬度加权 climate 距平的标准 ACC 一致。

    Args:
        pred: 模型预测，已是距平（减气候态后）。
        target: 真值距平。

    Returns:
        shape ``(C, 1, 1, 1)`` 的 ACC。
    """
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[-2]
    lat_t = np.arange(start=0, stop=num_lat)
    s = np.sum(np.cos(3.1416/180. * lat_np(lat_t, num_lat)))
    weight = np.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    # 标准 ACC：协方差 / (σ_pred * σ_target)
    result = np.sum(weight * pred * target, axis=(-1,-2), keepdims=True) / np.sqrt(np.sum(weight * pred * pred, axis=(-1,-2), keepdims=True) * np.sum(weight * target *
    target, axis=(-1,-2), keepdims=True))
    return result


def weighted_acc(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """逐 batch 的纬度加权 ACC（``weighted_acc_channels`` 后再对 batch 求均值）。

    Returns:
        shape ``(1, 1, 1, 1)`` 的平均 ACC。
    """
    result = weighted_acc_channels(pred, target)
    return np.mean(result, axis=0, keepdims=True)


def weighted_rmse_channels(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """每个通道的纬度加权 RMSE。

    输入形状 ``(N, C, H, W)``；返回 ``(C, 1, 1, 1)`` 的逐通道 RMSE。

    Args:
        pred: 模型预测。
        target: 真值。

    Returns:
        shape ``(C, 1, 1, 1)`` 的 RMSE。
    """
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[-2]
    lat_t = np.arange(start=0, stop=num_lat)

    s = np.sum(np.cos(3.1416/180. * lat_np(lat_t, num_lat)))
    weight = np.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    # weight = 1
    result = np.sqrt(np.mean(weight * (pred - target)**2., axis=(-1,-2), keepdims=True))
    return result


def weighted_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """逐 batch 的纬度加权 RMSE（``weighted_rmse_channels`` 后再对 batch 求均值）。"""
    result = weighted_rmse_channels(pred, target)
    return np.mean(result, axis=0)


def weighted_mae_channels(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """每个通道的纬度加权 MAE。

    Args:
        pred: 模型预测。
        target: 真值。

    Returns:
        shape ``(C, 1, 1, 1)`` 的 MAE。
    """
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[-2]
    lat_t = np.arange(start=0, stop=num_lat)

    s = np.sum(np.cos(3.1416/180. * lat_np(lat_t, num_lat)))
    weight = np.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    # weight = 1
    result = np.mean(weight * np.abs(pred - target), axis=(-1,-2), keepdims=True)
    return result


def weighted_mae(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """逐 batch 的纬度加权 MAE（``weighted_mae_channels`` 后再对 batch 求均值）。"""
    result = weighted_mae_channels(pred, target)
    return np.mean(result, axis=0)


def type_weighted_activity_channels(pred: np.ndarray) -> np.ndarray:
    """每个通道的纬度加权"activity"（标准差大小，反映预报波动幅度）。

    Args:
        pred: 已减气候态的预测距平，shape ``(N, C, H, W)``。

    Returns:
        shape ``(N, C)`` 的 activity。
    """
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[-2]
    #num_long = target.shape[2]
    lat_t = np.arange(start=0, stop=num_lat)

    northern_index = int(110. / 180. * num_lat + 0.5)
    souther_index = int(70. / 180. * num_lat + 0.5)

    s = np.sum(np.cos(3.1416/180. * lat_np(lat_t, num_lat)))
    weight = np.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = np.sqrt(np.mean(weight * (pred - np.mean(weight * pred, axis=(-1, -2), keepdims=True)) ** 2, axis=(-1, -2)))
    return result


def weighted_activity(pred, clim_time_mean_daily, data_std=None):
    """纬度加权 activity 指标。

    Args:
        pred: 模型预测；若是已标准化空间则 ``data_std`` 传 ``None``。
        clim_time_mean_daily: 逐日气候态，用于去均值。
        data_std: 若 ``pred / clim_time_mean_daily`` 在标准化空间，
            传对应的 ``data_std`` 把结果反标准化。

    Returns:
        标量 activity（``np.mean`` 后）。
    """
    if data_std is None:
        result = np.mean(type_weighted_activity_channels(pred - clim_time_mean_daily), axis=0)
    else:
        result = np.mean(type_weighted_activity_channels(pred - clim_time_mean_daily) * data_std, axis=0)

    return result


# PyTorch 版本
def lat_torch(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    """``lat_np`` 的 PyTorch 版本：grid index -> 纬度（度）。"""
    return 90 - j * 180/float(num_lat-1)


def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    """``latitude_weighting_factor`` 的 PyTorch 版本。"""
    return num_lat * torch.cos(3.1416/180. * lat_torch(j, num_lat))/s


def weighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PyTorch 版逐通道纬度加权 ACC。

    Returns:
        shape ``(C, 1)`` 的 ACC。
    """
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[-2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat_torch(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
    target, dim=(-1,-2)))
    return result


def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PyTorch 版逐 batch 纬度加权 ACC。"""
    result = weighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)


def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PyTorch 版逐通道纬度加权 RMSE。

    Returns:
        shape ``(C, 1)`` 的 RMSE。
    """
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[-2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat_torch(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result


def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PyTorch 版逐 batch 纬度加权 RMSE。"""
    result = weighted_rmse_torch_channels(pred, target)
    return torch.mean(result, dim=0)


def weighted_bias_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PyTorch 版逐通道纬度加权 Bias（``mean(weight * (pred - target))``）。

    Returns:
        shape ``(C, 1)`` 的 Bias。
    """
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[-2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat_torch(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.mean(weight * (pred - target), dim=(-1,-2))
    return result


def weighted_bias_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PyTorch 版逐 batch 纬度加权 Bias。"""
    result = weighted_bias_torch_channels(pred, target)
    return torch.mean(result, dim=0)


def weighted_mae_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PyTorch 版逐通道纬度加权 MAE。

    Returns:
        shape ``(C, 1)`` 的 MAE。
    """
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[-2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat_torch(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.mean(weight * torch.abs(pred - target), dim=(-1,-2))
    return result


def weighted_mae_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PyTorch 版逐 batch 纬度加权 MAE。"""
    result = weighted_mae_torch_channels(pred, target)
    return torch.mean(result, dim=0)


def weighted_latitude_weighting_factor_torch(j: torch.Tensor, real_num_lat:int, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    """扩展版加权因子：``real_num_lat`` 可与 ``num_lat`` 不同（用于子区域计算）。

    Args:
        j: 纬度索引。
        real_num_lat: 实际计算时使用的"目标纬度分辨率"。
        num_lat: 当前数组所在纬度分辨率。
        s: 归一化常数。

    Returns:
        加权因子 tensor。
    """
    return real_num_lat * torch.cos(3.1416/180. * lat_torch(j, num_lat)) / s


def type_weighted_activity_torch_channels(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    """PyTorch 版逐通道纬度加权 activity（标准差大小）。

    Args:
        pred: 已减气候态的距平，shape ``(N, C, H, W)``。
        metric_type: ``"all"`` / ``"northern"`` / ``"southern"``，
            分别对应全区域、北半球（lat >= 70°N）、南半球（lat <= 40°S）。

    Returns:
        shape ``(C, 1)`` 的 activity。
    """
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[-2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    # 半球边界：~70°N 与 ~40°S（根据 num_lat 反推索引）
    northern_index = int(110. / 180. * num_lat + 0.5)
    souther_index = int(70. / 180. * num_lat + 0.5)

    if metric_type == "all":
        s = torch.sum(torch.cos(3.1416/180. * lat_torch(lat_t, num_lat)))
        weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t, num_lat, num_lat, s), (1, 1, -1, 1))
        result = torch.sqrt(torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        return result

    elif metric_type == "northern":
        northern_s = torch.sum(torch.cos(3.1416/180. * lat_torch(lat_t, num_lat))[northern_index:])
        northern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[northern_index:], souther_index, num_lat, northern_s), (1, 1, -1, 1))
        northern_result = torch.sqrt(torch.mean(northern_weight * (pred[:, :, northern_index:] - torch.mean(northern_weight * pred[:, :, northern_index:], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        return northern_result
    elif metric_type == "southern":
        southern_s = torch.sum(torch.cos(3.1416/180. * lat_torch(lat_t, num_lat))[:souther_index])
        southern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[:souther_index], souther_index, num_lat, southern_s), (1, 1, -1, 1))
        southern_result = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :souther_index] - torch.mean(southern_weight * pred[:, :, :souther_index], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        return southern_result


def weighted_activity_torch(pred, clim_time_mean_daily, data_std=None):
    """PyTorch 版纬度加权 activity。

    Args:
        pred: 模型预测距平。
        clim_time_mean_daily: 气候态。
        data_std: 若在标准化空间则传对应 ``data_std``，否则 ``None``。

    Returns:
        标量 activity（``torch.mean`` 后）。
    """
    if data_std is None:
        result = torch.mean(type_weighted_activity_torch_channels(pred - clim_time_mean_daily, metric_type="all"), dim=0)
    else:
        result = torch.mean(type_weighted_activity_torch_channels(pred - clim_time_mean_daily, metric_type="all") * data_std, dim=0)

    return result