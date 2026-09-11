# -*- coding: utf-8 -*-
"""``xichen-forecast``：ERA5 起报的 AR 中期预报评估（1.0° 网格）。"""
import os

import click

from xichen.device import get_device
from xichen.forecast_eval import eval_forecast, load_init_times, make_lr_loader


@click.command()
@click.option("--ckpt", "ckpt_path", type=str, required=True,
              help="预报 checkpoint：扁平 .ckpt 文件路径（或旧版 logs 目录）。")
@click.option("--era5_lr_dir", type=click.Path(exists=True), required=True,
              help="ERA5 1.0° npy 根目录（含 normalized_mean_std/ 与 climatology_np181x360_2010_2021/）。")
@click.option("--forecast_config", type=click.Path(exists=True), default="configs/xichen_forecast.json",
              show_default=True, help="预报模型超参 JSON（随仓库分发，ckpt 不含超参）。")
@click.option("--output_dir", type=str, default="./outputs/forecast", show_default=True)
@click.option("--forecast_hours", type=int, default=240, show_default=True)
@click.option("--start_year", type=int, default=2023, show_default=True)
@click.option("--end_year", type=int, default=2024, show_default=True)
@click.option("--decorrelation_hours", type=int, default=6, show_default=True)
@click.option("--init_times_json", type=click.Path(exists=True), default=None,
              help="可选：显式起报时刻 JSON（ISO 字符串 list，或含 'init_times' 键的 dict）。"
                   "不指定 = 内置全年逐 decorrelation_hours 起报。")
@click.option("--dt", type=click.IntRange(min=1, max=24), default=6, show_default=True,
              help="预报评估步长（小时）；cascade 子模型 (24,12,6,3,1)h 自适应。")
@click.option("--forecast_name", type=str, default="xichen_forecast", show_default=True,
              help="输出 CSV/图文件名前缀。")
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--output_resolution", type=click.Choice(["1p0", "0p25"]), default="1p0", show_default=True,
              help="预报场输出分辨率：1p0=原生 1.0°；0p25=插值到 0.25° (721×1440)。")
@click.option("--eval_batch_size", type=int, default=1, show_default=True,
              help="每批 forward 并行处理的 init time 数（数据并行）。")
def main(
    ckpt_path,
    era5_lr_dir,
    forecast_config,
    output_dir,
    forecast_hours,
    start_year,
    end_year,
    decorrelation_hours,
    init_times_json,
    dt,
    forecast_name,
    device,
    output_resolution,
    eval_batch_size,
):
    """AR rollout against ERA5 truth on the 1.0deg (LR) grid."""
    config = {
        "era5_lr_dir": era5_lr_dir,
        "era5_hr_dir": era5_lr_dir,  # 1.0° 评估不使用 HR 归一化；仅占位
        "forecast_config": forecast_config,
        "forecast_hours": forecast_hours,
        "start_year": start_year,
        "end_year": end_year,
        "decorrelation_hours": decorrelation_hours,
        "dt": dt,
        "forecast_name": forecast_name,
        "device": get_device(device, 0),
        "forecast_pair": "lr",
        "resolution_tag": "1p0deg",
        "output_resolution": output_resolution,
        "eval_batch_size": eval_batch_size,
    }
    if init_times_json is not None:
        config["init_times"] = load_init_times(init_times_json)
    eval_forecast(
        make_lr_loader(era5_lr_dir),
        ckpt_path,
        config,
        output_dir,
    )


if __name__ == "__main__":
    main()
