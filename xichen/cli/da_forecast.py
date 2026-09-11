# -*- coding: utf-8 -*-
"""``xichen-da-forecast``：从 DA 循环的 6h 分析场起报的 10 天预报评估。"""
import click

from xichen.device import get_device
from xichen.forecast_eval import (
    check_all_present,
    eval_forecast,
    load_init_times,
    make_dacycle_init_loader,
    make_lr_loader,
)


@click.command()
@click.option("--ic_dir", type=click.Path(exists=True), required=True,
              help="dacycle 输出的 ic_6h/ 目录（NWP-Benchmark lead-0 分析场）。")
@click.option("--init_times_json", type=click.Path(exists=True), required=True,
              help="起报时刻 JSON（ISO 字符串 list；需与 ic_6h/ 中的分析场一一对应）。")
@click.option("--ckpt", "ckpt_path", type=str, required=True,
              help="预报 checkpoint：扁平 .ckpt 文件路径（或旧版 logs 目录）。")
@click.option("--era5_lr_dir", type=click.Path(exists=True), required=True,
              help="ERA5 1.0° npy 根目录（评估真值与气候态来源）。")
@click.option("--forecast_config", type=click.Path(exists=True), default="configs/xichen_forecast.json",
              show_default=True, help="预报模型超参 JSON。")
@click.option("--output_dir", type=str, default="./outputs/da_forecast", show_default=True)
@click.option("--forecast_hours", type=int, default=240, show_default=True)
@click.option("--forecast_dt", type=click.IntRange(min=1, max=24), default=6, show_default=True,
              help="预报评估步长（小时）。forecast_hours 必须能被 forecast_dt 整除。")
@click.option("--forecast_name", type=str, default="xichen_da_forecast", show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--output_resolution", type=click.Choice(["1p0", "0p25"]), default="1p0", show_default=True,
              help="预报场输出分辨率；IC 读取恒为 1.0°。")
@click.option("--eval_batch_size", type=int, default=1, show_default=True,
              help="每批 forward 并行处理的 init time 数。显存参考：单卡 24GB ≈ 4~8；80GB ≈ 16~32。")
def main(
    ic_dir,
    init_times_json,
    ckpt_path,
    era5_lr_dir,
    forecast_config,
    output_dir,
    forecast_hours,
    forecast_dt,
    forecast_name,
    device,
    output_resolution,
    eval_batch_size,
):
    """AR rollout from DA 6h analyses, scored against ERA5 truth on the 1.0deg grid."""
    init_times = load_init_times(init_times_json)
    check_all_present(ic_dir, init_times)

    config = {
        "era5_lr_dir": era5_lr_dir,
        "era5_hr_dir": era5_lr_dir,
        "forecast_config": forecast_config,
        "forecast_hours": forecast_hours,
        "dt": forecast_dt,
        "forecast_name": forecast_name,
        "device": get_device(device, 0),
        "forecast_pair": "lr",
        "resolution_tag": "dainit_1p0deg",
        "init_times": init_times,
        "eval_batch_size": eval_batch_size,
        "output_resolution": output_resolution,
    }
    eval_forecast(
        make_lr_loader(era5_lr_dir),
        ckpt_path,
        config,
        output_dir,
        init_loader=make_dacycle_init_loader(ic_dir),
    )


if __name__ == "__main__":
    main()
