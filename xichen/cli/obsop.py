# -*- coding: utf-8 -*-
"""``xichen-obsop``：观测算子 OMB 评估（产出 avg_obs_error.npz，即 DA 的 R 估计）。"""
import os

import click

from xichen.device import get_device
from xichen.obsop_eval import eval_obsoperator


@click.command()
@click.option("--era5_dir", type=click.Path(exists=True), required=True,
              help="ERA5 1.0° npy 根目录。")
@click.option("--obs_name", type=click.Choice(["atms", "amsua", "mhs", "hrs4"]), default="atms",
              show_default=True)
@click.option("--obs_dir", type=click.Path(exists=True), required=True,
              help="观测 npy 根目录（含 1b<obs>_merged_npy_1.0deg/）。")
@click.option("--ckpt", "model_name", type=str, required=True,
              help="观测算子 checkpoint：扁平 .ckpt 文件路径（或旧版 run 名）。")
@click.option("--obsop_config", type=click.Path(exists=True), default=None,
              help="观测算子超参 JSON；默认 configs/{obs_name}_obsop.json。")
@click.option("--save_dir", type=str, default="./outputs/obsop", show_default=True)
@click.option("--start_year", type=int, default=2023, show_default=True)
@click.option("--end_year", type=int, default=2024, show_default=True)
@click.option("--debug", is_flag=True, default=False,
              help="只评估 start_year 年 1 月 1–3 日（快速自检）。")
@click.option("--device", type=str, default="cuda", show_default=True)
def main(
    era5_dir,
    obs_name,
    obs_dir,
    model_name,
    obsop_config,
    save_dir,
    start_year,
    end_year,
    debug,
    device,
):
    """Run the OMB evaluation for one radiance observation operator."""
    device = get_device(device, 0)
    os.makedirs(f"{save_dir}", exist_ok=True)
    os.makedirs(f"{save_dir}/{obs_name}", exist_ok=True)
    eval_obsoperator(
        era5_dir,
        obs_name,
        obs_dir,
        save_dir,
        start_year,
        end_year,
        model_name,
        debug,
        device,
        obsop_config=obsop_config,
    )


if __name__ == "__main__":
    main()
