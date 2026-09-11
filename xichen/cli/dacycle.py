# -*- coding: utf-8 -*-
"""``xichen-dacycle``：确定性级联 DA 循环入口（12h 窗口 + 00/12UTC 6h 重分析）。"""
import os

from xichen.dacycle.common import (
    build_common_parser,
    load_config_and_env,
    log,
    print_timing_summary,
    run_dacycle_loop,
)
from xichen.dacycle.det import assim_cycle_xichen


def main():
    parser = build_common_parser()
    parser.add_argument(
        "--save_format",
        choices=["nwp", "npy", "both"],
        default=os.environ.get("SAVE_FORMAT", "nwp"),
        help=(
            "ic_6h 分析场的写出格式；默认 nwp（只写 nc）。"
            "npy = 只写 .npy（3D (69,181,360)）；both = nc + npy 双写。"
            "env var SAVE_FORMAT 可覆盖默认值。"
        ),
    )
    args = parser.parse_args()

    log.info("dacycle mode: DETERMINISTIC")

    cfg, cascade_cfg, components, obs_dict, device = load_config_and_env(args)
    cfg["save_format"] = args.save_format
    log.info("dacycle save_format: %s", cfg["save_format"])

    def _det_step(cfg, T, prev_xa, components, obs_dict, device, state):
        prev_xa, cycle_timing = assim_cycle_xichen(
            cfg, T, prev_xa, components, obs_dict, device,
        )
        return prev_xa, cycle_timing, state

    timing = run_dacycle_loop(cfg, components, obs_dict, device, _det_step)
    print_timing_summary(cfg, timing)


if __name__ == "__main__":
    main()
