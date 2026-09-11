# -*- coding: utf-8 -*-
"""设备选择（开源推理版：仅 CUDA / CPU）。"""
import torch

__all__ = ["get_device"]


def get_device(device_type: str = "cuda", local_rank: int = 0) -> torch.device:
    """获取 ``torch.device``。

    Args:
        device_type: ``"cuda"`` 或 ``"cpu"``。``"cuda"`` 不可用时回退 CPU 并给出提示。
        local_rank: GPU 编号；默认 ``0``（单卡）。

    Returns:
        ``torch.device("cuda:{local_rank}")`` 或 ``torch.device("cpu")``。
    """
    if device_type == "cuda":
        if not torch.cuda.is_available():
            print("[xichen] CUDA 不可用，回退到 CPU（推理将非常缓慢）")
            return torch.device("cpu")
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")
