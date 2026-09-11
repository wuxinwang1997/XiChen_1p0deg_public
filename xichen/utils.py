# -*- coding: utf-8 -*-
"""共享日志工具（开源推理版）。

导出：

- ``setup_logger`` / ``get_logger``：主进程过滤的统一 logger。
- ``RankFilter``：logging.Filter 子类，仅允许 ``rank == 0`` 的日志通过。

设计要点：

- ``logger.handlers.clear() + propagate=False`` 保证同一个 logger 多次
  ``setup_logger`` 不会重复挂 handler，避免日志重复打印。
- ``RankFilter`` 同时挂到 console 与 file handler。
"""
import logging
import sys
from pathlib import Path
from typing import Optional

__all__ = ["setup_logger", "get_logger", "RankFilter"]


class RankFilter(logging.Filter):
    """日志过滤器：只允许指定 rank 的日志通过。

    典型用法：DDP 多卡训练中，``rank=local_rank``；非主进程会被全部过滤，
    避免训练输出重复。
    """

    def __init__(self, rank: int = 0):
        super().__init__()
        self.rank = rank

    def filter(self, record) -> bool:
        """仅当 ``self.rank == 0`` 时返回 ``True``。"""
        return self.rank == 0


def setup_logger(
    name: str = "xichen",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    rank: int = 0,
) -> logging.Logger:
    """统一 logger 初始化（与 ``src.utils.logger.setup_logger`` 同语义）。

    会先 ``handlers.clear()`` + ``propagate=False``，再挂 console 与
    （可选）file 两个 handler，并都附加 ``RankFilter``。

    Args:
        name: logger 名称；默认 ``"xichen"``。
        log_file: 日志文件路径，可选；为 ``None`` 时不写文件。
        level: 日志级别；默认 ``logging.INFO``。
        rank: 当前进程 rank；非 0 的进程输出会被过滤。

    Returns:
        配置好的 ``logging.Logger`` 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 清空历史 handler，避免重复 setup 时重复打印
    logger.handlers.clear()
    # 关闭向上传播，防止根 logger 也输出同一行
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

    # 控制台 handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    console.addFilter(RankFilter(rank))
    logger.addHandler(console)

    # 文件 handler (仅主进程；RankFilter 兜底过滤)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RankFilter(rank))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "xichen") -> logging.Logger:
    """获取已配置的 logger（不重新初始化）。"""
    return logging.getLogger(name)
