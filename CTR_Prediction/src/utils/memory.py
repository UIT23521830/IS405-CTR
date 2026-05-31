from __future__ import annotations

import os
from typing import Any

import psutil


def memory_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)


def log_memory_usage(logger: Any, prefix: str = "memory") -> None:
    logger.info("%s_rss_mb=%.2f", prefix, memory_usage_mb())


def log_gpu_usage(logger: Any, prefix: str = "gpu") -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("%s_cuda_available=false", prefix)
            return
        for idx in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(idx) / (1024**2)
            reserved = torch.cuda.memory_reserved(idx) / (1024**2)
            logger.info(
                "%s_%d_allocated_mb=%.2f reserved_mb=%.2f",
                prefix,
                idx,
                allocated,
                reserved,
            )
    except Exception as exc:
        logger.warning("Could not log GPU usage: %s", exc)

