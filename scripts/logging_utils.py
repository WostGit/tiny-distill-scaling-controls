"""Lightweight logging helpers for tiny CPU experiments."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Iterator

import psutil


def configure_logging(name: str = "tiny_distill") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def memory_snapshot_mb() -> dict[str, float]:
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    vms_mb = process.memory_info().vms / (1024 * 1024)
    return {"rss_mb": round(rss_mb, 2), "vms_mb": round(vms_mb, 2)}


@contextmanager
def timed_stage(logger: logging.Logger, stage_name: str) -> Iterator[None]:
    start = time.perf_counter()
    logger.info("stage_start=%s", stage_name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("stage_end=%s elapsed_sec=%.3f", stage_name, elapsed)
