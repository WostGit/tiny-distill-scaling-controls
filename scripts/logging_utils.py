"""Logging helpers for tiny distillation runs."""

from __future__ import annotations

import json
import logging
import os
import resource
import time
from pathlib import Path
from typing import Any


def configure_logger(name: str = "tiny-distill") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def now_s() -> float:
    return time.perf_counter()


def elapsed_s(start_s: float) -> float:
    return time.perf_counter() - start_s


def get_memory_snapshot_mb() -> dict[str, float]:
    """Return simple process memory stats in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is KB on Linux.
    max_rss_mb = usage.ru_maxrss / 1024.0
    return {"max_rss_mb": round(max_rss_mb, 2)}


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def log_checkpoint_event(logger: logging.Logger, out_dir: str) -> None:
    logger.info("Checkpoint saved to %s", os.path.abspath(out_dir))
