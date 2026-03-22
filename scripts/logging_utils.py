import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict

import resource


def setup_logger(name: str = "tiny_distill") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


@dataclass
class RunStats:
    start_time: float

    def wall_seconds(self) -> float:
        return time.time() - self.start_time

    @staticmethod
    def max_rss_mb() -> float:
        # ru_maxrss is KB on Linux
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@contextmanager
def stage_timer(logger: logging.Logger, stage_name: str):
    start = time.time()
    logger.info("[stage=%s] start", stage_name)
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info("[stage=%s] done in %.2fs", stage_name, elapsed)


def log_checkpoint_event(logger: logging.Logger, checkpoint_path: str):
    exists = os.path.exists(checkpoint_path)
    logger.info("checkpoint_saved=%s path=%s", exists, checkpoint_path)


def write_run_log(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
