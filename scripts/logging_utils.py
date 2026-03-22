import json
import logging
import os
import time
from pathlib import Path

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover
    psutil = None


def configure_logger(name: str = "tiny_distill") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    return logger


def process_memory_mb() -> float:
    if psutil is None:
        return -1.0
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def timing_block(logger: logging.Logger, label: str):
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            logger.info("[%s] start", label)
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = time.time() - self.t0
            logger.info("[%s] done in %.2fs | rss=%.1fMB", label, dt, process_memory_mb())

    return _Timer()


def write_json(path: str, payload: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
