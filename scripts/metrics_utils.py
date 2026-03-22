"""Metrics utilities for evaluation and aggregation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

NUMBER_PATTERN = re.compile(r"-?\d+")


def extract_first_number(text: str) -> str:
    match = NUMBER_PATTERN.search(text.strip())
    return match.group(0) if match else text.strip()


def exact_match_accuracy(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    correct = 0
    for pred, ref in zip(predictions, references):
        if extract_first_number(pred) == extract_first_number(ref):
            correct += 1
    return correct / len(references)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def summarise_condition_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda x: (x["format"], x["budget"])):
        rows.append(
            {
                "format": item["format"],
                "budget": item["budget"],
                "accuracy": item["eval"]["accuracy"],
                "train_runtime_s": item["train"]["runtime_s"],
                "eval_runtime_s": item["eval"]["runtime_s"],
            }
        )

    by_format: dict[str, list[float]] = {}
    for row in rows:
        by_format.setdefault(row["format"], []).append(row["accuracy"])

    avg_by_format = {fmt: round(mean(vals), 4) for fmt, vals in by_format.items()}
    return {"rows": rows, "avg_accuracy_by_format": avg_by_format}
