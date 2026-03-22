"""Metrics utilities for tiny distillation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_answer(text: str) -> str:
    candidate = text.strip()
    if "Answer:" in candidate:
        candidate = candidate.split("Answer:")[-1].strip()
    if "\n" in candidate:
        candidate = candidate.splitlines()[0].strip()
    return normalize_text(candidate)


def exact_match(predictions: Iterable[str], references: Iterable[str]) -> float:
    pairs = list(zip(predictions, references))
    if not pairs:
        return 0.0
    score = sum(int(extract_answer(p) == extract_answer(r)) for p, r in pairs)
    return score / len(pairs)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
