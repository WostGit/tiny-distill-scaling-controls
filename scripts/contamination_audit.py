"""Simple contamination audit between train/eval questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from logging_utils import dump_json


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ngrams(text: str, n: int) -> set[str]:
    tokens = text.lower().split()
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def audit(train_files: Iterable[str], eval_file: str, out_json: str, n: int = 3) -> dict:
    train_qs: list[str] = []
    for f in train_files:
        train_qs.extend([r["question"].strip() for r in read_jsonl(f)])
    eval_qs = [r["question"].strip() for r in read_jsonl(eval_file)]

    train_set = set(train_qs)
    exact_overlap = [q for q in eval_qs if q in train_set]

    train_ngrams = set()
    for q in train_qs:
        train_ngrams |= ngrams(q, n)

    overlap_counts = []
    for q in eval_qs:
        ng = ngrams(q, n)
        frac = (len(ng & train_ngrams) / len(ng)) if ng else 0.0
        overlap_counts.append(frac)

    payload = {
        "protocol_note": "Train and eval are separate checked-in files with different prompt templates and disjoint IDs.",
        "train_files": list(train_files),
        "eval_file": eval_file,
        "exact_question_overlap_count": len(exact_overlap),
        "exact_question_overlap_examples": exact_overlap[:5],
        "mean_ngram_overlap_frac": round(sum(overlap_counts) / max(len(overlap_counts), 1), 4),
        "max_ngram_overlap_frac": round(max(overlap_counts) if overlap_counts else 0.0, 4),
        "ngram_n": n,
    }
    dump_json(out_json, payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-files", nargs="+", required=True)
    p.add_argument("--eval-file", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--ngram-n", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    audit(args.train_files, args.eval_file, args.out_json, args.ngram_n)
