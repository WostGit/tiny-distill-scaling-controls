from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from logging_utils import write_json


def load(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ngrams(text: str, n: int = 5) -> set[str]:
    toks = text.lower().split()
    return {" ".join(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}


def run_audit(train_files: list[str], eval_file: str, out_path: str):
    train_rows = []
    for file in train_files:
        train_rows.extend(load(file))
    eval_rows = load(eval_file)

    train_prompts = [r["prompt"].strip() for r in train_rows]
    eval_prompts = [r["prompt"].strip() for r in eval_rows]

    exact_overlap = len(set(train_prompts) & set(eval_prompts))
    train_ngrams = Counter()
    for p in train_prompts:
        train_ngrams.update(ngrams(p, n=5))

    max_ngram_overlap = 0
    avg_ngram_overlap = 0.0
    for p in eval_prompts:
        grams = ngrams(p, n=5)
        overlap = sum(1 for g in grams if g in train_ngrams)
        max_ngram_overlap = max(max_ngram_overlap, overlap)
        avg_ngram_overlap += overlap
    avg_ngram_overlap /= max(1, len(eval_prompts))

    payload = {
        "train_files": train_files,
        "eval_file": eval_file,
        "train_count": len(train_prompts),
        "eval_count": len(eval_prompts),
        "exact_prompt_overlap": exact_overlap,
        "max_5gram_overlap_per_eval_item": max_ngram_overlap,
        "avg_5gram_overlap_per_eval_item": avg_ngram_overlap,
        "protocol_note": "Train/eval operand ranges are intentionally disjoint to reduce near-duplicate leakage.",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-files", nargs="+", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--out-path", required=True)
    args = parser.parse_args()
    run_audit(args.train_files, args.eval_file, args.out_path)


if __name__ == "__main__":
    main()
