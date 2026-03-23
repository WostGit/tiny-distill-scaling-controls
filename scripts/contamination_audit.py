from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from metrics_utils import normalize_text, write_json


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_files", nargs="+", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output", default="outputs/metrics/contamination_audit.json")
    parser.add_argument("--ngram_n", type=int, default=4)
    args = parser.parse_args()

    train_rows = []
    for path in args.train_files:
        train_rows.extend(load_jsonl(path))
    eval_rows = load_jsonl(args.eval_file)

    train_prompts = [normalize_text(r["prompt"]) for r in train_rows]
    eval_prompts = [normalize_text(r["prompt"]) for r in eval_rows]

    train_set = set(train_prompts)
    exact_overlaps = [p for p in eval_prompts if p in train_set]

    train_ngrams = Counter()
    for p in train_prompts:
        toks = p.split()
        train_ngrams.update(ngrams(toks, args.ngram_n))

    eval_overlap_rates = []
    for p in eval_prompts:
        toks = p.split()
        e_ngrams = ngrams(toks, args.ngram_n)
        if not e_ngrams:
            eval_overlap_rates.append(0.0)
            continue
        overlap = sum(1 for ng in e_ngrams if ng in train_ngrams)
        eval_overlap_rates.append(overlap / len(e_ngrams))

    payload = {
        "train_examples": len(train_prompts),
        "eval_examples": len(eval_prompts),
        "exact_prompt_overlap_count": len(exact_overlaps),
        "exact_prompt_overlap_rate": len(exact_overlaps) / max(1, len(eval_prompts)),
        "mean_eval_ngram_overlap_rate": sum(eval_overlap_rates) / max(1, len(eval_overlap_rates)),
        "ngram_n": args.ngram_n,
        "protocol_note": (
            "Train prompts and eval prompts are generated from disjoint operand ranges to reduce direct leakage. "
            "This audit reports exact string overlap and token n-gram overlap as a lightweight contamination check."
        ),
    }
    write_json(Path(args.output), payload)


if __name__ == "__main__":
    main()
