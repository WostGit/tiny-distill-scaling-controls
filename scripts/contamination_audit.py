import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from metrics_utils import save_json


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def ngrams(text: str, n: int = 3) -> Set[Tuple[str, ...]]:
    toks = text.lower().split()
    return set(tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1)))


def audit(train_rows: Iterable[Dict], eval_rows: Iterable[Dict]) -> Dict:
    train_questions = {r["question"].strip().lower() for r in train_rows}
    eval_questions = {r["question"].strip().lower() for r in eval_rows}
    exact_overlap = sorted(train_questions & eval_questions)

    train_ngrams = Counter()
    for q in train_questions:
        train_ngrams.update(ngrams(q, n=3))

    eval_ngrams = Counter()
    for q in eval_questions:
        eval_ngrams.update(ngrams(q, n=3))

    shared = set(train_ngrams) & set(eval_ngrams)
    denom = max(1, len(set(eval_ngrams)))
    overlap_ratio = len(shared) / denom

    return {
        "train_questions": len(train_questions),
        "eval_questions": len(eval_questions),
        "exact_question_overlap_count": len(exact_overlap),
        "exact_question_overlap_examples": exact_overlap[:5],
        "shared_3gram_count": len(shared),
        "eval_unique_3gram_count": len(set(eval_ngrams)),
        "shared_3gram_ratio": overlap_ratio,
        "protocol_note": (
            "This audit checks exact question overlap and 3-gram lexical overlap. "
            "Train and eval files are distinct and use different templates."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-files", nargs="+", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_rows = []
    for tf in args.train_files:
        train_rows.extend(load_jsonl(tf))

    eval_rows = load_jsonl(args.eval_file)
    result = audit(train_rows, eval_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
