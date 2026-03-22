from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from metrics_utils import write_json


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def train_file_for_format(fmt: str) -> str:
    mapping = {
        "answer_only": "data/train_answer_only.jsonl",
        "short_rationale": "data/train_short_rationale.jsonl",
        "full_rationale": "data/train_full_rationale.jsonl",
    }
    return mapping[fmt]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["answer_only", "short_rationale", "full_rationale"],
    )
    parser.add_argument("--model_name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="outputs/metrics")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for fmt, budget in itertools.product(args.formats, args.budgets):
        run_id = f"{fmt}_n{budget}"
        run_dir = Path("outputs") / run_id
        train_metrics_path = out_dir / f"train_{run_id}.json"
        eval_metrics_path = out_dir / f"eval_{run_id}.json"
        preds_path = out_dir / f"preds_{run_id}.json"

        train_cmd = [
            sys.executable,
            "scripts/train_tiny_distill.py",
            "--train_file",
            train_file_for_format(fmt),
            "--budget",
            str(budget),
            "--supervision",
            fmt,
            "--model_name",
            args.model_name,
            "--seed",
            str(args.seed),
            "--output_dir",
            str(run_dir),
            "--metrics_path",
            str(train_metrics_path),
        ]
        eval_cmd = [
            sys.executable,
            "scripts/eval_tiny_distill.py",
            "--eval_file",
            "data/tiny_eval.jsonl",
            "--adapter_dir",
            str(run_dir / "adapter"),
            "--model_name",
            args.model_name,
            "--supervision",
            fmt,
            "--budget",
            str(budget),
            "--metrics_path",
            str(eval_metrics_path),
            "--predictions_path",
            str(preds_path),
        ]
        subprocess.run(train_cmd, check=True)
        subprocess.run(eval_cmd, check=True)

        train_metrics = load_json(train_metrics_path)
        eval_metrics = load_json(eval_metrics_path)
        summary.append(
            {
                "run_id": run_id,
                "supervision": fmt,
                "budget": budget,
                "train_loss": train_metrics["train_loss"],
                "exact_match": eval_metrics["exact_match"],
                "train_runtime_sec": train_metrics["train_runtime_sec"],
                "eval_runtime_sec": eval_metrics["eval_runtime_sec"],
            }
        )

    summary = sorted(summary, key=lambda x: (x["budget"], x["supervision"]))
    write_json(out_dir / "summary.json", {"runs": summary})

    lines = ["| supervision | budget | train_loss | exact_match |", "|---|---:|---:|---:|"]
    for row in summary:
        lines.append(
            f"| {row['supervision']} | {row['budget']} | {row['train_loss']:.4f} | {row['exact_match']:.4f} |"
        )
    (out_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
