"""Run tiny scaling study over budgets and supervision formats."""

from __future__ import annotations

import argparse
from pathlib import Path

from contamination_audit import audit
from eval_tiny_distill import eval_condition
from logging_utils import configure_logger, dump_json
from metrics_utils import load_json, summarise_condition_metrics
from train_tiny_distill import train_condition

FORMAT_TO_FILE = {
    "answer_only": "data/train_answer_only.jsonl",
    "short_rationale": "data/train_short_rationale.jsonl",
    "full_rationale": "data/train_full_rationale.jsonl",
}


def run_study(output_root: str, include_128: bool = False, base_model: str = "sshleifer/tiny-gpt2") -> None:
    logger = configure_logger()
    budgets = [16, 32, 64] + ([128] if include_128 else [])

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    contamination_json = root / "contamination_audit.json"
    audit(
        train_files=list(FORMAT_TO_FILE.values()),
        eval_file="data/tiny_eval.jsonl",
        out_json=str(contamination_json),
        n=3,
    )

    all_metrics: list[dict] = []
    for supervision_format, train_file in FORMAT_TO_FILE.items():
        for budget in budgets:
            run_name = f"{supervision_format}__n{budget}"
            run_dir = root / run_name
            train_metrics = train_condition(
                train_file=train_file,
                budget=budget,
                output_dir=str(run_dir),
                base_model=base_model,
            )
            eval_metrics_path = run_dir / "eval_metrics.json"
            eval_metrics = eval_condition(
                adapter_dir=str(run_dir / "adapter"),
                eval_file="data/tiny_eval.jsonl",
                output_json=str(eval_metrics_path),
                base_model=base_model,
            )
            all_metrics.append(
                {
                    "run_name": run_name,
                    "format": supervision_format,
                    "budget": budget,
                    "train": train_metrics,
                    "eval": eval_metrics,
                }
            )
            logger.info("Completed condition %s", run_name)

    summary = summarise_condition_metrics(all_metrics)
    summary["contamination_audit"] = load_json(contamination_json)
    dump_json(root / "summary_metrics.json", summary)

    # Small machine-readable table for easy diffing in CI artifacts.
    dump_json(root / "condition_table.json", {"conditions": summary["rows"]})

    # Also emit a tiny markdown table for human inspection.
    lines = [
        "| format | budget | accuracy | train_runtime_s | eval_runtime_s |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['format']} | {row['budget']} | {row['accuracy']:.4f} | {row['train_runtime_s']:.3f} | {row['eval_runtime_s']:.3f} |"
        )
    (root / "condition_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="outputs/metrics")
    p.add_argument("--include-128", action="store_true")
    p.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_study(output_root=args.output_root, include_128=args.include_128, base_model=args.base_model)
