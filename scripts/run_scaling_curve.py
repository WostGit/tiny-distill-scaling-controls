from __future__ import annotations

import argparse
import json
from pathlib import Path

from contamination_audit import run_audit
from eval_tiny_distill import evaluate_condition
from logging_utils import configure_logger, write_json
from metrics_utils import RunMetrics, summarize_runs
from train_tiny_distill import train_condition

LOGGER = configure_logger()

FORMAT_TO_FILE = {
    "answer_only": "data/train_answer_only.jsonl",
    "short_rationale": "data/train_short_rationale.jsonl",
    "full_rationale": "data/train_full_rationale.jsonl",
}


def write_markdown_table(rows: list[dict], out_path: str):
    lines = [
        "| supervision_format | budget | exact_match | avg_train_loss | train_s | eval_s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["supervision_format"], x["budget"])):
        lines.append(
            f"| {r['supervision_format']} | {r['budget']} | {r['exact_match']:.4f} | {r['avg_train_loss']:.4f} | {r['train_seconds']:.2f} | {r['eval_seconds']:.2f} |"
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--formats", nargs="+", default=["answer_only", "short_rationale", "full_rationale"])
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    args = parser.parse_args()

    metrics_dir = Path(args.output_root) / "metrics"
    run_dir = Path(args.output_root) / "runs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_audit(
        train_files=[FORMAT_TO_FILE[fmt] for fmt in args.formats],
        eval_file="data/tiny_eval.jsonl",
        out_path=str(metrics_dir / "contamination_audit.json"),
    )

    all_runs = []
    for fmt in args.formats:
        for budget in args.budgets:
            condition = f"{fmt}_n{budget}"
            cond_dir = run_dir / condition
            LOGGER.info("running condition=%s", condition)

            train_out = train_condition(
                train_file=FORMAT_TO_FILE[fmt],
                budget=budget,
                output_dir=str(cond_dir),
                model_name=args.model_name,
            )
            eval_path = metrics_dir / f"eval_{condition}.json"
            eval_metrics = evaluate_condition(
                adapter_dir=train_out.adapter_dir,
                eval_file="data/tiny_eval.jsonl",
                output_path=str(eval_path),
                model_name=args.model_name,
            )

            run_metrics = RunMetrics(
                supervision_format=fmt,
                budget=budget,
                train_examples=train_out.train_examples,
                eval_examples=eval_metrics["eval_examples"],
                exact_match=eval_metrics["exact_match"],
                avg_train_loss=train_out.avg_train_loss,
                train_seconds=train_out.train_seconds,
                eval_seconds=eval_metrics["eval_seconds"],
                peak_memory_mb=train_out.peak_memory_mb,
                model_name=args.model_name,
                output_dir=str(cond_dir),
            ).to_dict()
            write_json(str(metrics_dir / f"run_{condition}.json"), run_metrics)
            all_runs.append(run_metrics)

    summary = summarize_runs(all_runs)
    write_json(str(metrics_dir / "summary.json"), summary)
    write_markdown_table(all_runs, str(metrics_dir / "summary_table.md"))
    LOGGER.info("finished %d conditions", len(all_runs))


if __name__ == "__main__":
    main()
