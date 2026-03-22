import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from metrics_utils import load_json, markdown_table, save_json, summarize_runs

FORMAT_TO_FILE = {
    "answer_only": "data/train_answer_only.jsonl",
    "short_rationale": "data/train_short_rationale.jsonl",
    "full_rationale": "data/train_full_rationale.jsonl",
}


def run_cmd(cmd):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    p.add_argument("--budgets", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--include-128", action="store_true")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--metrics-dir", default="outputs/metrics")
    args = p.parse_args()

    budgets = list(args.budgets)
    if args.include_128 and 128 not in budgets:
        budgets.append(128)

    os.makedirs(args.outputs_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    audit_path = f"{args.metrics_dir}/contamination_audit.json"
    run_cmd(
        [
            sys.executable,
            "scripts/contamination_audit.py",
            "--train-files",
            *FORMAT_TO_FILE.values(),
            "--eval-file",
            "data/tiny_eval.jsonl",
            "--output",
            audit_path,
        ]
    )

    all_metrics = []
    for fmt, train_file in FORMAT_TO_FILE.items():
        for budget in sorted(budgets):
            run_name = f"{fmt}_n{budget}"
            adapter_dir = f"{args.outputs_dir}/checkpoints/{run_name}"
            train_metrics = f"{args.metrics_dir}/{run_name}_train.json"
            eval_metrics = f"{args.metrics_dir}/{run_name}_eval.json"

            run_cmd(
                [
                    sys.executable,
                    "scripts/train_tiny_distill.py",
                    "--model-name",
                    args.model_name,
                    "--train-file",
                    train_file,
                    "--output-dir",
                    adapter_dir,
                    "--metrics-file",
                    train_metrics,
                    "--supervision-format",
                    fmt,
                    "--budget",
                    str(budget),
                ]
            )
            run_cmd(
                [
                    sys.executable,
                    "scripts/eval_tiny_distill.py",
                    "--model-name",
                    args.model_name,
                    "--adapter-dir",
                    adapter_dir,
                    "--eval-file",
                    "data/tiny_eval.jsonl",
                    "--metrics-file",
                    eval_metrics,
                    "--supervision-format",
                    fmt,
                    "--budget",
                    str(budget),
                ]
            )

            merged = load_json(eval_metrics)
            merged["train"] = load_json(train_metrics)
            all_metrics.append(merged)

    summary = summarize_runs(all_metrics)
    summary_path = f"{args.metrics_dir}/summary.json"
    save_json(summary_path, summary)

    table = markdown_table(all_metrics)
    Path(f"{args.metrics_dir}/summary_table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
