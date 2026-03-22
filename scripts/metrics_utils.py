import json
import os
from typing import Dict, List


def save_json(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_runs(run_metrics: List[Dict]) -> Dict:
    by_format = {}
    for row in run_metrics:
        fmt = row["supervision_format"]
        by_format.setdefault(fmt, []).append(row)

    format_summary = {}
    for fmt, rows in by_format.items():
        rows = sorted(rows, key=lambda r: r["budget"])
        format_summary[fmt] = {
            "budgets": [r["budget"] for r in rows],
            "exact_match": [r["exact_match"] for r in rows],
            "best_exact_match": max(r["exact_match"] for r in rows),
        }

    return {
        "num_conditions": len(run_metrics),
        "runs": run_metrics,
        "by_supervision_format": format_summary,
    }


def markdown_table(run_metrics: List[Dict]) -> str:
    rows = sorted(run_metrics, key=lambda r: (r["supervision_format"], r["budget"]))
    lines = [
        "| supervision_format | budget | exact_match | eval_examples |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['supervision_format']} | {r['budget']} | {r['exact_match']:.3f} | {r['eval_examples']} |"
        )
    return "\n".join(lines)
