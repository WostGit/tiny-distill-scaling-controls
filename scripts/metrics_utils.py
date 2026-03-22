from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean


@dataclass
class RunMetrics:
    supervision_format: str
    budget: int
    train_examples: int
    eval_examples: int
    exact_match: float
    avg_train_loss: float
    train_seconds: float
    eval_seconds: float
    peak_memory_mb: float
    model_name: str
    output_dir: str

    def to_dict(self):
        return asdict(self)


def summarize_runs(runs: list[dict]) -> dict:
    grouped = {}
    for run in runs:
        grouped.setdefault(run["supervision_format"], []).append(run)

    by_format = {}
    for fmt, rows in grouped.items():
        by_format[fmt] = {
            "mean_exact_match": mean([r["exact_match"] for r in rows]),
            "best_budget": max(rows, key=lambda r: r["exact_match"])["budget"],
            "num_runs": len(rows),
        }

    table = sorted(runs, key=lambda r: (r["supervision_format"], r["budget"]))
    return {
        "num_conditions": len(runs),
        "formats": sorted(grouped.keys()),
        "budgets": sorted({r["budget"] for r in runs}),
        "by_format": by_format,
        "table": table,
    }
