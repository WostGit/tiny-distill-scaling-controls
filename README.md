# Tiny Distill Scaling Controls

This repository is a **self-contained, tiny reduced reproduction** of a reasoning-distillation scaling study. It runs a fixed LoRA student on very small training budgets and compares three supervision formats (answer-only, short rationale, full rationale) under a contamination-aware train/eval split. It is designed for CPU-only GitHub-hosted runners and intentionally prioritizes clarity and reproducibility over benchmark-level rigor.

## Research question
At tiny data budgets (16, 32, 64; optional 128), which supervision format transfers the most useful signal to the student under contamination-aware evaluation?

## Supervision formats
- **answer_only**: target contains only `Final answer: ...`
- **short_rationale**: target contains compressed reasoning plus final answer
- **full_rationale**: target contains a longer rationale plus final answer

All other training knobs stay fixed across conditions: model, optimizer, LoRA configuration, epoch count, and evaluation data.

## Contamination controls
- Distinct checked-in files for train and eval:
  - `data/train_answer_only.jsonl`
  - `data/train_short_rationale.jsonl`
  - `data/train_full_rationale.jsonl`
  - `data/tiny_eval.jsonl`
- `scripts/contamination_audit.py` runs before training and writes `outputs/metrics/contamination_audit.json`.
- Audit reports exact question overlap and shared 3-gram overlap ratio to make leakage risk explicit.

## What this study proves and does not prove
### Proves (within this tiny setup)
- End-to-end automation over a small scaling grid in GitHub Actions.
- Relative behavior of supervision formats under severe low-data constraints for this toy task family.
- Machine-readable per-condition metrics and aggregate summaries.

### Does not prove
- Generalization to large models, diverse reasoning tasks, or benchmark-grade protocols.
- Causal claims beyond this synthetic dataset and one fixed student/training setup.
- Production-quality contamination guarantees.

## Predicted results based on prior literature
A common pattern in distillation literature is that **compressed/short rationales** can be a strong middle ground at small budgets: they provide extra learning signal beyond answer-only while avoiding some verbosity/noise that can appear in long chains. In this tiny synthetic setup, a plausible expectation is:
1. short_rationale best or tied-best,
2. full_rationale competitive but less stable,
3. answer_only strongest at very low compute/noise settings only in some cases.

Treat these as hypotheses, not claims.

## Repository layout

```text
.
├── .github/workflows/scaling-study.yml
├── AGENTS.md
├── README.md
├── requirements.txt
├── data/
│   ├── train_answer_only.jsonl
│   ├── train_short_rationale.jsonl
│   ├── train_full_rationale.jsonl
│   └── tiny_eval.jsonl
├── scripts/
│   ├── contamination_audit.py
│   ├── eval_tiny_distill.py
│   ├── logging_utils.py
│   ├── metrics_utils.py
│   ├── run_scaling_curve.py
│   └── train_tiny_distill.py
└── outputs/metrics/.gitkeep
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_scaling_curve.py
```

Optional 128-point run:

```bash
python scripts/run_scaling_curve.py --include-128
```

## Outputs
Per condition:
- `outputs/metrics/<format>_n<budget>_train.json`
- `outputs/metrics/<format>_n<budget>_eval.json`

Aggregate:
- `outputs/metrics/summary.json`
- `outputs/metrics/summary_table.md`
- `outputs/metrics/contamination_audit.json`

## GitHub Actions
Workflow: `.github/workflows/scaling-study.yml`
- installs dependencies,
- runs the full tiny scaling sweep,
- uploads `outputs/metrics/` artifact.
