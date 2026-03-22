# Tiny Distill Scaling Controls

This repository is a **tiny reduced reproduction** of a reasoning-distillation scaling study. It measures how a small student model responds to tiny supervision budgets across answer-only, short-rationale, and full-rationale training formats, while using a contamination-aware train/eval split and lightweight overlap auditing. It is designed to run end-to-end on GitHub-hosted CPU runners and is explicitly **not** a benchmark-grade reproduction.

## Research question
At tiny data budgets (16/32/64 examples), which supervision format transfers the most useful signal to a fixed tiny student under contamination-aware evaluation?

## Supervision formats
- **answer-only**: target contains only `Answer: <value>`.
- **short rationale**: target includes a compressed one-line explanation and final answer.
- **full rationale**: target includes a longer explanation and final answer.

Only supervision format and data budget change between conditions. The student model, optimizer, LoRA config, epoch count, and eval set are held fixed.

## Contamination controls
- Train and eval are checked into separate files in `data/`.
- Eval problems are generated from disjoint operand ranges versus training data.
- `scripts/contamination_audit.py` performs:
  - exact normalized prompt overlap check
  - token n-gram overlap rate check
- Audit output is written to `outputs/metrics/contamination_audit.json`.

> Note: This is a lightweight contamination protocol for a toy setup, not a full forensic contamination analysis.

## What this study proves and does not prove
**What it can show**
- Relative behavior of three supervision formats in a tightly controlled tiny-data toy setting.
- Whether rationale density appears helpful under strict low-budget constraints.

**What it does not show**
- General model capability across realistic tasks.
- Production-grade conclusions about chain-of-thought utility.
- State-of-the-art performance comparisons.

## Predicted results based on prior literature
Based on prior distillation and rationale-supervision literature, a plausible expectation is:
1. **short rationale** often gives the best tradeoff at very small budgets (enough intermediate signal without excessive verbosity).
2. **answer-only** may underperform due to weak supervision density.
3. **full rationale** can help but may be noisier or harder to compress for very tiny students.

These are hypotheses; this repository is intended to test them in a reduced setting.

## Repository layout
```
README.md
AGENTS.md
requirements.txt
data/
  train_answer_only.jsonl
  train_short_rationale.jsonl
  train_full_rationale.jsonl
  tiny_eval.jsonl
scripts/
  train_tiny_distill.py
  eval_tiny_distill.py
  run_scaling_curve.py
  contamination_audit.py
  logging_utils.py
  metrics_utils.py
outputs/metrics/.gitkeep
.github/workflows/scaling-study.yml
```

## Running locally
```bash
python -m pip install -r requirements.txt
python scripts/contamination_audit.py \
  --train_files data/train_answer_only.jsonl data/train_short_rationale.jsonl data/train_full_rationale.jsonl \
  --eval_file data/tiny_eval.jsonl \
  --output outputs/metrics/contamination_audit.json
python scripts/run_scaling_curve.py --budgets 16 32 64 --formats answer_only short_rationale full_rationale
```

## Outputs
Per condition, the pipeline writes:
- train metrics JSON: `outputs/metrics/train_<format>_n<budget>.json`
- eval metrics JSON: `outputs/metrics/eval_<format>_n<budget>.json`
- predictions JSON: `outputs/metrics/preds_<format>_n<budget>.json`

Aggregate outputs:
- `outputs/metrics/summary.json`
- `outputs/metrics/summary_table.md`
- `outputs/metrics/contamination_audit.json`
