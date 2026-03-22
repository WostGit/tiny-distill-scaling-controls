# Tiny Distill Scaling Controls

This repository is a **self-contained tiny reduced reproduction** of a reasoning-distillation scaling study: we fine-tune the same very small LoRA student on tiny data budgets (16/32/64, optional 128) and compare three supervision formats (answer-only, short rationale, full rationale) under a contamination-aware train/eval protocol on CPU-only GitHub-hosted runners.

## Research question
At tiny data budgets, which supervision format transfers the most useful signal to the student under contamination-aware evaluation?

## Supervision formats
- **answer-only**: target is just the final answer.
- **short rationale / compressed rationale**: target includes a compact explanation plus final answer.
- **full rationale**: target includes a longer step-by-step explanation plus final answer.

The training model, optimizer, LoRA settings, epoch count, and evaluation procedure are fixed across conditions. Only supervision format and budget vary.

## Contamination controls
- Train and eval are in **separate checked-in files** under `data/`.
- We include an explicit contamination audit script: `scripts/contamination_audit.py`.
- The audit reports:
  - exact train/eval question-string overlap count
  - n-gram overlap statistics
  - a protocol note describing split hygiene
- The scaling runner writes `outputs/metrics/contamination_audit.json` for reviewer inspection.

## What this study proves and does not prove
### It can show
- Whether one supervision format seems to transfer better signal than others in this tiny setup.
- How sensitive the tiny student is to data budget changes from 16 to 64 (or 128) examples.
- Reproducible end-to-end experiment plumbing suitable for CI.

### It does not show
- Benchmark-grade performance or broad claims about reasoning generally.
- Generalization across model families, domains, or large-scale datasets.
- Robust causal conclusions beyond this constrained synthetic setting.

## Predicted results based on prior literature
A plausible expectation is:
1. **Short rationale** often helps most at tiny budgets by adding structured signal while avoiding excess verbosity.
2. **Answer-only** can be competitive when tasks are simple but may plateau sooner.
3. **Full rationale** may help or hurt depending on whether added verbosity introduces noise for such a tiny student and tiny data.

These are hypotheses, not guaranteed outcomes.

## Repository layout
```text
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
python scripts/run_scaling_curve.py --output-root outputs/metrics
# optional fourth scaling point
python scripts/run_scaling_curve.py --output-root outputs/metrics --include-128
```

## Output artifacts
Each condition writes:
- `outputs/metrics/<format>__n<budget>/train_metrics.json`
- `outputs/metrics/<format>__n<budget>/eval_metrics.json`

Aggregate outputs:
- `outputs/metrics/summary_metrics.json`
- `outputs/metrics/condition_table.json`
- `outputs/metrics/condition_table.md`
- `outputs/metrics/contamination_audit.json`

## GitHub Actions
Workflow: `.github/workflows/scaling-study.yml`
- Runs on `ubuntu-latest` CPU runner.
- Installs dependencies.
- Executes full scaling study.
- Uploads JSON/Markdown metrics artifacts.
