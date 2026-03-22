# Tiny Distill Scaling Controls (CPU-only, GitHub Actions first)

This repository is a **tiny reduced reproduction** of a reasoning-distillation scaling study that asks how student performance changes under very small data budgets and different supervision formats. It runs end-to-end on GitHub-hosted CPU runners, uses a tiny LoRA-tuned student (`sshleifer/tiny-gpt2`), and emits compact JSON artifacts so reviewers can compare conditions quickly.

## Research question

At tiny data budgets (16, 32, 64, optional 128 examples), which supervision format transfers the most useful signal to a fixed student model under contamination-aware evaluation?

## Supervision formats

- **answer-only**: target text contains only the final answer.
- **short rationale / compressed rationale**: target includes a short derivation plus final answer.
- **full rationale**: target includes a fuller chain of steps plus final answer.

All conditions keep model, optimizer, LoRA config, and epoch count fixed (1 epoch). Only budget and supervision format vary.

## Contamination controls

- Train and eval are split into separate checked-in files (`data/train_*.jsonl` vs `data/tiny_eval.jsonl`).
- Train and eval use intentionally disjoint operand ranges to reduce near-duplicate leakage.
- `scripts/contamination_audit.py` writes `outputs/metrics/contamination_audit.json` with:
  - exact prompt overlap count
  - simple 5-gram overlap statistics
  - protocol note documenting leakage-minimization strategy

## What this study proves and does not prove

**Can show:**
- Relative behavior among supervision formats in a controlled tiny setting.
- Whether small amounts of rationale-style supervision help more than answer-only at tiny scale.

**Cannot show:**
- Benchmark-grade SOTA claims.
- Generalization across tasks, domains, model families, or large data regimes.
- Causal claims beyond this small synthetic setup.

## Predicted results based on prior literature

A plausible expectation is that **short rationales** may outperform answer-only at the smallest budgets by providing extra supervision signal without overfitting to verbose reasoning style, while **full rationales** may catch up or vary depending on budget and model capacity. This repository is intended to test that qualitative pattern in a lightweight controlled setting.

## Quickstart (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_scaling_curve.py --budgets 16 32 64
```

Outputs:
- per-condition run metrics JSON: `outputs/metrics/run_<format>_n<budget>.json`
- per-condition eval JSON (+ predictions): `outputs/metrics/eval_<format>_n<budget>.json`
- aggregate summary JSON: `outputs/metrics/summary.json`
- compact markdown table: `outputs/metrics/summary_table.md`
- contamination audit JSON: `outputs/metrics/contamination_audit.json`

## GitHub Actions

Workflow: `.github/workflows/scaling-study.yml`

- Ubuntu latest, CPU only
- 1 epoch LoRA-only training per condition
- Runs >=3 budgets and 3 supervision formats
- Uploads `outputs/metrics` artifacts for review
