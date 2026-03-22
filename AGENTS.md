# AGENTS.md

## Purpose
This repository is a tiny, CPU-only reasoning distillation study designed for reproducible GitHub Actions runs.

## Working rules
- Keep the study explicitly tiny and reduced; do not present results as benchmark-grade.
- Preserve contamination-aware design: train/eval split files must remain separate.
- Keep scripts readable and straightforward over heavy abstraction.
- Prefer machine-readable JSON outputs in `outputs/metrics/`.

## Runbook
- Local run: `python scripts/run_scaling_curve.py --output-root outputs/metrics`
- Optional extra point: add `--include-128`
- CI entrypoint: `.github/workflows/scaling-study.yml`
