# AGENTS.md

## Purpose
This repository is a tiny, readability-first distillation study. Keep implementation simple and reproducible.

## Rules for changes
- Prefer explicit scripts over abstractions.
- Keep CPU-first assumptions.
- Avoid noisy logging; keep useful progress, timing, memory, and output paths.
- Preserve machine-readable JSON outputs in `outputs/metrics/`.
- Any experiment extension should keep contamination notes updated in README.

## Validation
When changing experiment logic:
1. Run `python scripts/contamination_audit.py ...` or `python scripts/run_scaling_curve.py ...`.
2. Confirm `outputs/metrics/summary.json` and `summary_table.md` are produced.
3. Ensure README still labels this as a tiny reduced reproduction (not benchmark-grade).
