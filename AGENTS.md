# AGENTS.md

## Scope
These instructions apply to the entire repository.

## Repo intent
This repository is a **tiny reduced reproduction** for studying supervision-format effects in reasoning distillation under tiny data budgets.

## Engineering guidelines
- Keep scripts runnable on CPU-only GitHub Actions runners.
- Prioritize readable, explicit code over framework-heavy abstractions.
- Keep metrics machine-readable (`.json`) and small.
- When changing experiment logic, update `README.md` and workflow docs in the same PR.

## Experiment constraints
- Only LoRA/PEFT adaptation for training.
- One epoch per condition.
- Keep train/eval split contamination-aware.
