# AGENTS.md

## Repository intent
This repository is a tiny reduced reproduction for reasoning-distillation scaling under tiny data budgets. Keep changes simple, readable, and reproducible on GitHub-hosted CPU runners.

## Implementation rules
- Use CPU-only defaults.
- Keep training at 1 epoch unless a task explicitly changes the protocol.
- Prefer compact JSON outputs in `outputs/metrics/`.
- Avoid adding heavyweight dependencies without a clear need.

## Logging and outputs
- Preserve concise but informative logs for timing, memory, and checkpoints.
- Keep artifacts deterministic where practical (seeded operations).
