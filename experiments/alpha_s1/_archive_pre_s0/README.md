# Archived pre-S0 alpha_s1 runs

These seven training runs were produced by the pre-S0 trainers
(`scripts/alpha_train_s1.py` and `scripts/alpha_train_s1_streaming.py`) before
the S0 milestone landed.

Each run directory contains the original artifacts but **only**
`models/stacker.joblib` — the base models (ridge / lgb / xgb / cat / mlp /
sequence) were never persisted by the old trainers, so these runs are not
loadable by the post-S0 `quant_research_stack.alpha.inference.load_predictor_from_run`.

They are preserved here for metric-history reference only. Do not point any
post-S0 tooling at them.

## Runs

| Run id | Trainer | Holdout R² (zero'd-3-of-5 stack) |
|---|---|---|
| 20260517-173937 | alpha_train_s1.py | n/a (early dev) |
| 20260517-211119 | alpha_train_s1.py | 0.59 (synthetic-only) |
| 20260518-102255 | alpha_train_s1_streaming.py | ~0 (leak-bug run; pre-7121a3f) |
| 20260519-074922 | alpha_train_s1_streaming.py | n/a |
| 20260519-202114 | alpha_train_s1_streaming.py | **0.0955** (leak-fixed milestone gate) |
| 20260520-225145 | alpha_train_s1_streaming.py | n/a (post-leak-fix retest) |
| 20260521-143039 | alpha_train_s1_streaming.py | n/a (post-leak-fix retest) |

The first post-S0 run (with all six base models on disk and an honest holdout)
will live at `experiments/alpha_s1/<new-run-id>/` and is produced by
`scripts/train_s1.py` per the S0 spec.

## Why this README is tracked

`experiments/` is in `.gitignore`, so the run directories themselves cannot
be committed. This README is force-added (`git add -f`) so the archival
**decision** is in git history even though the artifacts aren't.
