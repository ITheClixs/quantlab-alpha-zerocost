# S0 Retrain After Persistence Lands

## When to run

After the S0 implementation lands on `quant-llm-implementation` and all unit tests are green, the operator must trigger a fresh S1 retrain so the first loadable run exists.

Before this runbook executes, `experiments/alpha_s1/` contains only `_archive_pre_s0/` (the old runs are preserved but not loadable by `load_predictor_from_run`).

## Command

On the M4 24 GB box:

```bash
cd /Users/dmr/MachineLearning
make full-retrain-s1 STREAMING=1 TRAIN_CONFIG=configs/alpha_5m.yaml TRAIN_MAX_ROWS=5000000
```

Expected wall-clock: <= 24h per CLAUDE.md section 8 budget (S1 base training).

## What to check during the run

- `top` shows the trainer's RSS staying under 20 GB (the `--streaming` flag is the load-bearing reason).
- Per-fold metrics print as each fold finishes; expect `lgb_r2`, `xgb_r2`, `cat_r2`, `mlp_r2`, and `seq_r2` all populated (the new `seq_r2` column is the post-S0 addition).
- The final holdout R2 line prints when phase 5 completes.

## Verification after completion

1. Confirm the artifact layout:

```bash
LATEST=$(ls -t experiments/alpha_s1 | grep -v _archive_pre_s0 | head -1)
ls "experiments/alpha_s1/$LATEST/models/"
```

Expected files: `ridge.joblib`, `lightgbm.txt`, `lightgbm.config.json`, `xgboost.json`, `xgboost.config.json`, `catboost.cbm`, `catboost.config.json`, `mlp.pt`, `sequence.pt`, `stacker.joblib`.

The run directory should also contain `feature_cols.json` and `_artifact_sha256.json`; `_artifact_sha256.json` covers every artifact listed in CLAUDE.md section 13.

2. Verify the loader works:

```bash
make verify-loader RUN_DIR="experiments/alpha_s1/$LATEST"
```

Expected output: `OK; n_features = <number>`.

3. Confirm the section 13 gate:

```bash
PYTHONPATH=src uv run python -c "import json, sys; m = json.load(open('experiments/alpha_s1/$LATEST/metrics.json')); print('holdout R^2 =', m['holdout_weighted_zero_mean_r2']); sys.exit(0 if m['holdout_weighted_zero_mean_r2'] >= 0.012 else 1)"
```

Expected: prints a positive R2 and exits 0.

## Holdout R2 - honest vs. zeroed

Pre-S0 runs (the archived May-19 0.0955 result) computed the holdout R2 with xgb/cat/mlp columns zeroed in the holdout stack matrix. The S0 trainer uses all 6 final base models in the holdout stack. Expect the new R2 to differ, possibly slightly lower (the stacker weights were fit on OOF predictions that included all 5 / now 6 contributions, but pre-S0 holdout evaluation only saw 2 of them).

Record the new R2 here when this runbook is first executed:

| Run id | Holdout R2 (S0, all 6 models) | Notes |
|---|---:|---|
| `20260523-160541` | `0.005489` | first post-S0 5M row-budget run; loader verified; below the `0.012` section 13 gate |

If the new R2 falls below the section 13 gate of `0.012`, do not pretend by reverting to zeroed holdout. Open a follow-up spec investigating which base model dropped, then tune in S5/S6/S8.

## Failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| `ArtifactsMissingError` from `verify-loader` | one of the 6 model files missing | check the trainer log; the model whose phase-4 fit raised will be obvious |
| MLP or sequence file is suspiciously small (<1 MB) | `state_dict()` saved before `fit()` completed | check that `MLPAlphaModel.fit` exited without an exception in the streaming run |
| OOM during phase 4 (`_refit_on_full`) | running without `STREAMING=1` on a 24 GB box | rerun with `STREAMING=1` |
| Per-fold R2 for `seq` is large-negative across all folds | sequence model needs more epochs at this row count | acceptable for initial run; tune in S5/S6/S8 |
