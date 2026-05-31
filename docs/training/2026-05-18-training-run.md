# QuantLab Training Run - 2026-05-18

## Scope

This run attempted every training path that is currently executable on the local Mac:

- S1 bounded Jane Street ensemble training plus reduced Optuna search.
- S2 governor retrieval-index rebuild.
- S2 full LoRA-dataset regeneration.
- S2 bounded Qwen 0.5B LoRA adapter training.
- S2 fast governor smoke.

The full S1 47M-row profile and the full S2 LoRA profile are documented as blocked until
streaming and profile changes are implemented. They were not re-run in full because prior
attempts exposed deterministic resource blockers: S1 full materialization was killed by the
OS, and the original S2 LoRA profile projected far beyond the configured 8-hour budget.

## S1 Bounded Training

Command:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 make full-retrain-s1 \
  EXTRACT="scripts/alpha_extract_meta_features.py --model-dir models/huggingface/hasnain43__bert-stock-sentiment-v1" \
  TRAIN="scripts/train_s1.py" TRAIN_CONFIG="configs/alpha_fast.yaml" TRAIN_MAX_ROWS=100000 \
  OPTUNA="scripts/alpha_optuna_search.py --config configs/alpha_fast.yaml --max-rows 100000" \
  OPTUNA_ARGS="--n-trials 5"
```

Result:

- Run id: `20260517-211119`
- Training rows: `88296`
- Permanent holdout rows: `11704`
- Adversarial filter: `355 / 485` features kept
- Noise-floor filter: `225 / 355` features kept
- Holdout weighted zero-mean R2: `0.5908574892824593`

Fold metrics:

| Fold | Ridge R2 | LightGBM R2 | XGBoost R2 | CatBoost R2 | MLP R2 |
| ---: | -------: | ----------: | ---------: | ----------: | -----: |
| 0 | 0.338392 | 0.660113 | 0.658038 | 0.629743 | 0.431995 |
| 1 | 0.419535 | 0.636962 | 0.628333 | 0.586658 | 0.360816 |
| 2 | 0.522529 | 0.713879 | 0.712305 | 0.695082 | 0.517555 |

## S1 Optuna Search

Command:

```bash
PYTHONPATH=src uv run python scripts/alpha_optuna_search.py \
  --config configs/alpha_fast.yaml \
  --max-rows 100000 \
  --n-trials 5
```

Result:

- Best CV R2: `0.6668862387130859`
- Trials: `5`
- Best parameters:
  - `num_leaves`: `52`
  - `max_depth`: `-1`
  - `learning_rate`: `0.05399484409787434`
  - `feature_fraction`: `0.8005575058716043`
  - `bagging_fraction`: `0.8540362888980227`

## S2 Retrieval Indexes

Command:

```bash
make governor-build-indexes
```

Result:

- Corpus chunks: `1581`
- Corpus SHA256: `e13dbe1ebe79a7c7db382d0ec2eb12c3cbf1729139f064ecfc82ec7ad33e628f`
- Embedding model: `FinLang/finance-embeddings-investopedia`
- Vector dimension: `768`
- Index artifacts:
  - `models/governor/bm25_index.pkl`
  - `models/governor/dense_index.npy`
  - `models/governor/dense_index.faiss`
  - `models/governor/index_metadata.json`

## S2 LoRA Dataset

Commands:

```bash
make governor-lora-dataset
PYTHONPATH=src uv run python scripts/governor_lora_dataset.py \
  --limit 128 \
  --out-jsonl data/processed/research/lora_governor_fast.jsonl
```

Result:

- Full LoRA records: `4742`
- Fast LoRA records: `128`

## S2 Bounded LoRA Training

Command:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src uv run python scripts/governor_train_lora.py \
  --config configs/governor_fast.yaml \
  --dataset-jsonl data/processed/research/lora_governor_fast.jsonl
```

Result:

- Run id: `20260517-211722`
- Base model: `models/huggingface/Qwen__Qwen2.5-0.5B-Instruct`
- Train records: `116`
- Eval records: `12`
- Training steps: `20`
- Elapsed hours: `0.003585040834214952`
- Eval loss recorded as `held_out_perplexity`: `4.100590705871582`

## S2 Smoke

Command:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src uv run python scripts/s2_smoke.py \
  --config configs/governor_fast.yaml
```

Result:

- Smoke result: `rc=0`
- Smoke output is isolated to a temporary transport directory so prior read-only daily
  verdict files cannot break repeat runs.

## Blockers For Full Profiles

S1 full profile:

- The full Jane Street mirror contains about 47M rows.
- The current implementation materializes feature matrices in memory.
- Prior full-profile training was killed by the OS with `Error 137`.
- Required fix: implement the streaming/cache-capped S1 training path specified in the
  platform design before reattempting the full dataset.

S2 full LoRA profile:

- The original Qwen 0.5B LoRA profile starts a 1335-step run.
- First observed step time was about 203 seconds, projecting to roughly 75 hours.
- Required fix: keep the bounded MPS profile for local iteration, and redesign the full
  profile around shorter sequence lengths, fewer records, better collation, MLX/Metal
  tuning, or a remote GPU.

## Verification

The following checks passed after the code changes:

```bash
PYTHONPATH=src uv run pytest -q
uv run ruff check src scripts tests
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src uv run python scripts/s2_smoke.py --config configs/governor_fast.yaml
```
