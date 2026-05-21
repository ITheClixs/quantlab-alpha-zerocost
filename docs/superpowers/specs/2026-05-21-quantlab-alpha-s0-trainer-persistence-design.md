# QuantLab Alpha — S0: Trainer Base-Model Persistence (Design)

**Date:** 2026-05-21
**Status:** approved (brainstorming complete; awaiting spec review)
**Predecessor:** S1 milestone (CLAUDE.md §13 holdout gate cleared at 0.0955 weighted-zero-mean R²)
**Successor:** S5 — multi-universe equity backtests (NASDAQ + S&P + NYSE daily)
**Track:** sequential prerequisite for S5, S6, S8, and S4.1β

---

## 0. Goal

Make S1 base models loadable from disk so live/paper-trade serving (S4.1β+) and multi-universe retrains (S5/S6/S8) have something to load. Close the CLAUDE.md §13 required-artifact gap. Unify the two diverged trainers (`alpha_train_s1.py`, `alpha_train_s1_streaming.py`) into one testable training module. Persist `feature_cols.json` + sha256 so the loader can refuse callers with mismatched feature schemas.

## 1. Scope

### In scope

- New `src/quant_research_stack/alpha/training.py` — pure `train_s1(config: TrainConfig, registry: RunRegistry) -> RunResult` function.
- New `scripts/train_s1.py` thin CLI (~40 LOC) replacing both existing trainers.
- `save()` / `@classmethod load()` methods on `RidgeAlphaModel`, `LightGBMAlphaModel`, `XGBoostAlphaModel`, `CatBoostAlphaModel`, `MLPAlphaModel`, `Conv1DAlphaModel`, `LinearStacker`.
- Fit `Conv1DAlphaModel` (sequence) in the per-fold loop + holdout refit. Stacker grows from 5 → 6 base-model inputs (`feature_order: ["ridge","lgb","xgb","cat","mlp","seq"]`).
- `inference.load_predictor_from_run(run_dir: Path) -> S1Predictor`.
- `feature_cols.json` written next to models; sha256 indexed into `_artifact_sha256.json`.
- Archive `experiments/alpha_s1/*` (5 pre-S0 runs) → `experiments/alpha_s1/_archive_pre_s0/`.
- Final acceptance: a fresh `make full-retrain-s1` produces a May-21 run loadable by the new API.
- New runbook `docs/runbooks/s0_retrain_after_persistence.md`.
- `CLAUDE.md` §13 artifact list updated to match what `_persist_run` writes.

### Out of scope

- Any change to feature engineering (`features.py`), CV (`cv.py`), or metric definitions (`metrics.py`).
- Hyperparameter changes (Optuna re-search, learning-rate tweaks, etc.).
- Multi-universe retraining (deferred to S5/S6/S8).
- Live-serving daemon `s1_serve` (deferred to S4.1β; the May-20 brainstorm for it is set aside).
- Changes to `configs/promotion.yaml` (CLAUDE.md §1.13 — two-person review required).

## 2. Technical context

### 2.1 Current trainer behavior (May-21, pre-S0)

Two trainers exist, both writing only `stacker.joblib` (a dict of weights, not even a sklearn estimator):

- `scripts/alpha_train_s1.py` (207 LOC) — the one `make full-retrain-s1` invokes today.
- `scripts/alpha_train_s1_streaming.py` (435 LOC) — the trainer that actually produced the May-19 0.0955 holdout R².

Both fit five base models (`ridge`, `lgb`, `xgb`, `cat`, `mlp`) per fold for OOF predictions, then **deliberately discard them at end of fold**. `alpha_train_s1_streaming.py:421` reads:

> `# Persist the linear stacker (the only model still in scope here — base learners were [discarded after holdout refit])`

The holdout-refit phase fits `ridge` and `lgb` on full train + computes their holdout predictions, then **zeros the `xgb`/`cat`/`mlp` columns** in the holdout stack matrix before passing it to the stacker. The reported holdout R² is therefore not what the stacker was trained to produce.

### 2.2 The S1 model classes today

Five classes exist with only `fit()` + `predict()` methods — no serialization API:

- `RidgeAlphaModel` (sklearn Ridge inside)
- `LightGBMAlphaModel` (`lgb.Booster` inside)
- `XGBoostAlphaModel` (`xgb.Booster` inside)
- `CatBoostAlphaModel` (`CatBoostRegressor` inside)
- `MLPAlphaModel` (PyTorch `nn.Module` inside, plus an in-memory `StandardScaler`)

A sixth class, `Conv1DAlphaModel`, exists in `src/quant_research_stack/alpha/models/sequence.py` but is never imported by either trainer. CLAUDE.md §13 lists `sequence.pt` as a required milestone artifact.

### 2.3 The inference module today

`src/quant_research_stack/alpha/inference.py` exposes a `_StackPredictor` dataclass and a constructor `build_predictor_from_stack(base_funcs, stacker_weights, feature_cols)` that takes already-loaded base-model callables. There is no `load_predictor_from_run(run_dir)` — nothing converts on-disk artifacts back into an `S1Predictor`.

### 2.4 What this means for live serving

Until S0 lands, any "live S1 inference" path is impossible without first replaying pre-computed predictions from `predictions.parquet`. The May-20 `s1_serve` brainstorm hit this wall and was set aside in favor of S0.

## 3. Architecture

### 3.1 Before / after artifact layout

**Before (current state):**

```text
experiments/alpha_s1/<run_id>/
  metrics.json
  predictions.parquet            (OOF + holdout, 9 columns)
  feature_importance.parquet
  cv_folds.json
  metadata.json
  _artifact_sha256.json
  models/stacker.joblib           ← only this; dict of {weights, feature_order}
```

**After (S0):**

```text
experiments/alpha_s1/<run_id>/
  metrics.json                    unchanged
  predictions.parquet             unchanged
  feature_importance.parquet      unchanged
  cv_folds.json                   unchanged
  metadata.json                   unchanged
  feature_cols.json               NEW   ordered column names + sha256
  _artifact_sha256.json           extended (8 new entries)
  models/
    ridge.joblib                  NEW
    lightgbm.txt                  NEW   (LGB Booster native, text)
    lightgbm.config.json          NEW   (wrapper config sidecar)
    xgboost.json                  NEW   (XGB Booster native, JSON)
    xgboost.config.json           NEW
    catboost.cbm                  NEW   (CatBoost native, binary)
    catboost.config.json          NEW
    mlp.pt                        NEW   (state_dict + arch + scaler + config inside)
    sequence.pt                   NEW   (same pattern as mlp.pt)
    stacker.joblib                CHANGED (now 6-element feature_order)
```

### 3.2 Module layout

```text
src/quant_research_stack/alpha/
  training.py                    NEW  ~400 LOC
    @dataclass(frozen=True) TrainConfig          # Pydantic loader for configs/alpha.yaml
    @dataclass(frozen=True) RunResult            # paths + metrics + run_id
    def train_s1(config, registry) -> RunResult
    def _fit_one_fold(...)                       # private
    def _refit_on_full(...)                      # private
    def _persist_run(...)                        # private

  models/
    ridge.py                     + save() + classmethod load()
    lightgbm_model.py            + save() + classmethod load()
    xgboost_model.py             + save() + classmethod load()
    catboost_model.py            + save() + classmethod load()
    mlp.py                       + save() + classmethod load()  (handles state_dict + arch + scaler)
    sequence.py                  + save() + classmethod load()  (same pattern as mlp)

  stacking.py                    LinearStacker.save() / .load()
  inference.py                   + load_predictor_from_run(run_dir) -> S1Predictor
                                 + _BoundStackPredictor (private impl of S1Predictor)

scripts/
  train_s1.py                    NEW  ~40 LOC
  alpha_train_s1.py              DELETED
  alpha_train_s1_streaming.py    DELETED

tests/
  test_alpha_models_*.py         each gains a save/load roundtrip test
  test_alpha_stacking.py         + LinearStacker save/load roundtrip
  test_alpha_persistence.py      NEW  schema sha256, missing artifacts, column reorder
  test_alpha_training.py         NEW  end-to-end train_s1(...) on 10k synthetic rows
  conftest.py                    NEW  shared `synthetic_js` pytest fixture (deterministic 10k-row JS-shape)

Makefile                         full-retrain-s1 → scripts/train_s1.py
CLAUDE.md §13                    artifact list mirrors what _persist_run writes
docs/runbooks/s0_retrain_after_persistence.md  NEW
```

### 3.3 Migration plan

1. **Pre-flight:** `git mv experiments/alpha_s1/<each-run> experiments/alpha_s1/_archive_pre_s0/<each-run>`. All 5 pre-S0 runs preserved but moved aside.
2. The new training run writes to a fresh `experiments/alpha_s1/<new_run_id>/`.
3. `load_predictor_from_run` raises `ArtifactsMissingError("run pre-dates S0; retrain required")` if `feature_cols.json` or any of the 6 model files is missing. No silent reconstruction attempt.
4. `Makefile`'s `full-retrain-s1` target invokes `scripts/train_s1.py`. Streaming is opt-in via `make full-retrain-s1 STREAMING=1` (the Makefile passes `--streaming` to the script when `STREAMING` is non-empty). Operators running on the M4 24 GB target machine **should** pass `STREAMING=1` for memory-limited mode; the default (no flag) runs the full-in-memory pipeline.
5. References to `alpha_train_s1.py` / `alpha_train_s1_streaming.py` in `README.md`, `CLAUDE.md`, and runbooks updated to `train_s1.py`.

## 4. Per-class save/load contracts

Every wrapper gains `save(path: Path) -> None` and `@classmethod load(cls, path: Path) -> Self`. The on-disk format is always the library's native format — never pickle the wrapper. Wrapper-config (the `RidgeConfig` / `LightGBMConfig` / ... dataclass) is written either inside the file (when the format supports it) or as a sibling `<name>.config.json` (when it doesn't).

### 4.1 Tree + linear (5 classes)

| Class | Native format | Wrapper config | Sidecar? |
|---|---|---|---|
| `RidgeAlphaModel` | `joblib.dump` of `{"sklearn_estimator": Ridge, "config": asdict(config)}` | embedded | no |
| `LightGBMAlphaModel` | `booster.save_model(path.txt)` (LightGBM native text format) | sibling `lightgbm.config.json` | yes |
| `XGBoostAlphaModel` | `booster.save_model(path.json)` (XGB native JSON) | sibling `xgboost.config.json` | yes |
| `CatBoostAlphaModel` | `model.save_model(path.cbm)` (CatBoost native binary) | sibling `catboost.config.json` | yes |
| `LinearStacker` | `joblib.dump` of `{"weights", "feature_order", "intercept", "alpha"}` | embedded | no |

### 4.2 PyTorch models (2 classes)

`MLPAlphaModel` and `Conv1DAlphaModel` share the same save/load shape:

```python
def save(self, path: Path) -> None:
    torch.save({
        "state_dict": self._net.state_dict(),
        "arch": {
            "input_dim":   self._input_dim,
            "hidden_dims": list(self.config.hidden_dims),
            "dropout":     self.config.dropout,
            # for Conv1D: kernel_sizes, channels, etc.
        },
        "scaler": {
            "mean": self._scaler.mean_.tolist(),
            "std":  self._scaler.std_.tolist(),
        },
        "config": asdict(self.config),
    }, path)

@classmethod
def load(cls, path: Path) -> Self:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    inst = cls(MLPConfig(**payload["config"]))
    inst._input_dim = payload["arch"]["input_dim"]
    inst._net = _Net(**payload["arch"])
    inst._net.load_state_dict(payload["state_dict"])
    inst._net.eval()
    inst._scaler = _restore_scaler(payload["scaler"])
    return inst
```

Design constraints baked in:

| Aspect | Decision |
|---|---|
| Device on load | CPU; the caller may `.to('mps')` itself if MPS is wanted at inference. Prevents crashes on non-MPS machines. |
| `weights_only=False` | Required because we save a dict, not just state_dict. The source is our own training run — trust boundary intact. |
| `eval()` mode | Always called on load. Forgetting this leaves dropout active at inference and silently shifts predictions. |
| Scaler persistence | Mandatory. MLP/Conv1D fit an internal `StandardScaler`; without persisted mean/std, live inputs would be unscaled while the model expects scaled. |

## 5. Schema pinning

### 5.1 `feature_cols.json`

```json
{
  "feature_columns":      ["feature_00", "feature_01", "...", "feature_510"],
  "n_features":           511,
  "feature_cols_sha256":  "9f1a2b...",
  "target_column":        "responder_6",
  "weight_column":        "weight",
  "group_column":         "date_id"
}
```

`feature_cols_sha256` is `hashlib.sha256(json.dumps(feature_columns, separators=(',', ':')).encode()).hexdigest()` — canonical, order-sensitive, no whitespace dependence.

### 5.2 Loader strictness

```python
def load_predictor_from_run(run_dir: Path) -> S1Predictor:
    _assert_all_artifacts_present(run_dir)
    schema = _load_and_verify_schema(run_dir / "feature_cols.json")
    # ^ raises FeatureSchemaError if stored sha256 ≠ computed sha256

    base_models = {
        "ridge": RidgeAlphaModel.load(run_dir / "models" / "ridge.joblib"),
        "lgb":   LightGBMAlphaModel.load(run_dir / "models" / "lightgbm.txt"),
        "xgb":   XGBoostAlphaModel.load(run_dir / "models" / "xgboost.json"),
        "cat":   CatBoostAlphaModel.load(run_dir / "models" / "catboost.cbm"),
        "mlp":   MLPAlphaModel.load(run_dir / "models" / "mlp.pt"),
        "seq":   Conv1DAlphaModel.load(run_dir / "models" / "sequence.pt"),
    }
    stacker = LinearStacker.load(run_dir / "models" / "stacker.joblib")
    _assert_stacker_feature_order(stacker, expected=["ridge","lgb","xgb","cat","mlp","seq"])

    return _BoundStackPredictor(
        base_models=base_models,
        stacker=stacker,
        feature_columns=schema["feature_columns"],
    )
```

`_BoundStackPredictor.predict(row: pl.DataFrame)` implementation:

1. Assert `set(row.columns) >= set(self.feature_columns)`. Missing columns → `FeatureSchemaError`.
2. `x = row.select(self.feature_columns).to_numpy()[0]` — explicit reorder to training order. Extra columns in `row` are silently dropped (by construction of `select`).
3. Call each base model's `predict(x)` in `stacker.feature_order`, building the 6-element stack vector.
4. Return `(stacker.predict(stack)[0], confidence)` where confidence reuses the existing `_StackPredictor` agreement formula across the 6 base outputs.

### 5.3 Protection matrix

| Failure mode | Caught by |
|---|---|
| Live caller passes columns in wrong order | column-name-aware reorder in step 2 |
| Live caller passes a subset / missing columns | `set(row.columns) >= set(expected)` assert |
| Live caller passes extra columns not in training | silently dropped by `select` (acceptable) |
| `feature_cols.json` edited by hand | sha256 mismatch → `FeatureSchemaError` |
| Old run loaded with new code | missing files → `ArtifactsMissingError` |
| MLP loaded but dropout left on | loader calls `.eval()` |
| MLP loaded but scaler forgotten | scaler mean/std persisted in `.pt` payload |
| Stacker trained on 5 inputs, loader expects 6 | `_assert_stacker_feature_order` raises |

## 6. Training pipeline (training.py)

`train_s1(config, registry)` runs five phases. Each phase is a private helper to keep them individually testable.

### 6.1 Phase 1 — load + split + features (no behavior change)

```python
train_df, holdout_df, feature_cols = _load_and_split(config)
```

Reuses the existing adversarial filter and noise-floor drop.

### 6.2 Phase 2 — per-fold OOF (expanded to 6 base models)

```python
for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(train_feats)):
    fold_models, fold_oof = _fit_one_fold(
        fold_idx, x[tr_idx], y[tr_idx], w[tr_idx],
                  x[te_idx], y[te_idx], w[te_idx],
        config,
    )
    # fold_models is dict[str, BaseModel] — discarded at end of fold
    # fold_oof writes into the OOF arrays for ridge/lgb/xgb/cat/mlp/seq
```

Per-fold models are **not persisted** — same behavior as today. Only their OOF predictions matter for the stacker.

### 6.3 Phase 3 — stacker fit (input grows 5→6 columns)

```python
stack_x = np.column_stack([oof_ridge, oof_lgb, oof_xgb, oof_cat, oof_mlp, oof_seq])
stacker = LinearStacker(alpha=config.stacker_alpha)
stacker.fit(stack_x, y, w)
```

### 6.4 Phase 4 — refit on full train (NEW persisted base models)

```python
final_models = _refit_on_full(x_full, y_full, w_full, config)
# returns dict[str, BaseModel] for all 6 names
# each model.fit() on the entire train slice
# early-stopping eval = last 1000 rows of train (existing pattern preserved)
```

The 6 models produced here are the ones written to disk in phase 5.

### 6.5 Phase 5 — holdout eval + persist (BEHAVIOR CHANGE)

```python
h_stack = np.column_stack([
    final_models[name].predict(x_h)
    for name in stacker.feature_order   # all 6 now
])
holdout_pred = stacker.predict(h_stack)
holdout_r2   = weighted_zero_mean_r2(y_h, holdout_pred, w_h)
_persist_run(run_dir, final_models, stacker, feature_cols, metrics, predictions)
```

**Behavior change vs today.** Today's trainer zeroes the xgb/cat/mlp columns in `h_stack` before passing it to the stacker (because the per-fold models are gone and only ridge+lgb get refit). Post-S0 all six final models contribute. The reported holdout R² is therefore the honest stacker output and may differ from the May-19 0.0955 number. We track the difference in the run's `metrics.json` and document it in the runbook. If the new R² falls below the §13 gate of 0.012, that is a real signal, not a regression — the gate held before only by accident of the zero'd holdout stack.

### 6.6 `TrainConfig` / `RunResult`

```python
@dataclass(frozen=True)
class TrainConfig:
    data: DataConfig          # paths, target/weight/group cols, max_rows
    cv: CVConfig              # n_folds, purge_days, embargo_days, random_seed
    features: FeatureConfig   # lags, rolling windows, cross_sectional_ranks, noise feature
    models: ModelsConfig      # ridge/lgb/xgb/cat/mlp/seq nested configs (all Pydantic-validated)
    stacker_alpha: float
    streaming: bool
    max_rows: int             # honored only when streaming=True

@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    fold_metrics: list[dict[str, float]]
    holdout_weighted_zero_mean_r2: float
    n_features_after_adversarial: int
    n_features_after_noise_floor: int
    base_models_persisted: list[str]   # ["ridge","lgb","xgb","cat","mlp","seq"]
    stacker_path: Path
    feature_cols_path: Path
```

Pydantic loads `configs/alpha.yaml` (or `configs/alpha_5m.yaml` via `--config`) into `TrainConfig`. Validation failure exits non-zero before any model is fit.

## 7. `scripts/train_s1.py` — thin CLI

```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/alpha.yaml")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--experiments-root", default="experiments/alpha_s1")
    args = parser.parse_args()

    config_dict = yaml.safe_load(open(args.config))
    if args.streaming:
        config_dict["streaming"] = True
    if args.max_rows is not None:
        config_dict["max_rows"] = args.max_rows
    config = TrainConfig.from_dict(config_dict)

    registry = RunRegistry(root=Path(args.experiments_root))
    result = train_s1(config, registry)

    console.print(f"[bold green]Run complete:[/bold green] {result.run_dir}")
    console.print(f"  holdout weighted zero-mean R²: {result.holdout_weighted_zero_mean_r2:.6f}")
    console.print(f"  base models persisted: {result.base_models_persisted}")
    return 0
```

~40 LOC. No business logic. Everything testable lives in `training.py`.

## 8. Testing

| Test file | Coverage | Marker |
|---|---|---|
| `tests/test_alpha_models_ridge.py` | + roundtrip: fit on 100 rows, save to tmp, load, assert `np.allclose(reloaded.predict(x), original.predict(x))` byte-exact | unit |
| `tests/test_alpha_models_lightgbm.py` | + same roundtrip pattern, byte-exact | unit |
| `tests/test_alpha_models_xgboost.py` | + same, byte-exact | unit |
| `tests/test_alpha_models_catboost.py` | + same, byte-exact | unit |
| `tests/test_alpha_models_mlp.py` | + same; `atol=1e-7, rtol=1e-6` (torch float ordering); assert `.eval()` mode after load; assert dropout actually disabled (forward twice → identical) | unit |
| `tests/test_alpha_models_sequence.py` | + same as MLP | unit |
| `tests/test_alpha_stacking.py` | + `LinearStacker` save/load roundtrip; bit-exact | unit |
| `tests/test_alpha_persistence.py` | NEW. Schema sha256 mismatch → `FeatureSchemaError`. Missing model file → `ArtifactsMissingError`. Stacker `feature_order` mismatch → assertion. `_BoundStackPredictor.predict` rejects DataFrame with missing columns. Reorder test: caller passes columns shuffled; predictor reorders correctly. | unit |
| `tests/test_alpha_training.py` | NEW. `train_s1(...)` end-to-end on 10k synthetic JS-shaped rows; asserts all expected files present, then immediately calls `load_predictor_from_run(run_dir)` and verifies it returns a working predictor. Round-trip integration test (no marker; fast). | unit |

`tests/conftest.py` exposes a `synthetic_js` pytest fixture: deterministic JS-shaped 10k-row Polars DataFrame with 50 features × `weight` × `responder_6` × `date_id`. Stationary noise + a few linearly-predictive features. Single source of truth used by `test_alpha_persistence.py` and `test_alpha_training.py`.

**Tests intentionally not added:**

- No test asserts a specific holdout R² number — the synthetic data isn't calibrated for that.
- No test exercises `--streaming` mode on real data — covered by the May-21 retrain (§9.2).

## 9. Acceptance criteria

### 9.1 Code acceptance

1. `PYTHONPATH=src uv run pytest -q` — all tests green, including 8 new/extended tests.
2. `PYTHONPATH=src uv run ruff check src scripts tests` — clean.
3. `PYTHONPATH=src uv run mypy src` — clean.
4. `scripts/alpha_train_s1.py` and `scripts/alpha_train_s1_streaming.py` removed; `git grep alpha_train_s1` returns nothing.
5. `Makefile`'s `full-retrain-s1` target invokes `scripts/train_s1.py`.
6. `CLAUDE.md` §13 artifact list mirrors `_persist_run` output (no spec drift).
7. `experiments/alpha_s1/_archive_pre_s0/` contains the 5 pre-S0 runs untouched.

### 9.2 Retrain acceptance

8. `make full-retrain-s1 STREAMING=1` runs end-to-end on the M4 within CLAUDE.md §8 budget (≤ 24 h wall-clock).
9. The new `experiments/alpha_s1/<may-21-run-id>/` directory contains:
   - `metrics.json` with `holdout_weighted_zero_mean_r2 >= 0.012` (CLAUDE.md §13 gate).
   - All 6 base-model files + `stacker.joblib` + `feature_cols.json` + extended `_artifact_sha256.json`.
   - `models/sequence.pt` is non-empty (sequence model actually fit, not skipped).
10. The following one-liner succeeds:
    ```bash
    PYTHONPATH=src uv run python -c "from pathlib import Path; from quant_research_stack.alpha.inference import load_predictor_from_run; p = load_predictor_from_run(Path('experiments/alpha_s1/<may-21-run-id>')); print(p.expected_feature_columns[:5])"
    ```
11. The retrain's holdout R² is honest — phase 5 uses all 6 final models, not zeros for 3 of them. The before/after R² difference is recorded in `docs/runbooks/s0_retrain_after_persistence.md`.

### 9.3 Documentation acceptance

12. `docs/runbooks/s0_retrain_after_persistence.md` exists, documents the operator command, the expected wall-clock, the loader verification step, and the honest-vs-zeroed R² comparison.
13. `CLAUDE.md` §13's required-artifacts block matches what `_persist_run` writes — `ls experiments/alpha_s1/<may-21-run-id>/models/` and the §13 listing agree.

## 10. What S0 unblocks

- **S5/S6/S8** (multi-universe retrains). Each can `from quant_research_stack.alpha.training import train_s1` and call it programmatically with a per-universe `TrainConfig`. No CLI subprocess gymnastics; loop testable.
- **S4.1β live serving** (the deferred `s1_serve`). When that spec re-opens, `load_predictor_from_run` is ready. No new persistence work needed.
- **Validation extensions.** `tv_validation_report` (or any future validator) can verify a production run's artifact bundle is complete and loadable before opening a paper-trading session.

## 11. References

- `CLAUDE.md` §13 — completion criteria for S1 milestone (artifact list updated by this spec).
- `docs/superpowers/specs/2026-05-14-quantlab-alpha-platform-design.md` — master architecture; §2 (S1 detailed) is the authority for the model hierarchy.
- `docs/superpowers/plans/2026-05-14-quantlab-alpha-s1-implementation.md` — original S1 implementation plan; this spec closes the persistence gap that plan deferred.
- `docs/superpowers/specs/2026-05-20-quantlab-alpha-s4_1alpha-tradingview-paper-validation-design.md` — downstream consumer (validation tooling will benefit from honest holdout R²).
- ADRs 0001 (two-tier tabular/LLM) and 0002 (three-stage promotion gate) — unchanged by this spec.
