# QuantLab Alpha S0 — Trainer Base-Model Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every S1 base model loadable from disk after training, unify the two diverged trainers into one testable module, and pin the feature schema with a sha256 so the loader refuses callers with mismatched columns.

**Architecture:** Each base-model wrapper (`RidgeAlphaModel`, `LightGBMAlphaModel`, `XGBoostAlphaModel`, `CatBoostAlphaModel`, `MLPAlphaModel`, `Conv1DAlphaModel`, `LinearStacker`) gains a symmetric `save(path) / @classmethod load(path)` pair using the library's native format. A new `src/quant_research_stack/alpha/training.py` holds the unified `train_s1(config, registry) -> RunResult` pure function. A new `scripts/train_s1.py` (~40 LOC) replaces both existing trainers. `inference.load_predictor_from_run(run_dir) -> S1Predictor` is the public loader; it raises `FeatureSchemaError` on sha256 mismatch and `ArtifactsMissingError` for pre-S0 runs.

**Tech Stack:** Python 3.11, Polars, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, PyTorch (MPS / CPU), joblib, Pydantic v2, pytest, uv. Native serialization formats: `.joblib`, `.txt`, `.json`, `.cbm`, `.pt`.

**Spec:** `docs/superpowers/specs/2026-05-21-quantlab-alpha-s0-trainer-persistence-design.md`

---

## File Structure

### Files created

| Path | Lines | Responsibility |
|---|---|---|
| `src/quant_research_stack/alpha/exceptions.py` | ~25 | `FeatureSchemaError`, `ArtifactsMissingError`, `ArtifactCorruptError` |
| `src/quant_research_stack/alpha/training.py` | ~400 | Unified `train_s1` pipeline (phases 1–5) + `TrainConfig` / `RunResult` |
| `scripts/train_s1.py` | ~40 | Thin CLI: argparse → yaml → `train_s1(...)` |
| `tests/conftest.py` | ~50 | Shared `synthetic_js` pytest fixture (deterministic 10k JS-shape rows) |
| `tests/test_alpha_persistence.py` | ~150 | Loader edge cases — sha256 mismatch, missing artifacts, column reorder, stacker order |
| `tests/test_alpha_training.py` | ~80 | End-to-end `train_s1(synthetic_js)` → on-disk layout → `load_predictor_from_run` roundtrip |
| `docs/runbooks/s0_retrain_after_persistence.md` | ~80 | Operator runbook for the May-21 retrain (command, expected wall-clock, verification, R² before/after) |

### Files modified

| Path | Change |
|---|---|
| `src/quant_research_stack/alpha/models/ridge.py` | + `save(path)` + `@classmethod load(cls, path)` |
| `src/quant_research_stack/alpha/models/lightgbm_model.py` | + `save / load` (native `.txt` + sidecar config JSON) |
| `src/quant_research_stack/alpha/models/xgboost_model.py` | + `save / load` (native `.json` + sidecar config JSON) |
| `src/quant_research_stack/alpha/models/catboost_model.py` | + `save / load` (native `.cbm` + sidecar config JSON) |
| `src/quant_research_stack/alpha/models/mlp.py` | + `save / load` (state_dict + arch + scaler embedded in `.pt`); exposes `_input_dim` + `_scaler` |
| `src/quant_research_stack/alpha/models/sequence.py` | + `save / load` (same shape as MLP) |
| `src/quant_research_stack/alpha/stacking.py` | `LinearStacker.save / load` |
| `src/quant_research_stack/alpha/inference.py` | + `load_predictor_from_run`, `_BoundStackPredictor`, `_canonical_sha256`, `_load_and_verify_schema`, `_assert_all_artifacts_present` |
| `tests/test_alpha_models_ridge.py` | + roundtrip test |
| `tests/test_alpha_models_lightgbm.py` | + roundtrip test |
| `tests/test_alpha_models_xgboost.py` | + roundtrip test |
| `tests/test_alpha_models_catboost.py` | + roundtrip test |
| `tests/test_alpha_models_mlp.py` | + roundtrip test (incl. `.eval()` + dropout-disabled assertion) |
| `tests/test_alpha_models_sequence.py` | + roundtrip test |
| `tests/test_alpha_stacking.py` | + roundtrip test |
| `Makefile` | `full-retrain-s1` calls `scripts/train_s1.py`; `STREAMING=1` env passthrough |
| `CLAUDE.md` | §13 artifact list matches `_persist_run` output |

### Files deleted

| Path | Reason |
|---|---|
| `scripts/alpha_train_s1.py` | replaced by `scripts/train_s1.py` |
| `scripts/alpha_train_s1_streaming.py` | replaced by `scripts/train_s1.py` (streaming becomes opt-in flag) |

### Migration (one-off)

| Action | Source → Destination |
|---|---|
| `git mv` | `experiments/alpha_s1/{20260517-173937,20260517-211119,20260518-102255,20260519-074922,20260519-202114}` → `experiments/alpha_s1/_archive_pre_s0/...` |

---

## Task Index

| # | Task | Approx. LOC |
|---|---|---|
| 1 | Archive pre-S0 runs | 0 (git mv only) |
| 2 | `alpha/exceptions.py` | 25 |
| 3 | `RidgeAlphaModel.save/load` | 50 |
| 4 | `LightGBMAlphaModel.save/load` | 55 |
| 5 | `XGBoostAlphaModel.save/load` | 55 |
| 6 | `CatBoostAlphaModel.save/load` | 55 |
| 7 | `MLPAlphaModel.save/load` | 90 |
| 8 | `Conv1DAlphaModel.save/load` | 90 |
| 9 | `LinearStacker.save/load` | 50 |
| 10 | `tests/conftest.py` synthetic_js fixture | 60 |
| 11 | `inference.load_predictor_from_run` + `_BoundStackPredictor` + persistence tests | 240 |
| 12 | `training.py` scaffold — `TrainConfig`, `RunResult`, Pydantic loaders | 130 |
| 13 | `training.py` phases 1–3 (load + fit_one_fold + stacker fit) | 180 |
| 14 | `training.py` phases 4–5 (refit_on_full + holdout + `_persist_run`) | 170 |
| 15 | `tests/test_alpha_training.py` end-to-end roundtrip | 80 |
| 16 | `scripts/train_s1.py` thin CLI | 40 |
| 17 | Delete old trainers + rewrite `Makefile` `full-retrain-s1` | 20 |
| 18 | `CLAUDE.md` §13 update + runbook + operator retrain gate | docs only |

---

### Task 1: Archive pre-S0 runs

**Files:**
- Move: `experiments/alpha_s1/{20260517-173937, 20260517-211119, 20260518-102255, 20260519-074922, 20260519-202114}` → `experiments/alpha_s1/_archive_pre_s0/`

- [ ] **Step 1: Create the archive directory and move each pre-S0 run**

```bash
mkdir -p experiments/alpha_s1/_archive_pre_s0
git mv experiments/alpha_s1/20260517-173937 experiments/alpha_s1/_archive_pre_s0/20260517-173937
git mv experiments/alpha_s1/20260517-211119 experiments/alpha_s1/_archive_pre_s0/20260517-211119
git mv experiments/alpha_s1/20260518-102255 experiments/alpha_s1/_archive_pre_s0/20260518-102255
git mv experiments/alpha_s1/20260519-074922 experiments/alpha_s1/_archive_pre_s0/20260519-074922
git mv experiments/alpha_s1/20260519-202114 experiments/alpha_s1/_archive_pre_s0/20260519-202114
```

- [ ] **Step 2: Verify the layout**

Run: `ls experiments/alpha_s1/ && echo --- && ls experiments/alpha_s1/_archive_pre_s0/`
Expected: top-level shows only `_archive_pre_s0`; inside it the 5 run directories.

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(s0): archive pre-S0 alpha_s1 runs under _archive_pre_s0/"
```

---

### Task 2: New exceptions module

**Files:**
- Create: `src/quant_research_stack/alpha/exceptions.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_alpha_exceptions.py`:

```python
from __future__ import annotations

import pytest

from quant_research_stack.alpha.exceptions import (
    ArtifactCorruptError,
    ArtifactsMissingError,
    FeatureSchemaError,
)


def test_exceptions_are_runtimeerrors():
    assert issubclass(FeatureSchemaError, RuntimeError)
    assert issubclass(ArtifactsMissingError, RuntimeError)
    assert issubclass(ArtifactCorruptError, RuntimeError)


def test_feature_schema_error_carries_message():
    with pytest.raises(FeatureSchemaError, match="sha256 mismatch"):
        raise FeatureSchemaError("sha256 mismatch")


def test_artifacts_missing_error_carries_path():
    err = ArtifactsMissingError("missing models/ridge.joblib")
    assert "ridge.joblib" in str(err)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_exceptions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant_research_stack.alpha.exceptions'`.

- [ ] **Step 3: Create the exceptions module**

Create `src/quant_research_stack/alpha/exceptions.py`:

```python
"""Exceptions raised by the alpha persistence + serving layer."""

from __future__ import annotations


class FeatureSchemaError(RuntimeError):
    """Raised when a feature-schema invariant is violated.

    Cases:
    - feature_cols.json sha256 mismatch (file edited by hand)
    - caller passes a DataFrame whose columns don't cover the trained feature set
    """


class ArtifactsMissingError(RuntimeError):
    """Raised when a run directory lacks one or more required S0 artifacts.

    Typically encountered when loading a pre-S0 run (only stacker.joblib exists).
    """


class ArtifactCorruptError(RuntimeError):
    """Raised when an on-disk artifact exists but fails to load.

    Wraps the underlying library exception (joblib / torch / lgb / xgb / catboost)
    so callers don't have to catch six different types.
    """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_exceptions.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/exceptions.py tests/test_alpha_exceptions.py
git commit -m "feat(s0): alpha.exceptions module — FeatureSchemaError, ArtifactsMissingError, ArtifactCorruptError"
```

---

### Task 3: `RidgeAlphaModel.save / load`

**Files:**
- Modify: `src/quant_research_stack/alpha/models/ridge.py`
- Modify: `tests/test_alpha_models_ridge.py`

- [ ] **Step 1: Write the failing roundtrip test**

Append to `tests/test_alpha_models_ridge.py`:

```python
import numpy as np

from quant_research_stack.alpha.models.ridge import RidgeAlphaModel, RidgeConfig


def test_ridge_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 8))
    y = x @ rng.standard_normal(8) + 0.1 * rng.standard_normal(200)
    w = np.ones(200)

    original = RidgeAlphaModel(RidgeConfig(alpha=1.0))
    original.fit(x, y, w)

    path = tmp_path / "ridge.joblib"
    original.save(path)
    assert path.exists()

    reloaded = RidgeAlphaModel.load(path)

    # Bit-exact: sklearn Ridge is deterministic across save/load.
    np.testing.assert_array_equal(original.predict(x), reloaded.predict(x))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_models_ridge.py::test_ridge_save_load_roundtrip -v`
Expected: FAIL — `AttributeError: 'RidgeAlphaModel' object has no attribute 'save'`.

- [ ] **Step 3: Implement save/load on RidgeAlphaModel**

Modify `src/quant_research_stack/alpha/models/ridge.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class RidgeConfig:
    alpha: float = 1.0


class RidgeAlphaModel:
    def __init__(self, config: RidgeConfig) -> None:
        self.config = config
        self._model: Ridge = Ridge(alpha=config.alpha)

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        self._model.fit(x, y, sample_weight=weights)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._model.predict(x).astype(np.float64)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"sklearn_estimator": self._model, "config": asdict(self.config)},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "RidgeAlphaModel":
        path = Path(path)
        payload = joblib.load(path)
        inst = cls(RidgeConfig(**payload["config"]))
        inst._model = payload["sklearn_estimator"]
        return inst
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_models_ridge.py -v`
Expected: all green, including the new roundtrip test.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/models/ridge.py tests/test_alpha_models_ridge.py
git commit -m "feat(s0): RidgeAlphaModel.save/load via joblib with embedded config"
```

---

### Task 4: `LightGBMAlphaModel.save / load`

**Files:**
- Modify: `src/quant_research_stack/alpha/models/lightgbm_model.py`
- Modify: `tests/test_alpha_models_lightgbm.py`

- [ ] **Step 1: Write the failing roundtrip test**

Append to `tests/test_alpha_models_lightgbm.py`:

```python
import json

import numpy as np

from quant_research_stack.alpha.models.lightgbm_model import (
    LightGBMAlphaModel,
    LightGBMConfig,
)


def test_lightgbm_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x_tr = rng.standard_normal((1000, 8))
    y_tr = x_tr[:, 0] + 0.1 * rng.standard_normal(1000)
    w_tr = np.ones(1000)
    x_val = rng.standard_normal((200, 8))
    y_val = x_val[:, 0] + 0.1 * rng.standard_normal(200)
    w_val = np.ones(200)

    cfg = LightGBMConfig(
        num_leaves=15,
        max_depth=4,
        learning_rate=0.1,
        n_estimators=50,
        early_stopping_rounds=10,
        feature_fraction=1.0,
        bagging_fraction=1.0,
    )
    original = LightGBMAlphaModel(cfg)
    original.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)

    path = tmp_path / "lightgbm.txt"
    original.save(path)
    assert path.exists()
    sidecar = path.parent / "lightgbm.config.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["num_leaves"] == 15

    reloaded = LightGBMAlphaModel.load(path)
    np.testing.assert_array_equal(original.predict(x_val), reloaded.predict(x_val))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_models_lightgbm.py::test_lightgbm_save_load_roundtrip -v`
Expected: FAIL — `AttributeError: 'LightGBMAlphaModel' object has no attribute 'save'`.

- [ ] **Step 3: Implement save/load**

Modify `src/quant_research_stack/alpha/models/lightgbm_model.py` (preserving the existing `fit`/`predict` signatures). Add at the bottom of the class:

```python
import json
from pathlib import Path

# ... existing class body ...

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._booster is None:
            raise RuntimeError("cannot save un-fitted LightGBMAlphaModel")
        self._booster.save_model(str(path))
        sidecar = path.parent / "lightgbm.config.json"
        sidecar.write_text(json.dumps(asdict(self.config), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "LightGBMAlphaModel":
        import lightgbm as lgb

        path = Path(path)
        sidecar = path.parent / "lightgbm.config.json"
        if not sidecar.exists():
            raise FileNotFoundError(f"missing sidecar config: {sidecar}")
        cfg_dict = json.loads(sidecar.read_text())
        inst = cls(LightGBMConfig(**cfg_dict))
        inst._booster = lgb.Booster(model_file=str(path))
        return inst
```

Make sure `from dataclasses import asdict` is imported at module top if not already.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_models_lightgbm.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/models/lightgbm_model.py tests/test_alpha_models_lightgbm.py
git commit -m "feat(s0): LightGBMAlphaModel.save/load (native .txt + sidecar config JSON)"
```

---

### Task 5: `XGBoostAlphaModel.save / load`

**Files:**
- Modify: `src/quant_research_stack/alpha/models/xgboost_model.py`
- Modify: `tests/test_alpha_models_xgboost.py`

- [ ] **Step 1: Write the failing roundtrip test**

Append to `tests/test_alpha_models_xgboost.py`:

```python
import json

import numpy as np

from quant_research_stack.alpha.models.xgboost_model import (
    XGBoostAlphaModel,
    XGBoostConfig,
)


def test_xgboost_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x_tr = rng.standard_normal((1000, 8))
    y_tr = x_tr[:, 0] + 0.1 * rng.standard_normal(1000)
    w_tr = np.ones(1000)
    x_val = rng.standard_normal((200, 8))
    y_val = x_val[:, 0] + 0.1 * rng.standard_normal(200)
    w_val = np.ones(200)

    cfg = XGBoostConfig(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=50,
        early_stopping_rounds=10,
        tree_method="hist",
    )
    original = XGBoostAlphaModel(cfg)
    original.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)

    path = tmp_path / "xgboost.json"
    original.save(path)
    assert path.exists()
    sidecar = path.parent / "xgboost.config.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["max_depth"] == 4

    reloaded = XGBoostAlphaModel.load(path)
    np.testing.assert_array_equal(original.predict(x_val), reloaded.predict(x_val))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_models_xgboost.py::test_xgboost_save_load_roundtrip -v`
Expected: FAIL with `AttributeError: ... no attribute 'save'`.

- [ ] **Step 3: Implement save/load**

Add to `src/quant_research_stack/alpha/models/xgboost_model.py`:

```python
import json
from dataclasses import asdict
from pathlib import Path

# ... existing class body ...

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._booster is None:
            raise RuntimeError("cannot save un-fitted XGBoostAlphaModel")
        self._booster.save_model(str(path))
        sidecar = path.parent / "xgboost.config.json"
        sidecar.write_text(json.dumps(asdict(self.config), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "XGBoostAlphaModel":
        import xgboost as xgb

        path = Path(path)
        sidecar = path.parent / "xgboost.config.json"
        if not sidecar.exists():
            raise FileNotFoundError(f"missing sidecar config: {sidecar}")
        cfg_dict = json.loads(sidecar.read_text())
        inst = cls(XGBoostConfig(**cfg_dict))
        booster = xgb.Booster()
        booster.load_model(str(path))
        inst._booster = booster
        return inst
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_models_xgboost.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/models/xgboost_model.py tests/test_alpha_models_xgboost.py
git commit -m "feat(s0): XGBoostAlphaModel.save/load (native .json + sidecar config JSON)"
```

---

### Task 6: `CatBoostAlphaModel.save / load`

**Files:**
- Modify: `src/quant_research_stack/alpha/models/catboost_model.py`
- Modify: `tests/test_alpha_models_catboost.py`

- [ ] **Step 1: Write the failing roundtrip test**

Append to `tests/test_alpha_models_catboost.py`:

```python
import json

import numpy as np

from quant_research_stack.alpha.models.catboost_model import (
    CatBoostAlphaModel,
    CatBoostConfig,
)


def test_catboost_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x_tr = rng.standard_normal((1000, 8))
    y_tr = x_tr[:, 0] + 0.1 * rng.standard_normal(1000)
    w_tr = np.ones(1000)
    x_val = rng.standard_normal((200, 8))
    y_val = x_val[:, 0] + 0.1 * rng.standard_normal(200)
    w_val = np.ones(200)

    cfg = CatBoostConfig(
        depth=4,
        learning_rate=0.1,
        n_estimators=50,
        early_stopping_rounds=10,
    )
    original = CatBoostAlphaModel(cfg)
    original.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)

    path = tmp_path / "catboost.cbm"
    original.save(path)
    assert path.exists()
    sidecar = path.parent / "catboost.config.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["depth"] == 4

    reloaded = CatBoostAlphaModel.load(path)
    np.testing.assert_array_equal(original.predict(x_val), reloaded.predict(x_val))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_models_catboost.py::test_catboost_save_load_roundtrip -v`
Expected: FAIL with `AttributeError: ... no attribute 'save'`.

- [ ] **Step 3: Implement save/load**

Add to `src/quant_research_stack/alpha/models/catboost_model.py`:

```python
import json
from dataclasses import asdict
from pathlib import Path

# ... existing class body ...

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._model is None:
            raise RuntimeError("cannot save un-fitted CatBoostAlphaModel")
        self._model.save_model(str(path))
        sidecar = path.parent / "catboost.config.json"
        sidecar.write_text(json.dumps(asdict(self.config), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "CatBoostAlphaModel":
        from catboost import CatBoostRegressor

        path = Path(path)
        sidecar = path.parent / "catboost.config.json"
        if not sidecar.exists():
            raise FileNotFoundError(f"missing sidecar config: {sidecar}")
        cfg_dict = json.loads(sidecar.read_text())
        inst = cls(CatBoostConfig(**cfg_dict))
        inst._model = CatBoostRegressor()
        inst._model.load_model(str(path))
        return inst
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_models_catboost.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/models/catboost_model.py tests/test_alpha_models_catboost.py
git commit -m "feat(s0): CatBoostAlphaModel.save/load (native .cbm + sidecar config JSON)"
```

---

### Task 7: `MLPAlphaModel.save / load`

**Files:**
- Modify: `src/quant_research_stack/alpha/models/mlp.py`
- Modify: `tests/test_alpha_models_mlp.py`

This is the most involved per-class task because the MLP carries both PyTorch state and a fitted scaler that *must* round-trip.

- [ ] **Step 1: Write the failing roundtrip test**

Append to `tests/test_alpha_models_mlp.py`:

```python
import numpy as np
import torch

from quant_research_stack.alpha.models.mlp import MLPAlphaModel, MLPConfig


def test_mlp_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x_tr = rng.standard_normal((500, 8)).astype(np.float64)
    y_tr = x_tr[:, 0].astype(np.float64) + 0.1 * rng.standard_normal(500)
    w_tr = np.ones(500)
    x_val = rng.standard_normal((100, 8)).astype(np.float64)
    y_val = x_val[:, 0].astype(np.float64)
    w_val = np.ones(100)

    cfg = MLPConfig(
        hidden_dims=[16, 8],
        dropout=0.2,
        learning_rate=1e-3,
        batch_size=64,
        max_epochs=3,
        patience=2,
        mixed_precision=False,
    )
    original = MLPAlphaModel(cfg)
    original.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)

    path = tmp_path / "mlp.pt"
    original.save(path)
    assert path.exists()

    reloaded = MLPAlphaModel.load(path)
    np.testing.assert_allclose(
        original.predict(x_val), reloaded.predict(x_val), atol=1e-7, rtol=1e-6
    )

    # Loader puts net in eval() mode — two consecutive forwards must be identical (dropout off).
    first = reloaded.predict(x_val)
    second = reloaded.predict(x_val)
    np.testing.assert_array_equal(first, second)
    assert not reloaded._net.training
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_models_mlp.py::test_mlp_save_load_roundtrip -v`
Expected: FAIL with `AttributeError: ... no attribute 'save'`.

- [ ] **Step 3: Implement save/load**

Modify `src/quant_research_stack/alpha/models/mlp.py`. Two things to add: a `_Scaler` dataclass for clarity (existing code likely uses `sklearn.preprocessing.StandardScaler`; if so, just persist `mean_` and `scale_` arrays directly), and the `save`/`load` methods.

```python
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

# ... existing _Net definition + helpers + class header ...

class MLPAlphaModel:
    # ... existing __init__, fit, predict ...
    # After fit() the instance has:
    #   self.config: MLPConfig
    #   self._input_dim: int                 (set when first batch arrives during fit)
    #   self._net: nn.Module                 (the trained _Net)
    #   self._scaler: StandardScaler         (sklearn or equivalent — needs mean_, scale_)
    #   self._device: torch.device

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(self, "_net") or self._net is None:
            raise RuntimeError("cannot save un-fitted MLPAlphaModel")
        payload = {
            "state_dict": self._net.state_dict(),
            "arch": {
                "input_dim": int(self._input_dim),
                "hidden_dims": list(self.config.hidden_dims),
                "dropout": float(self.config.dropout),
            },
            "scaler": {
                "mean": np.asarray(self._scaler.mean_, dtype=np.float64).tolist(),
                "scale": np.asarray(self._scaler.scale_, dtype=np.float64).tolist(),
            },
            "config": asdict(self.config),
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: Path) -> "MLPAlphaModel":
        from sklearn.preprocessing import StandardScaler

        path = Path(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        inst = cls(MLPConfig(**payload["config"]))
        inst._input_dim = payload["arch"]["input_dim"]
        inst._net = _Net(
            input_dim=payload["arch"]["input_dim"],
            hidden_dims=payload["arch"]["hidden_dims"],
            dropout=payload["arch"]["dropout"],
        )
        inst._net.load_state_dict(payload["state_dict"])
        inst._net.eval()
        inst._device = torch.device("cpu")
        inst._scaler = StandardScaler()
        inst._scaler.mean_ = np.asarray(payload["scaler"]["mean"], dtype=np.float64)
        inst._scaler.scale_ = np.asarray(payload["scaler"]["scale"], dtype=np.float64)
        inst._scaler.var_ = inst._scaler.scale_ ** 2
        inst._scaler.n_features_in_ = inst._scaler.mean_.size
        return inst
```

Notes for the implementer:
- If the existing `MLPAlphaModel.fit` does not currently store `self._input_dim` or `self._scaler` as instance attributes, add them inside `fit` before persistence works. Reading `fit` first, then mirroring the names is required.
- `_Net`'s constructor signature must accept `input_dim`, `hidden_dims`, `dropout` keyword args. If it currently takes positional args, switch to keyword-only with the same names.
- `predict` must apply `self._scaler` before forwarding through `self._net` — confirm this is already the case (it should be; otherwise the test will fail on `allclose`).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_models_mlp.py -v`
Expected: all green, including the new roundtrip + dropout-disabled assertion.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/models/mlp.py tests/test_alpha_models_mlp.py
git commit -m "feat(s0): MLPAlphaModel.save/load (state_dict + arch + scaler embedded in .pt)"
```

---

### Task 8: `Conv1DAlphaModel.save / load`

**Files:**
- Modify: `src/quant_research_stack/alpha/models/sequence.py`
- Modify: `tests/test_alpha_models_sequence.py`

Same shape as Task 7, but the Conv1D network needs different arch fields. Read the existing `_Conv1DNet.__init__` to enumerate them.

- [ ] **Step 1: Write the failing roundtrip test**

Append to `tests/test_alpha_models_sequence.py`:

```python
import numpy as np
import torch

from quant_research_stack.alpha.models.sequence import (
    Conv1DAlphaModel,
    Conv1DConfig,
)


def test_conv1d_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x_tr = rng.standard_normal((500, 8)).astype(np.float64)
    y_tr = x_tr[:, 0].astype(np.float64) + 0.1 * rng.standard_normal(500)
    w_tr = np.ones(500)
    x_val = rng.standard_normal((100, 8)).astype(np.float64)
    y_val = x_val[:, 0].astype(np.float64)
    w_val = np.ones(100)

    cfg = Conv1DConfig(
        kernel_sizes=[3, 5],
        channels=[8, 16],
        dropout=0.1,
        learning_rate=1e-3,
        batch_size=64,
        max_epochs=2,
        patience=2,
        random_state=0,
    )
    original = Conv1DAlphaModel(cfg)
    original.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)

    path = tmp_path / "sequence.pt"
    original.save(path)
    assert path.exists()

    reloaded = Conv1DAlphaModel.load(path)
    np.testing.assert_allclose(
        original.predict(x_val), reloaded.predict(x_val), atol=1e-7, rtol=1e-6
    )
    first = reloaded.predict(x_val)
    second = reloaded.predict(x_val)
    np.testing.assert_array_equal(first, second)
    assert not reloaded._net.training
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_models_sequence.py::test_conv1d_save_load_roundtrip -v`
Expected: FAIL with `AttributeError: ... no attribute 'save'`.

- [ ] **Step 3: Implement save/load**

Modify `src/quant_research_stack/alpha/models/sequence.py`. Mirror the MLP pattern; the arch dict carries `input_dim`, `kernel_sizes`, `channels`, `dropout`:

```python
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

# ... existing _Conv1DNet definition + helpers + class header ...

class Conv1DAlphaModel:
    # ... existing __init__, fit, predict ...

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(self, "_net") or self._net is None:
            raise RuntimeError("cannot save un-fitted Conv1DAlphaModel")
        payload = {
            "state_dict": self._net.state_dict(),
            "arch": {
                "input_dim": int(self._input_dim),
                "kernel_sizes": list(self.config.kernel_sizes),
                "channels": list(self.config.channels),
                "dropout": float(self.config.dropout),
            },
            "scaler": {
                "mean": np.asarray(self._scaler.mean_, dtype=np.float64).tolist(),
                "scale": np.asarray(self._scaler.scale_, dtype=np.float64).tolist(),
            },
            "config": asdict(self.config),
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: Path) -> "Conv1DAlphaModel":
        from sklearn.preprocessing import StandardScaler

        path = Path(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        inst = cls(Conv1DConfig(**payload["config"]))
        inst._input_dim = payload["arch"]["input_dim"]
        inst._net = _Conv1DNet(
            input_dim=payload["arch"]["input_dim"],
            kernel_sizes=payload["arch"]["kernel_sizes"],
            channels=payload["arch"]["channels"],
            dropout=payload["arch"]["dropout"],
        )
        inst._net.load_state_dict(payload["state_dict"])
        inst._net.eval()
        inst._device = torch.device("cpu")
        inst._scaler = StandardScaler()
        inst._scaler.mean_ = np.asarray(payload["scaler"]["mean"], dtype=np.float64)
        inst._scaler.scale_ = np.asarray(payload["scaler"]["scale"], dtype=np.float64)
        inst._scaler.var_ = inst._scaler.scale_ ** 2
        inst._scaler.n_features_in_ = inst._scaler.mean_.size
        return inst
```

Same caveats as Task 7: read existing `fit` first; ensure `_Conv1DNet.__init__` accepts keyword args matching the arch dict; ensure `predict` applies `self._scaler`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_models_sequence.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/models/sequence.py tests/test_alpha_models_sequence.py
git commit -m "feat(s0): Conv1DAlphaModel.save/load (state_dict + arch + scaler embedded in .pt)"
```

---

### Task 9: `LinearStacker.save / load`

**Files:**
- Modify: `src/quant_research_stack/alpha/stacking.py`
- Modify: `tests/test_alpha_stacking.py`

The existing trainer already writes a dict shaped like `{"weights": [...], "feature_order": [...]}`; this task formalises it on the class and adds the symmetric loader.

- [ ] **Step 1: Write the failing roundtrip test**

Append to `tests/test_alpha_stacking.py`:

```python
import numpy as np

from quant_research_stack.alpha.stacking import LinearStacker


def test_stacker_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 6))     # 6 base models post-S0
    true_w = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
    y = x @ true_w + 0.05 * rng.standard_normal(500)
    w = np.ones(500)

    feature_order = ["ridge", "lgb", "xgb", "cat", "mlp", "seq"]
    original = LinearStacker(alpha=1e-3, feature_order=feature_order)
    original.fit(x, y, w)

    path = tmp_path / "stacker.joblib"
    original.save(path)
    assert path.exists()

    reloaded = LinearStacker.load(path)
    assert reloaded.feature_order == feature_order
    np.testing.assert_array_equal(original.predict(x), reloaded.predict(x))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_stacking.py -v`
Expected: FAIL — either `feature_order` is not a constructor arg yet, or `save` doesn't exist.

- [ ] **Step 3: Implement save/load + feature_order on LinearStacker**

Modify `src/quant_research_stack/alpha/stacking.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from numpy.typing import NDArray


class LinearStacker:
    """Ridge-regularised linear stacker over base-model OOF predictions."""

    def __init__(self, alpha: float, feature_order: Sequence[str] | None = None) -> None:
        self._alpha = float(alpha)
        self._feature_order: list[str] = list(feature_order) if feature_order is not None else []
        self._weights: NDArray[np.float64] | None = None
        self._intercept: float = 0.0

    @property
    def feature_order(self) -> list[str]:
        return list(self._feature_order)

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        # weighted ridge closed form
        w_sqrt = np.sqrt(np.maximum(weights, 0.0))
        xw = x * w_sqrt[:, None]
        yw = y * w_sqrt
        a = xw.T @ xw + self._alpha * np.eye(x.shape[1])
        b = xw.T @ yw
        self._weights = np.linalg.solve(a, b)
        self._intercept = 0.0

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._weights is None:
            raise RuntimeError("LinearStacker.predict called before fit/load")
        return (x @ self._weights + self._intercept).astype(np.float64)

    def weights(self) -> NDArray[np.float64]:
        if self._weights is None:
            raise RuntimeError("LinearStacker.weights called before fit/load")
        return self._weights.copy()

    def save(self, path: Path) -> None:
        if self._weights is None:
            raise RuntimeError("cannot save un-fitted LinearStacker")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "weights": self._weights.tolist(),
                "feature_order": list(self._feature_order),
                "intercept": float(self._intercept),
                "alpha": float(self._alpha),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "LinearStacker":
        path = Path(path)
        payload = joblib.load(path)
        inst = cls(alpha=payload["alpha"], feature_order=payload["feature_order"])
        inst._weights = np.asarray(payload["weights"], dtype=np.float64)
        inst._intercept = float(payload["intercept"])
        return inst
```

If the existing `LinearStacker` already implements `fit`/`predict` and is consumed elsewhere, preserve those signatures exactly and only **add** `feature_order`, `save`, `load`. Confirm with `git grep "LinearStacker"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_stacking.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/stacking.py tests/test_alpha_stacking.py
git commit -m "feat(s0): LinearStacker.save/load with feature_order"
```

---

### Task 10: `tests/conftest.py` with `synthetic_js` fixture

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_alpha_persistence.py` (creating the file if it doesn't exist yet):

```python
import polars as pl


def test_synthetic_js_fixture_shape(synthetic_js):
    df = synthetic_js
    assert isinstance(df, pl.DataFrame)
    assert df.height == 10_000
    # 50 feature columns + date_id + responder_6 + weight
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    assert len(feature_cols) == 50
    assert "responder_6" in df.columns
    assert "weight" in df.columns
    assert "date_id" in df.columns


def test_synthetic_js_fixture_deterministic(synthetic_js):
    # Same fixture invocation twice within a session must give identical content.
    # (pytest caches fixture results per scope; this asserts the underlying generator is seeded.)
    first = synthetic_js
    second = synthetic_js
    assert first.equals(second)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_persistence.py -v`
Expected: FAIL — `fixture 'synthetic_js' not found`.

- [ ] **Step 3: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures for the alpha test suite."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture(scope="session")
def synthetic_js() -> pl.DataFrame:
    """Deterministic JS-shaped 10k-row Polars DataFrame.

    Layout:
      - date_id     : monotonic int64 in [0, 1000)
      - feature_00  : informative (linear with target)
      - feature_01  : informative (linear with target, negative weight)
      - feature_02..49 : Gaussian noise
      - weight      : float, in (0, 2]
      - responder_6 : target — linear combo of informative features + noise
    """
    rng = np.random.default_rng(seed=20260521)
    n_rows = 10_000
    n_features = 50

    feature_matrix = rng.standard_normal((n_rows, n_features)).astype(np.float64)
    # Two informative features.
    informative_w = np.array([0.4, -0.3], dtype=np.float64)
    signal = feature_matrix[:, :2] @ informative_w
    noise = 0.5 * rng.standard_normal(n_rows)
    target = (signal + noise).astype(np.float64)

    date_id = (np.arange(n_rows) // 10).astype(np.int64)  # 10 rows per date_id
    weight = (0.5 + rng.uniform(size=n_rows)).astype(np.float64)

    cols: dict[str, np.ndarray] = {f"feature_{i:02d}": feature_matrix[:, i] for i in range(n_features)}
    cols["date_id"] = date_id
    cols["responder_6"] = target
    cols["weight"] = weight

    return pl.DataFrame(cols)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_persistence.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_alpha_persistence.py
git commit -m "test(s0): tests/conftest.py exposes deterministic synthetic_js fixture"
```

---

### Task 11: `inference.load_predictor_from_run` + `_BoundStackPredictor`

**Files:**
- Modify: `src/quant_research_stack/alpha/inference.py`
- Modify: `tests/test_alpha_persistence.py`

This task wires the loader + predictor together. It depends on Tasks 2–9.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_alpha_persistence.py`:

```python
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest
import torch

from quant_research_stack.alpha.exceptions import (
    ArtifactsMissingError,
    FeatureSchemaError,
)
from quant_research_stack.alpha.inference import (
    _canonical_sha256,
    load_predictor_from_run,
)
from quant_research_stack.alpha.models.catboost_model import CatBoostAlphaModel, CatBoostConfig
from quant_research_stack.alpha.models.lightgbm_model import LightGBMAlphaModel, LightGBMConfig
from quant_research_stack.alpha.models.mlp import MLPAlphaModel, MLPConfig
from quant_research_stack.alpha.models.ridge import RidgeAlphaModel, RidgeConfig
from quant_research_stack.alpha.models.sequence import Conv1DAlphaModel, Conv1DConfig
from quant_research_stack.alpha.models.xgboost_model import XGBoostAlphaModel, XGBoostConfig
from quant_research_stack.alpha.stacking import LinearStacker


def _build_minimal_run(run_dir: Path) -> list[str]:
    """Construct a minimal valid run directory: 6 base models, stacker, feature_cols.json."""
    rng = np.random.default_rng(0)
    n = 300
    n_features = 8
    x_tr = rng.standard_normal((n, n_features))
    y_tr = x_tr[:, 0] + 0.1 * rng.standard_normal(n)
    w_tr = np.ones(n)
    x_val = rng.standard_normal((50, n_features))
    y_val = x_val[:, 0]
    w_val = np.ones(50)

    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    RidgeAlphaModel(RidgeConfig(alpha=1.0)).fit(x_tr, y_tr, w_tr).save(models_dir / "ridge.joblib")  # type: ignore[func-returns-value]
    # Some classes don't chain; build, fit, save separately:
    r = RidgeAlphaModel(RidgeConfig(alpha=1.0)); r.fit(x_tr, y_tr, w_tr); r.save(models_dir / "ridge.joblib")
    l = LightGBMAlphaModel(LightGBMConfig(num_leaves=7, max_depth=3, learning_rate=0.1,
                                          n_estimators=20, early_stopping_rounds=5,
                                          feature_fraction=1.0, bagging_fraction=1.0))
    l.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val); l.save(models_dir / "lightgbm.txt")
    xg = XGBoostAlphaModel(XGBoostConfig(max_depth=3, learning_rate=0.1, n_estimators=20,
                                         early_stopping_rounds=5, tree_method="hist"))
    xg.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val); xg.save(models_dir / "xgboost.json")
    cb = CatBoostAlphaModel(CatBoostConfig(depth=3, learning_rate=0.1, n_estimators=20,
                                           early_stopping_rounds=5))
    cb.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val); cb.save(models_dir / "catboost.cbm")
    mp = MLPAlphaModel(MLPConfig(hidden_dims=[8], dropout=0.0, learning_rate=1e-3,
                                 batch_size=64, max_epochs=2, patience=2,
                                 mixed_precision=False))
    mp.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val); mp.save(models_dir / "mlp.pt")
    seq = Conv1DAlphaModel(Conv1DConfig(kernel_sizes=[3], channels=[8], dropout=0.0,
                                        learning_rate=1e-3, batch_size=64, max_epochs=2,
                                        patience=2, random_state=0))
    seq.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val); seq.save(models_dir / "sequence.pt")

    feature_order = ["ridge", "lgb", "xgb", "cat", "mlp", "seq"]
    # Tiny stacker — fit on the val slice with all 6 outputs.
    stack_x = np.column_stack([
        r.predict(x_val), l.predict(x_val), xg.predict(x_val),
        cb.predict(x_val), mp.predict(x_val), seq.predict(x_val),
    ])
    stacker = LinearStacker(alpha=1e-3, feature_order=feature_order)
    stacker.fit(stack_x, y_val, w_val)
    stacker.save(models_dir / "stacker.joblib")

    feature_cols = [f"feature_{i:02d}" for i in range(n_features)]
    sha = _canonical_sha256(feature_cols)
    (run_dir / "feature_cols.json").write_text(json.dumps({
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "feature_cols_sha256": sha,
        "target_column": "responder_6",
        "weight_column": "weight",
        "group_column": "date_id",
    }, indent=2))
    return feature_cols


def test_canonical_sha256_is_order_sensitive():
    a = _canonical_sha256(["x", "y", "z"])
    b = _canonical_sha256(["x", "z", "y"])
    assert a != b


def test_canonical_sha256_is_whitespace_independent_in_inputs():
    # The function takes a list, not a string — so trailing whitespace in a name DOES change the sha.
    # This test pins that the canonicalisation does not silently strip whitespace.
    a = _canonical_sha256(["x"])
    b = _canonical_sha256(["x "])
    assert a != b


def test_load_predictor_from_run_happy_path(tmp_path):
    feature_cols = _build_minimal_run(tmp_path)
    predictor = load_predictor_from_run(tmp_path)
    assert sorted(predictor.expected_feature_columns) == sorted(feature_cols)

    # predict on a one-row DataFrame
    row = pl.DataFrame({c: [0.5] for c in feature_cols})
    pred, conf = predictor.predict(row)
    assert isinstance(pred, float)
    assert 0.0 <= conf <= 1.0


def test_load_predictor_from_run_rejects_pre_s0(tmp_path):
    # Only stacker.joblib exists, no other artifacts — pre-S0 layout.
    (tmp_path / "models").mkdir()
    joblib.dump({"weights": [0.2] * 6, "feature_order": ["ridge","lgb","xgb","cat","mlp","seq"],
                 "intercept": 0.0, "alpha": 1e-3},
                tmp_path / "models" / "stacker.joblib")
    with pytest.raises(ArtifactsMissingError):
        load_predictor_from_run(tmp_path)


def test_load_predictor_from_run_detects_sha_tamper(tmp_path):
    feature_cols = _build_minimal_run(tmp_path)
    schema_path = tmp_path / "feature_cols.json"
    schema = json.loads(schema_path.read_text())
    schema["feature_cols_sha256"] = "0" * 64
    schema_path.write_text(json.dumps(schema))
    with pytest.raises(FeatureSchemaError, match="sha256 mismatch"):
        load_predictor_from_run(tmp_path)


def test_predictor_rejects_missing_columns(tmp_path):
    _build_minimal_run(tmp_path)
    predictor = load_predictor_from_run(tmp_path)
    bad = pl.DataFrame({"feature_00": [0.1], "feature_01": [0.2]})
    with pytest.raises(FeatureSchemaError):
        predictor.predict(bad)


def test_predictor_reorders_columns(tmp_path):
    feature_cols = _build_minimal_run(tmp_path)
    predictor = load_predictor_from_run(tmp_path)
    in_order = pl.DataFrame({c: [0.5] for c in feature_cols})
    shuffled_cols = list(reversed(feature_cols))
    shuffled = pl.DataFrame({c: [0.5] for c in shuffled_cols})
    np.testing.assert_array_equal(
        np.array(predictor.predict(in_order)),
        np.array(predictor.predict(shuffled)),
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_persistence.py -v`
Expected: imports fail — `_canonical_sha256`, `load_predictor_from_run` not defined.

- [ ] **Step 3: Implement loader infrastructure in `inference.py`**

Modify `src/quant_research_stack/alpha/inference.py` (preserving the existing `S1Predictor` Protocol + `_StackPredictor` + `build_predictor_from_stack`):

```python
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.alpha.exceptions import (
    ArtifactCorruptError,
    ArtifactsMissingError,
    FeatureSchemaError,
)
from quant_research_stack.alpha.models.catboost_model import CatBoostAlphaModel
from quant_research_stack.alpha.models.lightgbm_model import LightGBMAlphaModel
from quant_research_stack.alpha.models.mlp import MLPAlphaModel
from quant_research_stack.alpha.models.ridge import RidgeAlphaModel
from quant_research_stack.alpha.models.sequence import Conv1DAlphaModel
from quant_research_stack.alpha.models.xgboost_model import XGBoostAlphaModel
from quant_research_stack.alpha.stacking import LinearStacker


# ... existing S1Predictor / _StackPredictor / build_predictor_from_stack stay above ...


_EXPECTED_BASE_MODEL_FILES: dict[str, str] = {
    "ridge": "ridge.joblib",
    "lgb":   "lightgbm.txt",
    "xgb":   "xgboost.json",
    "cat":   "catboost.cbm",
    "mlp":   "mlp.pt",
    "seq":   "sequence.pt",
}

_EXPECTED_FEATURE_ORDER: tuple[str, ...] = ("ridge", "lgb", "xgb", "cat", "mlp", "seq")


def _canonical_sha256(feature_columns: list[str]) -> str:
    """Canonical, order-sensitive sha256 over the feature-column list."""
    payload = json.dumps(list(feature_columns), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _assert_all_artifacts_present(run_dir: Path) -> None:
    schema_path = run_dir / "feature_cols.json"
    if not schema_path.exists():
        raise ArtifactsMissingError(f"missing feature_cols.json in {run_dir}")
    for filename in _EXPECTED_BASE_MODEL_FILES.values():
        p = run_dir / "models" / filename
        if not p.exists():
            raise ArtifactsMissingError(f"missing {p}")
    if not (run_dir / "models" / "stacker.joblib").exists():
        raise ArtifactsMissingError(f"missing {run_dir / 'models' / 'stacker.joblib'}")


def _load_and_verify_schema(schema_path: Path) -> dict:
    try:
        schema = json.loads(schema_path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactCorruptError(f"feature_cols.json is not valid JSON: {exc}") from exc
    expected = schema["feature_cols_sha256"]
    computed = _canonical_sha256(schema["feature_columns"])
    if expected != computed:
        raise FeatureSchemaError(
            f"feature_cols.json sha256 mismatch in {schema_path}: stored={expected} computed={computed}"
        )
    return schema


@dataclass
class _BoundStackPredictor:
    base_models: dict[str, object]              # keyed by feature_order names
    stacker: LinearStacker
    feature_columns: list[str]

    @property
    def expected_feature_columns(self) -> list[str]:
        return list(self.feature_columns)

    def predict(self, row: pl.DataFrame) -> tuple[float, float]:
        if row.height != 1:
            raise ValueError(f"S1 predicts one row at a time; got height={row.height}")
        missing = set(self.feature_columns) - set(row.columns)
        if missing:
            raise FeatureSchemaError(
                f"caller passed DataFrame missing required columns: {sorted(missing)}"
            )
        x = row.select(self.feature_columns).to_numpy()[0]
        base_outs = np.empty(len(self.stacker.feature_order), dtype=np.float64)
        for i, name in enumerate(self.stacker.feature_order):
            model = self.base_models[name]
            base_outs[i] = float(model.predict(x.reshape(1, -1))[0])  # type: ignore[attr-defined]
        pred = float(self.stacker.predict(base_outs.reshape(1, -1))[0])
        # Confidence reuses _StackPredictor's agreement formula.
        signs = np.sign(base_outs)
        if signs.size == 0 or float(np.std(base_outs)) == 0.0:
            conf = 1.0
        else:
            mean_sign = np.sign(np.mean(signs))
            conf = float(np.clip(np.mean(signs == mean_sign), 0.0, 1.0))
        return pred, conf


def load_predictor_from_run(run_dir: Path) -> _BoundStackPredictor:
    """Reconstruct an S1Predictor from a persisted training run.

    Raises:
        ArtifactsMissingError  — run_dir lacks one or more required S0 artifacts.
        FeatureSchemaError     — feature_cols.json sha256 doesn't match its contents.
        ArtifactCorruptError   — any of the model files fails to load.
    """
    run_dir = Path(run_dir)
    _assert_all_artifacts_present(run_dir)
    schema = _load_and_verify_schema(run_dir / "feature_cols.json")

    models_dir = run_dir / "models"
    try:
        base_models: dict[str, object] = {
            "ridge": RidgeAlphaModel.load(models_dir / "ridge.joblib"),
            "lgb":   LightGBMAlphaModel.load(models_dir / "lightgbm.txt"),
            "xgb":   XGBoostAlphaModel.load(models_dir / "xgboost.json"),
            "cat":   CatBoostAlphaModel.load(models_dir / "catboost.cbm"),
            "mlp":   MLPAlphaModel.load(models_dir / "mlp.pt"),
            "seq":   Conv1DAlphaModel.load(models_dir / "sequence.pt"),
        }
        stacker = LinearStacker.load(models_dir / "stacker.joblib")
    except FileNotFoundError as exc:
        raise ArtifactsMissingError(str(exc)) from exc
    except Exception as exc:
        raise ArtifactCorruptError(f"failed to load a model artifact under {models_dir}: {exc}") from exc

    if tuple(stacker.feature_order) != _EXPECTED_FEATURE_ORDER:
        raise FeatureSchemaError(
            f"stacker.feature_order={stacker.feature_order} != expected {_EXPECTED_FEATURE_ORDER}"
        )

    return _BoundStackPredictor(
        base_models=base_models,
        stacker=stacker,
        feature_columns=list(schema["feature_columns"]),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_persistence.py -v`
Expected: all green (8 tests including the conftest fixture tests from Task 10).

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/inference.py tests/test_alpha_persistence.py
git commit -m "feat(s0): inference.load_predictor_from_run + _BoundStackPredictor with sha256 + reorder"
```

---

### Task 12: `training.py` scaffold — `TrainConfig`, `RunResult`, Pydantic loaders

**Files:**
- Create: `src/quant_research_stack/alpha/training.py`
- Modify: `tests/test_alpha_training.py` (created here for the first time)

- [ ] **Step 1: Write the failing test**

Create `tests/test_alpha_training.py`:

```python
from __future__ import annotations

import yaml

from quant_research_stack.alpha.training import (
    CVConfig,
    DataConfig,
    FeatureConfig,
    ModelsConfig,
    TrainConfig,
)


def test_train_config_from_yaml_smoke():
    cfg_dict = {
        "data": {
            "jane_street_root": "data/raw/huggingface/TnnnT0326__Jane_Street_Competition",
            "synthetic_root": "data/raw/kaggle/datasets/christoffer__synthetic-jane-street-dataset",
            "preprocessed_alt_root": "data/raw/kaggle/datasets/saurabhshahane__jane-street-preprocessed-train",
            "group_column": "date_id",
            "target_column": "responder_6",
            "weight_column": "weight",
            "max_rows": 2_000_000,
            "permanent_holdout_fraction": 0.2,
        },
        "cv": {
            "n_folds": 3,
            "purge_days": 5,
            "embargo_days": 5,
            "random_seed": 42,
        },
        "features": {
            "lag_windows": [1, 5],
            "rolling_windows": [5, 20],
            "cross_sectional_ranks": False,
            "include_noise_feature": True,
            "noise_seed": 42,
        },
        "models": {
            "ridge": {"alpha": 1.0},
            "lightgbm": {
                "num_leaves": 31, "max_depth": -1, "learning_rate": 0.05,
                "n_estimators": 200, "early_stopping_rounds": 20,
                "feature_fraction": 0.9, "bagging_fraction": 0.9,
            },
            "xgboost": {
                "max_depth": 6, "learning_rate": 0.05, "n_estimators": 200,
                "early_stopping_rounds": 20, "tree_method": "hist",
            },
            "catboost": {
                "depth": 6, "learning_rate": 0.05, "n_estimators": 200,
                "early_stopping_rounds": 20,
            },
            "mlp": {
                "hidden_dims": [64, 32], "dropout": 0.2, "learning_rate": 1e-3,
                "batch_size": 1024, "max_epochs": 30, "patience": 3,
                "mixed_precision": False,
            },
            "sequence": {
                "kernel_sizes": [3, 5], "channels": [16, 32], "dropout": 0.1,
                "learning_rate": 1e-3, "batch_size": 1024, "max_epochs": 30,
                "patience": 3, "random_state": 0,
            },
        },
        "stacker_alpha": 1e-3,
        "streaming": False,
        "max_rows_streaming": 5_000_000,
    }
    cfg = TrainConfig.from_dict(cfg_dict)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.cv, CVConfig)
    assert isinstance(cfg.features, FeatureConfig)
    assert isinstance(cfg.models, ModelsConfig)
    assert cfg.cv.n_folds == 3
    assert cfg.models.ridge.alpha == 1.0
    assert cfg.streaming is False


def test_train_config_rejects_bad_alpha():
    bad = {"alpha": -1.0}  # alpha must be > 0
    import pytest

    with pytest.raises(Exception):
        from quant_research_stack.alpha.training import RidgeModelConfig
        RidgeModelConfig(**bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_training.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant_research_stack.alpha.training'`.

- [ ] **Step 3: Create `src/quant_research_stack/alpha/training.py` with the scaffold**

```python
"""Unified S1 training pipeline.

Public surface:
    train_s1(config: TrainConfig, registry: RunRegistry) -> RunResult

All Pydantic configs validate at construction time. The public function is pure
in the sense that it takes a config + a registry and returns a RunResult — no
global state, no CLI argument parsing, no console output beyond progress logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# -----------------------------------------------------------------------------
# Configs (Pydantic v2)
# -----------------------------------------------------------------------------


class DataConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    jane_street_root: str
    synthetic_root: str | None = None
    preprocessed_alt_root: str | None = None
    group_column: str = "date_id"
    target_column: str = "responder_6"
    weight_column: str = "weight"
    max_rows: int = Field(gt=0)
    permanent_holdout_fraction: float = Field(ge=0.05, le=0.5)


class CVConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    n_folds: int = Field(ge=2, le=20)
    purge_days: int = Field(ge=0)
    embargo_days: int = Field(ge=0)
    random_seed: int = 42


class FeatureConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    lag_windows: list[int] = Field(default_factory=lambda: [1, 5])
    rolling_windows: list[int] = Field(default_factory=lambda: [5, 20])
    cross_sectional_ranks: bool = False
    include_noise_feature: bool = True
    noise_seed: int = 42


class RidgeModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    alpha: float = Field(gt=0.0)


class LightGBMModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    num_leaves: int = Field(gt=1)
    max_depth: int
    learning_rate: float = Field(gt=0.0, le=1.0)
    n_estimators: int = Field(gt=0)
    early_stopping_rounds: int = Field(ge=0)
    feature_fraction: float = Field(gt=0.0, le=1.0)
    bagging_fraction: float = Field(gt=0.0, le=1.0)


class XGBoostModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    max_depth: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0, le=1.0)
    n_estimators: int = Field(gt=0)
    early_stopping_rounds: int = Field(ge=0)
    tree_method: Literal["hist", "approx", "exact"] = "hist"


class CatBoostModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    depth: int = Field(gt=0, le=16)
    learning_rate: float = Field(gt=0.0, le=1.0)
    n_estimators: int = Field(gt=0)
    early_stopping_rounds: int = Field(ge=0)


class MLPModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    hidden_dims: list[int]
    dropout: float = Field(ge=0.0, lt=1.0)
    learning_rate: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(ge=0)
    mixed_precision: bool = False


class SequenceModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    kernel_sizes: list[int]
    channels: list[int]
    dropout: float = Field(ge=0.0, lt=1.0)
    learning_rate: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(ge=0)
    random_state: int = 0

    @model_validator(mode="after")
    def _check_shapes(self) -> "SequenceModelConfig":
        if len(self.kernel_sizes) != len(self.channels):
            raise ValueError("kernel_sizes and channels must have the same length")
        return self


class ModelsConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    ridge: RidgeModelConfig
    lightgbm: LightGBMModelConfig
    xgboost: XGBoostModelConfig
    catboost: CatBoostModelConfig
    mlp: MLPModelConfig
    sequence: SequenceModelConfig


class TrainConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    data: DataConfig
    cv: CVConfig
    features: FeatureConfig
    models: ModelsConfig
    stacker_alpha: float = Field(gt=0.0)
    streaming: bool = False
    max_rows_streaming: int = Field(gt=0, default=5_000_000)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainConfig":
        return cls.model_validate(payload)


# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    fold_metrics: list[dict[str, float]]
    holdout_weighted_zero_mean_r2: float
    n_features_after_adversarial: int
    n_features_after_noise_floor: int
    base_models_persisted: list[str]
    stacker_path: Path
    feature_cols_path: Path


# Placeholder; real implementation lands in Task 13 + 14.
def train_s1(config: TrainConfig, registry: object) -> RunResult:
    raise NotImplementedError("train_s1 is implemented in Task 13 + 14")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_training.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/training.py tests/test_alpha_training.py
git commit -m "feat(s0): training.py scaffold — TrainConfig + RunResult Pydantic v2 surface"
```

---

### Task 13: `training._fit_one_fold` + stacker fit (phases 1–3)

**Files:**
- Modify: `src/quant_research_stack/alpha/training.py`
- Modify: `tests/test_alpha_training.py`

This task ports the existing per-fold logic from `alpha_train_s1_streaming.py` into the new module **with the sequence model added as the 6th base learner**. The trainer reads remain reusable; we are extracting + adding the 6th column.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_alpha_training.py`:

```python
import numpy as np

from quant_research_stack.alpha.training import (
    _fit_one_fold,
    ModelsConfig,
    RidgeModelConfig,
    LightGBMModelConfig,
    XGBoostModelConfig,
    CatBoostModelConfig,
    MLPModelConfig,
    SequenceModelConfig,
)


def _minimal_models_config() -> ModelsConfig:
    return ModelsConfig(
        ridge=RidgeModelConfig(alpha=1.0),
        lightgbm=LightGBMModelConfig(num_leaves=7, max_depth=3, learning_rate=0.1,
                                     n_estimators=20, early_stopping_rounds=5,
                                     feature_fraction=1.0, bagging_fraction=1.0),
        xgboost=XGBoostModelConfig(max_depth=3, learning_rate=0.1, n_estimators=20,
                                   early_stopping_rounds=5, tree_method="hist"),
        catboost=CatBoostModelConfig(depth=3, learning_rate=0.1, n_estimators=20,
                                     early_stopping_rounds=5),
        mlp=MLPModelConfig(hidden_dims=[8], dropout=0.0, learning_rate=1e-3,
                           batch_size=64, max_epochs=2, patience=2,
                           mixed_precision=False),
        sequence=SequenceModelConfig(kernel_sizes=[3], channels=[8], dropout=0.0,
                                     learning_rate=1e-3, batch_size=64,
                                     max_epochs=2, patience=2, random_state=0),
    )


def test_fit_one_fold_returns_six_base_predictions():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 8))
    y = x[:, 0] + 0.1 * rng.standard_normal(500)
    w = np.ones(500)
    tr_idx = np.arange(0, 400)
    te_idx = np.arange(400, 500)
    cfg = _minimal_models_config()

    fold_oof = _fit_one_fold(
        fold_idx=0,
        x_tr=x[tr_idx], y_tr=y[tr_idx], w_tr=w[tr_idx],
        x_te=x[te_idx], y_te=y[te_idx], w_te=w[te_idx],
        models_config=cfg,
    )

    assert set(fold_oof.keys()) == {"ridge", "lgb", "xgb", "cat", "mlp", "seq"}
    for name, preds in fold_oof.items():
        assert preds.shape == (te_idx.size,), f"{name} returned shape {preds.shape}"
        assert preds.dtype == np.float64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_training.py::test_fit_one_fold_returns_six_base_predictions -v`
Expected: FAIL — `ImportError: cannot import name '_fit_one_fold'`.

- [ ] **Step 3: Implement `_fit_one_fold` + `_fit_stacker`**

Append to `src/quant_research_stack/alpha/training.py`:

```python
import numpy as np
from numpy.typing import NDArray

from quant_research_stack.alpha.models.catboost_model import CatBoostAlphaModel, CatBoostConfig
from quant_research_stack.alpha.models.lightgbm_model import LightGBMAlphaModel, LightGBMConfig
from quant_research_stack.alpha.models.mlp import MLPAlphaModel, MLPConfig
from quant_research_stack.alpha.models.ridge import RidgeAlphaModel, RidgeConfig
from quant_research_stack.alpha.models.sequence import Conv1DAlphaModel, Conv1DConfig
from quant_research_stack.alpha.models.xgboost_model import XGBoostAlphaModel, XGBoostConfig
from quant_research_stack.alpha.stacking import LinearStacker


_BASE_MODEL_NAMES: tuple[str, ...] = ("ridge", "lgb", "xgb", "cat", "mlp", "seq")


def _fit_one_fold(
    *,
    fold_idx: int,
    x_tr: NDArray[np.float64],
    y_tr: NDArray[np.float64],
    w_tr: NDArray[np.float64],
    x_te: NDArray[np.float64],
    y_te: NDArray[np.float64],
    w_te: NDArray[np.float64],
    models_config: ModelsConfig,
) -> dict[str, NDArray[np.float64]]:
    """Fit all 6 base models on (x_tr, y_tr, w_tr); predict each on (x_te) and return OOF.

    Returns a dict keyed by base-model name with each value an (n_te,) float64 array.
    Per-fold models are NOT persisted (consistent with pre-S0 behaviour); only the
    refit-on-full models in phase 4 land on disk.
    """
    out: dict[str, NDArray[np.float64]] = {}

    ridge = RidgeAlphaModel(RidgeConfig(alpha=models_config.ridge.alpha))
    ridge.fit(x_tr, y_tr, w_tr)
    out["ridge"] = ridge.predict(x_te).astype(np.float64)

    lcfg = models_config.lightgbm
    lgb = LightGBMAlphaModel(LightGBMConfig(
        num_leaves=lcfg.num_leaves, max_depth=lcfg.max_depth,
        learning_rate=lcfg.learning_rate, n_estimators=lcfg.n_estimators,
        early_stopping_rounds=lcfg.early_stopping_rounds,
        feature_fraction=lcfg.feature_fraction, bagging_fraction=lcfg.bagging_fraction,
    ))
    lgb.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
    out["lgb"] = lgb.predict(x_te).astype(np.float64)

    xcfg = models_config.xgboost
    xgb = XGBoostAlphaModel(XGBoostConfig(
        max_depth=xcfg.max_depth, learning_rate=xcfg.learning_rate,
        n_estimators=xcfg.n_estimators, early_stopping_rounds=xcfg.early_stopping_rounds,
        tree_method=xcfg.tree_method,
    ))
    xgb.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
    out["xgb"] = xgb.predict(x_te).astype(np.float64)

    ccfg = models_config.catboost
    cat = CatBoostAlphaModel(CatBoostConfig(
        depth=ccfg.depth, learning_rate=ccfg.learning_rate,
        n_estimators=ccfg.n_estimators, early_stopping_rounds=ccfg.early_stopping_rounds,
    ))
    cat.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
    out["cat"] = cat.predict(x_te).astype(np.float64)

    mcfg = models_config.mlp
    mlp = MLPAlphaModel(MLPConfig(
        hidden_dims=list(mcfg.hidden_dims), dropout=mcfg.dropout,
        learning_rate=mcfg.learning_rate, batch_size=mcfg.batch_size,
        max_epochs=mcfg.max_epochs, patience=mcfg.patience,
        mixed_precision=mcfg.mixed_precision,
    ))
    mlp.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
    out["mlp"] = mlp.predict(x_te).astype(np.float64)

    scfg = models_config.sequence
    seq = Conv1DAlphaModel(Conv1DConfig(
        kernel_sizes=list(scfg.kernel_sizes), channels=list(scfg.channels),
        dropout=scfg.dropout, learning_rate=scfg.learning_rate,
        batch_size=scfg.batch_size, max_epochs=scfg.max_epochs,
        patience=scfg.patience, random_state=scfg.random_state,
    ))
    seq.fit(x_tr, y_tr, w_tr, x_te, y_te, w_te)
    out["seq"] = seq.predict(x_te).astype(np.float64)

    return out


def _fit_stacker(
    *,
    oof_by_name: dict[str, NDArray[np.float64]],
    y_full: NDArray[np.float64],
    w_full: NDArray[np.float64],
    stacker_alpha: float,
) -> LinearStacker:
    """Stack 6 OOF columns in the canonical _BASE_MODEL_NAMES order and fit LinearStacker."""
    stack_x = np.column_stack([oof_by_name[n] for n in _BASE_MODEL_NAMES])
    stacker = LinearStacker(alpha=stacker_alpha, feature_order=list(_BASE_MODEL_NAMES))
    stacker.fit(stack_x, y_full, w_full)
    return stacker
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_training.py -v`
Expected: all tests in this file pass, including the new `_fit_one_fold` test.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/training.py tests/test_alpha_training.py
git commit -m "feat(s0): training._fit_one_fold + _fit_stacker — 6 base models, OOF returned"
```

---

### Task 14: `training._refit_on_full` + `_persist_run` + `train_s1` (phases 4–5)

**Files:**
- Modify: `src/quant_research_stack/alpha/training.py`
- Modify: `tests/test_alpha_training.py`

This task implements the **behaviour change** described in spec §6.5: the holdout stack uses all 6 final models instead of zeroing xgb/cat/mlp columns. It also implements `_persist_run` (writes feature_cols.json + 6 model files + extended sha256 index + updated stacker) and the public `train_s1` entry point.

- [ ] **Step 1: Write the failing end-to-end test**

Append to `tests/test_alpha_training.py`:

```python
import json
from pathlib import Path

import numpy as np

from quant_research_stack.alpha.inference import (
    _EXPECTED_BASE_MODEL_FILES,
    load_predictor_from_run,
)
from quant_research_stack.alpha.registry import RunRegistry
from quant_research_stack.alpha.training import train_s1, TrainConfig


def _train_config_for_synthetic(tmp_path: Path) -> dict:
    return {
        "data": {
            "jane_street_root": str(tmp_path),  # ignored by training when synthetic_dataframe is passed
            "group_column": "date_id",
            "target_column": "responder_6",
            "weight_column": "weight",
            "max_rows": 10_000,
            "permanent_holdout_fraction": 0.2,
        },
        "cv": {"n_folds": 2, "purge_days": 0, "embargo_days": 0, "random_seed": 0},
        "features": {
            "lag_windows": [], "rolling_windows": [],
            "cross_sectional_ranks": False,
            "include_noise_feature": False, "noise_seed": 0,
        },
        "models": {
            "ridge": {"alpha": 1.0},
            "lightgbm": {"num_leaves": 7, "max_depth": 3, "learning_rate": 0.1,
                         "n_estimators": 20, "early_stopping_rounds": 5,
                         "feature_fraction": 1.0, "bagging_fraction": 1.0},
            "xgboost": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 20,
                        "early_stopping_rounds": 5, "tree_method": "hist"},
            "catboost": {"depth": 3, "learning_rate": 0.1, "n_estimators": 20,
                         "early_stopping_rounds": 5},
            "mlp": {"hidden_dims": [8], "dropout": 0.0, "learning_rate": 1e-3,
                    "batch_size": 64, "max_epochs": 2, "patience": 2,
                    "mixed_precision": False},
            "sequence": {"kernel_sizes": [3], "channels": [8], "dropout": 0.0,
                         "learning_rate": 1e-3, "batch_size": 64,
                         "max_epochs": 2, "patience": 2, "random_state": 0},
        },
        "stacker_alpha": 1e-3,
        "streaming": False,
        "max_rows_streaming": 10_000,
    }


def test_train_s1_end_to_end_on_synthetic(synthetic_js, tmp_path):
    cfg = TrainConfig.from_dict(_train_config_for_synthetic(tmp_path))
    registry = RunRegistry(root=tmp_path / "experiments")
    result = train_s1(cfg, registry, synthetic_dataframe=synthetic_js)

    # Run dir exists with all expected files.
    assert result.run_dir.exists()
    for fname in _EXPECTED_BASE_MODEL_FILES.values():
        assert (result.run_dir / "models" / fname).exists(), f"missing {fname}"
    assert (result.run_dir / "models" / "stacker.joblib").exists()
    assert (result.run_dir / "feature_cols.json").exists()
    assert (result.run_dir / "_artifact_sha256.json").exists()
    assert (result.run_dir / "metrics.json").exists()
    assert (result.run_dir / "predictions.parquet").exists()

    # Loader returns a working predictor.
    predictor = load_predictor_from_run(result.run_dir)
    assert sorted(predictor.expected_feature_columns) == sorted(
        [c for c in synthetic_js.columns if c.startswith("feature_")]
    )

    # And R² is a real number (not NaN/inf), even if the synthetic data isn't calibrated to a gate.
    assert np.isfinite(result.holdout_weighted_zero_mean_r2)
    assert set(result.base_models_persisted) == {"ridge","lgb","xgb","cat","mlp","seq"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_training.py::test_train_s1_end_to_end_on_synthetic -v`
Expected: FAIL — `NotImplementedError: train_s1 is implemented in Task 13 + 14` (because Task 12's scaffold raises this).

- [ ] **Step 3: Implement `_refit_on_full`, `_persist_run`, and `train_s1`**

Append to `src/quant_research_stack/alpha/training.py`:

```python
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_research_stack.alpha.inference import _canonical_sha256
from quant_research_stack.alpha.metrics import weighted_zero_mean_r2
from quant_research_stack.alpha.registry import RunMetadata, RunRegistry


def _refit_on_full(
    *,
    x_full: NDArray[np.float64],
    y_full: NDArray[np.float64],
    w_full: NDArray[np.float64],
    models_config: ModelsConfig,
) -> dict[str, object]:
    """Refit each base model on the entire training slice. These are the models persisted to disk."""
    # Early-stopping eval set = last 1000 rows of train (matching pre-S0 streaming trainer pattern).
    eval_n = min(1000, x_full.shape[0] // 5)
    x_eval = x_full[-eval_n:]
    y_eval = y_full[-eval_n:]
    w_eval = w_full[-eval_n:]

    finals: dict[str, object] = {}

    ridge = RidgeAlphaModel(RidgeConfig(alpha=models_config.ridge.alpha))
    ridge.fit(x_full, y_full, w_full)
    finals["ridge"] = ridge

    lcfg = models_config.lightgbm
    lgb = LightGBMAlphaModel(LightGBMConfig(
        num_leaves=lcfg.num_leaves, max_depth=lcfg.max_depth,
        learning_rate=lcfg.learning_rate, n_estimators=lcfg.n_estimators,
        early_stopping_rounds=lcfg.early_stopping_rounds,
        feature_fraction=lcfg.feature_fraction, bagging_fraction=lcfg.bagging_fraction,
    ))
    lgb.fit(x_full, y_full, w_full, x_eval, y_eval, w_eval)
    finals["lgb"] = lgb

    xcfg = models_config.xgboost
    xgb = XGBoostAlphaModel(XGBoostConfig(
        max_depth=xcfg.max_depth, learning_rate=xcfg.learning_rate,
        n_estimators=xcfg.n_estimators, early_stopping_rounds=xcfg.early_stopping_rounds,
        tree_method=xcfg.tree_method,
    ))
    xgb.fit(x_full, y_full, w_full, x_eval, y_eval, w_eval)
    finals["xgb"] = xgb

    ccfg = models_config.catboost
    cat = CatBoostAlphaModel(CatBoostConfig(
        depth=ccfg.depth, learning_rate=ccfg.learning_rate,
        n_estimators=ccfg.n_estimators, early_stopping_rounds=ccfg.early_stopping_rounds,
    ))
    cat.fit(x_full, y_full, w_full, x_eval, y_eval, w_eval)
    finals["cat"] = cat

    mcfg = models_config.mlp
    mlp = MLPAlphaModel(MLPConfig(
        hidden_dims=list(mcfg.hidden_dims), dropout=mcfg.dropout,
        learning_rate=mcfg.learning_rate, batch_size=mcfg.batch_size,
        max_epochs=mcfg.max_epochs, patience=mcfg.patience,
        mixed_precision=mcfg.mixed_precision,
    ))
    mlp.fit(x_full, y_full, w_full, x_eval, y_eval, w_eval)
    finals["mlp"] = mlp

    scfg = models_config.sequence
    seq = Conv1DAlphaModel(Conv1DConfig(
        kernel_sizes=list(scfg.kernel_sizes), channels=list(scfg.channels),
        dropout=scfg.dropout, learning_rate=scfg.learning_rate,
        batch_size=scfg.batch_size, max_epochs=scfg.max_epochs,
        patience=scfg.patience, random_state=scfg.random_state,
    ))
    seq.fit(x_full, y_full, w_full, x_eval, y_eval, w_eval)
    finals["seq"] = seq

    return finals


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _persist_run(
    *,
    run_dir: Path,
    finals: dict[str, object],
    stacker: LinearStacker,
    feature_cols: list[str],
    data_config: DataConfig,
) -> None:
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    finals["ridge"].save(models_dir / "ridge.joblib")        # type: ignore[attr-defined]
    finals["lgb"].save(models_dir / "lightgbm.txt")          # type: ignore[attr-defined]
    finals["xgb"].save(models_dir / "xgboost.json")          # type: ignore[attr-defined]
    finals["cat"].save(models_dir / "catboost.cbm")          # type: ignore[attr-defined]
    finals["mlp"].save(models_dir / "mlp.pt")                # type: ignore[attr-defined]
    finals["seq"].save(models_dir / "sequence.pt")           # type: ignore[attr-defined]
    stacker.save(models_dir / "stacker.joblib")

    schema_path = run_dir / "feature_cols.json"
    schema_path.write_text(json.dumps({
        "feature_columns": list(feature_cols),
        "n_features": len(feature_cols),
        "feature_cols_sha256": _canonical_sha256(list(feature_cols)),
        "target_column": data_config.target_column,
        "weight_column": data_config.weight_column,
        "group_column": data_config.group_column,
    }, indent=2, sort_keys=True))

    # Extend _artifact_sha256.json (registry may have created it for metrics.json).
    sha_index_path = run_dir / "_artifact_sha256.json"
    sha_index: dict[str, str] = {}
    if sha_index_path.exists():
        sha_index = json.loads(sha_index_path.read_text())
    sha_index["feature_cols.json"] = _sha256_file(schema_path)
    for fname in ["ridge.joblib", "lightgbm.txt", "xgboost.json",
                  "catboost.cbm", "mlp.pt", "sequence.pt", "stacker.joblib"]:
        sha_index[f"models/{fname}"] = _sha256_file(models_dir / fname)
    sha_index_path.write_text(json.dumps(sha_index, indent=2, sort_keys=True))


def _holdout_eval(
    *,
    finals: dict[str, object],
    stacker: LinearStacker,
    x_h: NDArray[np.float64],
    y_h: NDArray[np.float64],
    w_h: NDArray[np.float64],
) -> tuple[float, dict[str, float], NDArray[np.float64]]:
    """Phase 5 — uses ALL 6 final models (no zeroing). Returns (R², per-model R², ensemble preds)."""
    per_model = {
        name: finals[name].predict(x_h).astype(np.float64)   # type: ignore[attr-defined]
        for name in _BASE_MODEL_NAMES
    }
    h_stack = np.column_stack([per_model[n] for n in stacker.feature_order])
    holdout_pred = stacker.predict(h_stack)
    holdout_r2 = float(weighted_zero_mean_r2(y_h, holdout_pred, w_h))
    per_model_r2 = {
        f"{name}_r2": float(weighted_zero_mean_r2(y_h, preds, w_h))
        for name, preds in per_model.items()
    }
    return holdout_r2, per_model_r2, holdout_pred


def _load_and_split(
    *,
    config: TrainConfig,
    synthetic_dataframe: pl.DataFrame | None,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Phase 1 — load + split + feature filter.

    If synthetic_dataframe is provided (tests), use it directly and bypass JS-on-disk loaders.
    """
    if synthetic_dataframe is not None:
        df = synthetic_dataframe
    else:
        # Real-data loader — port from alpha_train_s1_streaming.py:_streaming_load_and_filter.
        # Implementer reads the existing streaming loader and pastes it here verbatim,
        # then renames it _streaming_load. The non-streaming path uses a single read_parquet.
        from quant_research_stack.alpha.io import load_jane_street_panel

        df = load_jane_street_panel(
            root=Path(config.data.jane_street_root),
            max_rows=config.data.max_rows,
            group_column=config.data.group_column,
        )

    group_col = config.data.group_column
    target_col = config.data.target_column
    weight_col = config.data.weight_column

    groups = df[group_col].unique().sort()
    n_groups = groups.len()
    holdout_n = max(1, int(n_groups * config.data.permanent_holdout_fraction))
    holdout_groups = groups.tail(holdout_n)
    train_groups = groups.head(n_groups - holdout_n)

    train_df = df.filter(pl.col(group_col).is_in(train_groups))
    holdout_df = df.filter(pl.col(group_col).is_in(holdout_groups))

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        raise RuntimeError("no feature_* columns found in input frame")
    return train_df, holdout_df, feature_cols


def train_s1(
    config: TrainConfig,
    registry: RunRegistry,
    *,
    synthetic_dataframe: pl.DataFrame | None = None,
) -> RunResult:
    """End-to-end S1 training. Writes a complete run dir loadable by load_predictor_from_run."""
    train_df, holdout_df, feature_cols = _load_and_split(
        config=config, synthetic_dataframe=synthetic_dataframe,
    )

    target_col = config.data.target_column
    weight_col = config.data.weight_column

    x_full = train_df.select(feature_cols).to_numpy().astype(np.float64)
    y_full = train_df[target_col].to_numpy().astype(np.float64)
    w_full = train_df[weight_col].to_numpy().astype(np.float64)
    x_full = np.nan_to_num(x_full, nan=0.0)

    n = x_full.shape[0]
    fold_size = n // config.cv.n_folds

    oof: dict[str, NDArray[np.float64]] = {
        name: np.zeros(n, dtype=np.float64) for name in _BASE_MODEL_NAMES
    }
    fold_metrics: list[dict[str, float]] = []

    for fold_idx in range(config.cv.n_folds):
        te_start = fold_idx * fold_size
        te_end = (fold_idx + 1) * fold_size if fold_idx < config.cv.n_folds - 1 else n
        te_idx = np.arange(te_start, te_end)
        tr_idx = np.concatenate([np.arange(0, te_start), np.arange(te_end, n)])

        fold_oof = _fit_one_fold(
            fold_idx=fold_idx,
            x_tr=x_full[tr_idx], y_tr=y_full[tr_idx], w_tr=w_full[tr_idx],
            x_te=x_full[te_idx], y_te=y_full[te_idx], w_te=w_full[te_idx],
            models_config=config.models,
        )
        for name in _BASE_MODEL_NAMES:
            oof[name][te_idx] = fold_oof[name]

        fold_metrics.append({
            "fold": float(fold_idx),
            **{f"{name}_r2": float(weighted_zero_mean_r2(y_full[te_idx], fold_oof[name], w_full[te_idx]))
               for name in _BASE_MODEL_NAMES},
        })

    stacker = _fit_stacker(
        oof_by_name=oof, y_full=y_full, w_full=w_full,
        stacker_alpha=config.stacker_alpha,
    )

    finals = _refit_on_full(
        x_full=x_full, y_full=y_full, w_full=w_full,
        models_config=config.models,
    )

    x_h = holdout_df.select(feature_cols).to_numpy().astype(np.float64)
    y_h = holdout_df[target_col].to_numpy().astype(np.float64)
    w_h = holdout_df[weight_col].to_numpy().astype(np.float64)
    x_h = np.nan_to_num(x_h, nan=0.0)

    holdout_r2, per_model_r2, holdout_pred = _holdout_eval(
        finals=finals, stacker=stacker, x_h=x_h, y_h=y_h, w_h=w_h,
    )

    git_sha = _git_sha()
    meta = RunMetadata(
        version="0.2.0-s0",
        git_sha=git_sha,
        data_hashes={"jane_street_root": config.data.jane_street_root},
        hyperparams=config.model_dump(),
        fold_definition={"n_folds": config.cv.n_folds,
                         "purge": config.cv.purge_days,
                         "embargo": config.cv.embargo_days},
    )
    run_id = registry.create_run(meta)
    run_dir = Path(registry.root) / run_id

    metrics_payload = {
        "fold_metrics": fold_metrics,
        "holdout_weighted_zero_mean_r2": holdout_r2,
        "holdout_per_model_r2": per_model_r2,
        "n_features_after_adversarial": len(feature_cols),
        "n_features_after_noise_floor": len(feature_cols),  # adversarial/noise filter unchanged
        "training_rows": int(n),
        "holdout_rows": int(x_h.shape[0]),
        "profile": "s0_unified_full_holdout_refit",
    }
    registry.save_artifact(run_id, "metrics.json", json.dumps(metrics_payload, indent=2).encode())

    # predictions.parquet — holdout + OOF.
    preds_df = pl.DataFrame({
        "split": ["holdout"] * x_h.shape[0] + ["train_oof"] * n,
        "target_actual": np.concatenate([y_h, y_full]).astype(np.float32),
        "weight": np.concatenate([w_h, w_full]).astype(np.float32),
        "stacked": np.concatenate([
            holdout_pred.astype(np.float32),
            stacker.predict(np.column_stack([oof[n] for n in stacker.feature_order])).astype(np.float32),
        ]),
        **{name: np.concatenate([
              finals[name].predict(x_h).astype(np.float32),    # type: ignore[attr-defined]
              oof[name].astype(np.float32),
           ]) for name in _BASE_MODEL_NAMES},
    })
    preds_df.write_parquet(run_dir / "predictions.parquet")

    _persist_run(
        run_dir=run_dir, finals=finals, stacker=stacker,
        feature_cols=feature_cols, data_config=config.data,
    )

    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        fold_metrics=fold_metrics,
        holdout_weighted_zero_mean_r2=holdout_r2,
        n_features_after_adversarial=len(feature_cols),
        n_features_after_noise_floor=len(feature_cols),
        base_models_persisted=list(_BASE_MODEL_NAMES),
        stacker_path=run_dir / "models" / "stacker.joblib",
        feature_cols_path=run_dir / "feature_cols.json",
    )


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_alpha_training.py::test_train_s1_end_to_end_on_synthetic -v`
Expected: PASS — `1 passed`. (The test runs in ≤ 30s on the M4 because the fold counts + epoch budgets are tiny.)

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha/training.py tests/test_alpha_training.py
git commit -m "feat(s0): training._refit_on_full + _persist_run + train_s1 end-to-end (phases 4-5, honest holdout)"
```

---

### Task 15: Full whole-repo verification + ruff + mypy

**Files:** (no source changes; verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `PYTHONPATH=src uv run pytest -q`
Expected: every existing test plus the 8 new tests pass.

- [ ] **Step 2: Run ruff**

Run: `PYTHONPATH=src uv run ruff check src scripts tests`
Expected: `All checks passed.`

- [ ] **Step 3: Run mypy**

Run: `PYTHONPATH=src uv run mypy src`
Expected: `Success: no issues found in ...`. If any newly-added save/load methods have type issues (joblib returns `Any`), add narrowly-scoped `# type: ignore[...]` comments only where strictly needed; never silence whole-file checks.

- [ ] **Step 4: Commit (if any narrow type-ignore comments were added)**

If any type-ignore tweaks were needed in step 3:

```bash
git add -p src/quant_research_stack/alpha/
git commit -m "chore(s0): narrow type-ignore comments to keep mypy clean"
```

Otherwise skip — nothing to commit.

---

### Task 16: `scripts/train_s1.py` thin CLI

**Files:**
- Create: `scripts/train_s1.py`

- [ ] **Step 1: Write the file**

```python
"""Unified S1 training CLI. Replaces both alpha_train_s1.py and alpha_train_s1_streaming.py.

Usage:
    PYTHONPATH=src uv run python scripts/train_s1.py \
        --config configs/alpha.yaml \
        [--streaming] \
        [--max-rows N] \
        [--experiments-root experiments/alpha_s1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from rich.console import Console

from quant_research_stack.alpha.registry import RunRegistry
from quant_research_stack.alpha.training import TrainConfig, train_s1

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantLab S1 training (unified, post-S0)")
    p.add_argument("--config", default="configs/alpha.yaml")
    p.add_argument("--streaming", action="store_true",
                   help="memory-limited mode (M4 24 GB target)")
    p.add_argument("--max-rows", type=int, default=None,
                   help="override max_rows_streaming when --streaming is set")
    p.add_argument("--experiments-root", default="experiments/alpha_s1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_dict = yaml.safe_load(open(args.config))
    if args.streaming:
        cfg_dict["streaming"] = True
    if args.max_rows is not None:
        cfg_dict["max_rows_streaming"] = args.max_rows
    config = TrainConfig.from_dict(cfg_dict)

    registry = RunRegistry(root=Path(args.experiments_root))
    result = train_s1(config, registry)

    console.print(f"[bold green]Run complete:[/bold green] {result.run_dir}")
    console.print(f"  holdout weighted zero-mean R²: {result.holdout_weighted_zero_mean_r2:.6f}")
    console.print(f"  base models persisted: {result.base_models_persisted}")
    console.print(f"  stacker: {result.stacker_path}")
    console.print(f"  feature_cols: {result.feature_cols_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run the CLI's `--help`**

Run: `PYTHONPATH=src uv run python scripts/train_s1.py --help`
Expected: prints usage with the four flags above, exits 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_s1.py
git commit -m "feat(s0): scripts/train_s1.py — unified thin CLI replacing both old trainers"
```

---

### Task 17: Delete old trainers + rewrite `Makefile full-retrain-s1`

**Files:**
- Delete: `scripts/alpha_train_s1.py`
- Delete: `scripts/alpha_train_s1_streaming.py`
- Modify: `Makefile`

- [ ] **Step 1: Confirm no lingering callers of the old scripts**

Run: `git grep -nE "alpha_train_s1(_streaming)?\.py" -- ':!docs/superpowers/' ':!docs/architecture/'`
Expected: only references inside the new spec / plan, otherwise nothing. If the Makefile / docs / runbooks still mention the old names, capture them — they're updated below.

- [ ] **Step 2: Delete the old trainers**

```bash
git rm scripts/alpha_train_s1.py scripts/alpha_train_s1_streaming.py
```

- [ ] **Step 3: Rewrite the `Makefile` `train`/`train-streaming`/`full-retrain-s1` block**

Replace the existing block (lines 1–37 of the current `Makefile`, the S1 section) with:

```makefile
PY := PYTHONPATH=src uv run
TRAIN := scripts/train_s1.py
EXTRACT := scripts/alpha_extract_meta_features.py
OPTUNA := scripts/alpha_optuna_search.py
TRAIN_CONFIG ?= configs/alpha.yaml
TRAIN_MAX_ROWS ?=
STREAMING ?=
OPTUNA_ARGS ?= --n-trials 200

.PHONY: test lint type extract train optuna full-retrain-s1 clean-experiments

test:
	$(PY) pytest -q

lint:
	uv run ruff check src scripts tests

type:
	uv run mypy src

extract:
	$(PY) python $(EXTRACT)

train:
	$(PY) python $(TRAIN) --config $(TRAIN_CONFIG) \
	  $(if $(STREAMING),--streaming,) \
	  $(if $(TRAIN_MAX_ROWS),--max-rows $(TRAIN_MAX_ROWS),)

optuna:
	$(PY) python $(OPTUNA) $(OPTUNA_ARGS)

full-retrain-s1: test lint extract train optuna
	@echo "S1 full retrain complete. See experiments/alpha_s1/<latest>/metrics.json"
	@echo "Run `make verify-loader RUN_DIR=experiments/alpha_s1/<latest>` to confirm load_predictor_from_run succeeds."

verify-loader:
	$(PY) python -c "from pathlib import Path; from quant_research_stack.alpha.inference import load_predictor_from_run; p = load_predictor_from_run(Path('$(RUN_DIR)')); print('OK; n_features =', len(p.expected_feature_columns))"

clean-experiments:
	rm -rf experiments/alpha_s1/*
```

- [ ] **Step 4: Smoke-run the Makefile help / dry-run**

Run: `make train STREAMING=1 -n`
Expected: prints the resolved command with `--streaming` set.

Run: `make full-retrain-s1 -n`
Expected: shows the sequenced targets `test lint extract train optuna`.

- [ ] **Step 5: Commit**

```bash
git add Makefile
git commit -m "build(s0): Makefile full-retrain-s1 points at train_s1.py; STREAMING=1 passthrough; verify-loader target"
```

---

### Task 18: `CLAUDE.md` §13 update + runbook + operator retrain gate

**Files:**
- Modify: `CLAUDE.md`
- Create: `docs/runbooks/s0_retrain_after_persistence.md`

- [ ] **Step 1: Update `CLAUDE.md` §13's artifact block**

Find the `## 13. Completion criteria for the S1 milestone` section. Replace the `models/` block in the artifact listing with:

```text
metadata.json     git_sha, data_hashes, hyperparams, fold definition
predictions.parquet
metrics.json      weighted_zero_mean_r2 >= 0.012 on holdout
feature_importance.parquet
cv_folds.json
feature_cols.json ordered list + sha256
_artifact_sha256.json  sha256 over every artifact above
models/
  ridge.joblib
  lightgbm.txt
  lightgbm.config.json
  xgboost.json
  xgboost.config.json
  catboost.cbm
  catboost.config.json
  mlp.pt
  sequence.pt
  stacker.joblib
report.md
audit_log_smoke.jsonl    proof S1 wrote to the audit format expected by S4
```

The reproduction line stays — `make full-retrain-s1` (with `STREAMING=1` recommended for the M4 24 GB box).

- [ ] **Step 2: Create the operator runbook**

```bash
mkdir -p docs/runbooks
```

Create `docs/runbooks/s0_retrain_after_persistence.md`:

```markdown
# S0 Retrain After Persistence Lands

## When to run

After the S0 implementation lands on `quant-llm-implementation` and all unit tests are green, the operator must trigger a fresh S1 retrain so the first **loadable** run exists.

Before this runbook executes, `experiments/alpha_s1/` contains only `_archive_pre_s0/` (the old runs are preserved but not loadable by `load_predictor_from_run`).

## Command

On the M4 24 GB box:

```bash
cd /Users/dmr/MachineLearning
make full-retrain-s1 STREAMING=1 TRAIN_CONFIG=configs/alpha_5m.yaml TRAIN_MAX_ROWS=5000000
```

Expected wall-clock: **≤ 24h** per CLAUDE.md §8 budget (S1 base training).

## What to check during the run

- `top` shows the trainer's RSS staying under 20 GB (the `--streaming` flag is the load-bearing reason).
- Per-fold metrics print as each fold finishes; expect `lgb_r2`, `xgb_r2`, `cat_r2`, `mlp_r2`, `seq_r2` all populated (the new `seq_r2` column is the post-S0 addition).
- The final holdout R² line prints when phase 5 completes.

## Verification after completion

1. Confirm the artifact layout:

```bash
LATEST=$(ls -t experiments/alpha_s1 | grep -v _archive_pre_s0 | head -1)
ls "experiments/alpha_s1/$LATEST/models/"
```

Expected files: `ridge.joblib`, `lightgbm.txt`, `lightgbm.config.json`, `xgboost.json`, `xgboost.config.json`, `catboost.cbm`, `catboost.config.json`, `mlp.pt`, `sequence.pt`, `stacker.joblib`.

2. Verify the loader works:

```bash
make verify-loader RUN_DIR="experiments/alpha_s1/$LATEST"
```

Expected output: `OK; n_features = <number>`.

3. Confirm the §13 gate:

```bash
PYTHONPATH=src uv run python -c "import json, sys; m = json.load(open('experiments/alpha_s1/$LATEST/metrics.json')); print('holdout R^2 =', m['holdout_weighted_zero_mean_r2']); sys.exit(0 if m['holdout_weighted_zero_mean_r2'] >= 0.012 else 1)"
```

Expected: prints a positive R² and exits 0.

## Holdout R² — honest vs. zero'd (behaviour change)

Pre-S0 runs (the archived May-19 0.0955 result) computed the holdout R² with xgb/cat/mlp columns **zeroed** in the holdout stack matrix. The S0 trainer uses all 6 final base models in the holdout stack. Expect the new R² to differ — possibly slightly lower (the stacker weights were fit on OOF predictions that included all 5 / now 6 contributions, but pre-S0 holdout evaluation only saw 2 of them).

Record the new R² here when this runbook is first executed:

| Run id | Holdout R² (S0, all 6 models) | Notes |
|---|---|---|
| `<may-21-or-later>` | `<fill in>` | first post-S0 run |

If the new R² falls below the §13 gate of `0.012`, **do not pretend** by reverting to zero'd holdout — open a follow-up spec investigating which base model dropped, and tune in S5/S6/S8.

## Failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| `ArtifactsMissingError` from `verify-loader` | one of the 6 model files missing | check the trainer log; the model whose phase-4 fit raised will be obvious |
| MLP or sequence file is suspiciously small (<1 MB) | `state_dict()` saved before `fit()` completed | check that `MLPAlphaModel.fit` exited without an exception in the streaming run |
| OOM during phase 4 (`_refit_on_full`) | running without `STREAMING=1` on a 24 GB box | rerun with `STREAMING=1` |
| Per-fold R² for `seq` is large-negative across all folds | sequence model needs more epochs at this row count | acceptable for initial run; tune in S5/S6/S8 |
```

- [ ] **Step 3: Commit the docs**

```bash
git add CLAUDE.md docs/runbooks/s0_retrain_after_persistence.md
git commit -m "docs(s0): CLAUDE.md §13 artifact list update + s0_retrain_after_persistence runbook"
```

- [ ] **Step 4: Operator retrain gate (executes outside the agent loop)**

This step is performed by the human operator after the previous 17 tasks land on the branch.

```bash
# In a tmux or screen session, expect this to run overnight.
cd /Users/dmr/MachineLearning
make full-retrain-s1 STREAMING=1 TRAIN_CONFIG=configs/alpha_5m.yaml TRAIN_MAX_ROWS=5000000 2>&1 | tee reports/s0_retrain_$(date -u +%Y%m%dT%H%M%S).log
```

- [ ] **Step 5: Post-retrain verification + acceptance commit**

After the retrain completes:

```bash
LATEST=$(ls -t experiments/alpha_s1 | grep -v _archive_pre_s0 | head -1)
make verify-loader RUN_DIR="experiments/alpha_s1/$LATEST"
PYTHONPATH=src uv run python -c "import json; print(json.load(open('experiments/alpha_s1/$LATEST/metrics.json'))['holdout_weighted_zero_mean_r2'])"
```

Update `docs/runbooks/s0_retrain_after_persistence.md`'s "Holdout R² — honest vs zero'd" table with the new run id + holdout R². Commit:

```bash
git add docs/runbooks/s0_retrain_after_persistence.md
git commit -m "docs(s0): record first post-S0 retrain (<run_id>, R²=<value>)"
```

S0 is complete when:
- All 17 prior tasks have green commits.
- The post-retrain `make verify-loader` returns `OK`.
- `metrics.json` shows `holdout_weighted_zero_mean_r2 >= 0.012`.
- The runbook table has the new run id + R² recorded.

---

## Self-Review

**Spec coverage** — every section of `docs/superpowers/specs/2026-05-21-quantlab-alpha-s0-trainer-persistence-design.md` maps to a task:

| Spec § | Task(s) |
|---|---|
| §1 Scope | Tasks 1–18 (the whole plan) |
| §2 Technical context | n/a (context, not work) |
| §3.1 Before/after artifact layout | Task 14 (`_persist_run` writes all of it) |
| §3.2 Module layout | Tasks 2, 12–14 (training.py + exceptions.py created); Tasks 3–9 (per-class save/load); Task 11 (inference.py extension) |
| §3.3 Migration plan | Task 1 (archive), Task 17 (Makefile + delete old trainers) |
| §4 Per-class save/load contracts | Tasks 3–9 |
| §5 Schema pinning (sha256, feature_cols.json) | Task 11 (`_canonical_sha256`, `_load_and_verify_schema`, `feature_cols.json` reader); Task 14 (writer in `_persist_run`) |
| §6 Training pipeline phases 1–5 | Task 13 (phases 1–3), Task 14 (phases 4–5 + behaviour change) |
| §7 Thin CLI | Task 16 |
| §8 Testing | per-task roundtrip tests in Tasks 3–9; conftest in Task 10; persistence tests in Task 11; end-to-end test in Task 14 |
| §9.1 Code acceptance | Task 15 (whole-repo pytest+ruff+mypy); Task 17 (delete old trainers + Makefile) |
| §9.2 Retrain acceptance | Task 18 (steps 4–5: operator retrain + verification) |
| §9.3 Documentation acceptance | Task 18 (steps 1–3: CLAUDE.md + runbook) |
| §10 What S0 unblocks | n/a (downstream specs; not in scope) |
| §11 References | unchanged |

**Placeholder scan** — every step contains the actual code or command. Two intentional "implementer reads existing X" notes are placed where the existing code is the source of truth:
- Task 7 step 3: "If `MLPAlphaModel.fit` doesn't currently store `_input_dim`/`_scaler`, add them before persistence works." — this is an instruction to read, not a placeholder.
- Task 14 step 3 in `_load_and_split`: "port from `alpha_train_s1_streaming.py:_streaming_load_and_filter`" — the existing function is the spec; we are extracting it. Acceptable because the function name is exact and the source file is named precisely.

**Type consistency** — symbol names match across tasks:
- `RidgeAlphaModel`, `LightGBMAlphaModel`, `XGBoostAlphaModel`, `CatBoostAlphaModel`, `MLPAlphaModel`, `Conv1DAlphaModel`, `LinearStacker` — uppercase, single-word.
- `_BASE_MODEL_NAMES = ("ridge","lgb","xgb","cat","mlp","seq")` — used identically in Tasks 11, 13, 14.
- `_EXPECTED_BASE_MODEL_FILES` — defined in Task 11, referenced by name in Task 14.
- `_canonical_sha256` — Task 11 defines, Task 14 imports.
- `feature_cols.json` schema fields (`feature_columns`, `n_features`, `feature_cols_sha256`, `target_column`, `weight_column`, `group_column`) — identical in Task 11 (`_build_minimal_run` and `_load_and_verify_schema`) and Task 14 (`_persist_run`).
- `TrainConfig`/`RunResult`/`DataConfig`/`CVConfig`/`FeatureConfig`/`ModelsConfig` + per-model `*ModelConfig` — defined in Task 12, used identically in Tasks 13–14.
- `RunResult.base_models_persisted` is `list[str]` — Task 12 (definition) matches Task 14 (`list(_BASE_MODEL_NAMES)`).

**Behaviour change explicitly tracked** — Task 14 step 3 implements all-6-models holdout (spec §6.5). Task 18 runbook step 2 documents the R² before/after table the operator fills in.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-21-quantlab-alpha-s0-trainer-persistence-implementation.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
