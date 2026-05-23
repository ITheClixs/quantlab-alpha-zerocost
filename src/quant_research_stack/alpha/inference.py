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


class S1Predictor(Protocol):
    def predict(self, row: pl.DataFrame) -> tuple[float, float]: ...


@dataclass
class _StackPredictor:
    base_funcs: list[Callable[[np.ndarray], float]]
    weights: NDArray[np.float64]
    feature_cols: list[str]

    def predict(self, row: pl.DataFrame) -> tuple[float, float]:
        if row.height != 1:
            raise ValueError(f"S1 predicts one row at a time; got height={row.height}")
        x = row.select(self.feature_cols).to_numpy()[0]
        base_outs = np.fromiter((f(x) for f in self.base_funcs), dtype=np.float64, count=len(self.base_funcs))
        pred = float(np.dot(self.weights, base_outs))
        # confidence: normalized agreement among base models (1.0 = unanimous sign, 0.0 = split)
        signs = np.sign(base_outs)
        if signs.size == 0 or float(np.std(base_outs)) == 0.0:
            conf = 1.0
        else:
            same_sign = float(np.mean(signs == np.sign(np.mean(signs))))
            conf = float(np.clip(same_sign, 0.0, 1.0))
        return pred, conf


def build_predictor_from_stack(
    base_funcs: list[Callable[[np.ndarray], float]],
    stacker_weights: NDArray[np.float64],
    feature_cols: list[str],
) -> S1Predictor:
    if len(base_funcs) != stacker_weights.size:
        raise ValueError("base_funcs and stacker_weights length mismatch")
    return _StackPredictor(base_funcs=base_funcs, weights=stacker_weights, feature_cols=feature_cols)


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
    base_models: dict[str, object]
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
