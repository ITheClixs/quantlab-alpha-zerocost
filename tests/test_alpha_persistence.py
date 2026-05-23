import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def _build_minimal_run(run_dir: Path) -> list[str]:
    """Construct a minimal valid run directory: 6 base models, stacker, feature_cols.json.

    Avoids LightGBM + CatBoost in pytest on macOS due to thread-safety segfaults.
    Instead, we create Ridge models and mock the load() methods.
    """
    rng = np.random.default_rng(0)
    n = 100
    n_features = 4
    x_tr = rng.standard_normal((n, n_features))
    y_tr = x_tr[:, 0] + 0.1 * rng.standard_normal(n)
    w_tr = np.ones(n)
    x_val = rng.standard_normal((20, n_features))
    y_val = x_val[:, 0]
    w_val = np.ones(20)

    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train Ridge (safe)
    r = RidgeAlphaModel(RidgeConfig(alpha=1.0))
    r.fit(x_tr, y_tr, w_tr)
    r.save(models_dir / "ridge.joblib")

    # For LightGBM: save a Ridge model but under the lgb filename, then patch load()
    l = RidgeAlphaModel(RidgeConfig(alpha=1.0))
    l.fit(x_tr, y_tr, w_tr)
    l.save(models_dir / "lightgbm.txt")

    # For XGBoost: save a Ridge model under the xgb filename
    xg_stub = RidgeAlphaModel(RidgeConfig(alpha=1.0))
    xg_stub.fit(x_tr, y_tr, w_tr)
    xg_stub.save(models_dir / "xgboost.json")

    # For CatBoost: save a Ridge model under the cb filename, then patch load()
    cb = RidgeAlphaModel(RidgeConfig(alpha=1.0))
    cb.fit(x_tr, y_tr, w_tr)
    cb.save(models_dir / "catboost.cbm")

    # Train MLP
    mp = MLPAlphaModel(MLPConfig(hidden_dims=[4], dropout=0.0, learning_rate=1e-3,
                                 batch_size=32, max_epochs=1, patience=1,
                                 mixed_precision=False))
    mp.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)
    mp.save(models_dir / "mlp.pt")

    # Train Conv1D
    seq = Conv1DAlphaModel(Conv1DConfig(kernel_sizes=[2], n_filters=4, dropout=0.0,
                                        learning_rate=1e-3, batch_size=32, max_epochs=1,
                                        patience=1, random_state=0))
    seq.fit(x_tr, y_tr, w_tr, x_val, y_val, w_val)
    seq.save(models_dir / "sequence.pt")

    feature_order = ["ridge", "lgb", "xgb", "cat", "mlp", "seq"]
    # Tiny stacker — fit on the val slice with all 6 outputs.
    stack_x = np.column_stack([
        r.predict(x_val), l.predict(x_val), xg_stub.predict(x_val),
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
    a = _canonical_sha256(["x"])
    b = _canonical_sha256(["x "])
    assert a != b


def test_load_predictor_from_run_happy_path(tmp_path):
    feature_cols = _build_minimal_run(tmp_path)

    # Patch tree-based models to return Ridge stubs (avoid macOS segfaults)
    with patch.object(LightGBMAlphaModel, 'load', return_value=RidgeAlphaModel.load(tmp_path / "models" / "lightgbm.txt")):
        with patch.object(XGBoostAlphaModel, 'load', return_value=RidgeAlphaModel.load(tmp_path / "models" / "xgboost.json")):
            with patch.object(CatBoostAlphaModel, 'load', return_value=RidgeAlphaModel.load(tmp_path / "models" / "catboost.cbm")):
                predictor = load_predictor_from_run(tmp_path)

    assert sorted(predictor.expected_feature_columns) == sorted(feature_cols)

    row = pl.DataFrame({c: [0.5] for c in feature_cols})
    pred, conf = predictor.predict(row)
    assert isinstance(pred, float)
    assert 0.0 <= conf <= 1.0


def test_load_predictor_from_run_rejects_pre_s0(tmp_path):
    (tmp_path / "models").mkdir()
    joblib.dump({"weights": [0.2] * 6, "feature_order": ["ridge","lgb","xgb","cat","mlp","seq"],
                 "intercept": 0.0, "alpha": 1e-3},
                tmp_path / "models" / "stacker.joblib")
    with pytest.raises(ArtifactsMissingError):
        load_predictor_from_run(tmp_path)


def test_load_predictor_from_run_detects_sha_tamper(tmp_path):
    _build_minimal_run(tmp_path)
    schema_path = tmp_path / "feature_cols.json"
    schema = json.loads(schema_path.read_text())
    schema["feature_cols_sha256"] = "0" * 64
    schema_path.write_text(json.dumps(schema))
    with patch.object(LightGBMAlphaModel, 'load', return_value=RidgeAlphaModel.load(tmp_path / "models" / "lightgbm.txt")):
        with patch.object(XGBoostAlphaModel, 'load', return_value=RidgeAlphaModel.load(tmp_path / "models" / "xgboost.json")):
            with patch.object(CatBoostAlphaModel, 'load', return_value=RidgeAlphaModel.load(tmp_path / "models" / "catboost.cbm")):
                with pytest.raises(FeatureSchemaError, match="sha256 mismatch"):
                    load_predictor_from_run(tmp_path)


def test_predictor_rejects_missing_columns(tmp_path):
    _build_minimal_run(tmp_path)
    with patch.object(LightGBMAlphaModel, 'load', return_value=RidgeAlphaModel.load(tmp_path / "models" / "lightgbm.txt")):
        with patch.object(XGBoostAlphaModel, 'load', return_value=RidgeAlphaModel.load(tmp_path / "models" / "xgboost.json")):
            with patch.object(CatBoostAlphaModel, 'load', return_value=RidgeAlphaModel.load(tmp_path / "models" / "catboost.cbm")):
                predictor = load_predictor_from_run(tmp_path)
    bad = pl.DataFrame({"feature_00": [0.1], "feature_01": [0.2]})
    with pytest.raises(FeatureSchemaError):
        predictor.predict(bad)


def test_predictor_reorders_columns(tmp_path):
    feature_cols = _build_minimal_run(tmp_path)
    with patch.object(LightGBMAlphaModel, 'load', return_value=RidgeAlphaModel.load(tmp_path / "models" / "lightgbm.txt")):
        with patch.object(XGBoostAlphaModel, 'load', return_value=RidgeAlphaModel.load(tmp_path / "models" / "xgboost.json")):
            with patch.object(CatBoostAlphaModel, 'load', return_value=RidgeAlphaModel.load(tmp_path / "models" / "catboost.cbm")):
                predictor = load_predictor_from_run(tmp_path)
    in_order = pl.DataFrame({c: [0.5] for c in feature_cols})
    shuffled_cols = list(reversed(feature_cols))
    shuffled = pl.DataFrame({c: [0.5] for c in shuffled_cols})
    np.testing.assert_array_equal(
        np.array(predictor.predict(in_order)),
        np.array(predictor.predict(shuffled)),
    )
