"""Inference loader from a persisted run dir."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode
from quant_research_stack.alpha_eq.inference import (
    FeatureSchemaMismatchError,
    load_predictor_from_run,
)
from quant_research_stack.alpha_eq.training.persist import persist_fast_v1_run


def _toy_panel(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(40):
        d = date(2020, 1, 1) + timedelta(days=i)
        for s in range(6):
            rows.append({
                "date": d,
                "symbol": f"S{s}",
                "f1": float(rng.standard_normal()),
                "f2": float(rng.standard_normal()),
                "y_xs": float(rng.standard_normal()),
            })
    return pl.DataFrame(rows)


def test_load_predictor_predicts(tmp_path: Path) -> None:
    cfg = AlphaEqConfig(mode=TrainingMode.FAST_V1)
    persist_fast_v1_run(
        run_dir=tmp_path, config=cfg, feature_cols=["f1", "f2"],
        dev_panel=_toy_panel(), target="y_xs",
    )
    predictor = load_predictor_from_run(tmp_path)
    out = predictor.predict_batch(
        pl.DataFrame({"f1": [0.1, -0.2, 0.3], "f2": [0.4, 0.5, -0.6]})
    )
    assert out.shape == (3,)


def test_load_predictor_schema_mismatch(tmp_path: Path) -> None:
    cfg = AlphaEqConfig(mode=TrainingMode.FAST_V1)
    persist_fast_v1_run(
        run_dir=tmp_path, config=cfg, feature_cols=["f1", "f2"],
        dev_panel=_toy_panel(), target="y_xs",
    )
    predictor = load_predictor_from_run(tmp_path)
    with pytest.raises(FeatureSchemaMismatchError):
        predictor.predict_batch(pl.DataFrame({"f1": [0.1]}))
