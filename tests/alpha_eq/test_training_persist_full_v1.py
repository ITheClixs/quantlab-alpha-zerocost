"""Full_v1 persistence — adds CatBoost + MLP (Conv1D optional)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode
from quant_research_stack.alpha_eq.training.persist import (
    REQUIRED_FULL_V1_ARTIFACTS,
    persist_full_v1_run,
)


def _toy_panel() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        d = date(2020, 1, 1) + timedelta(days=i)
        for s in range(8):
            rows.append({
                "date": d,
                "symbol": f"S{s}",
                "f1": float(rng.standard_normal()),
                "f2": float(rng.standard_normal()),
                "f3": float(rng.standard_normal()),
                "y_xs": float(rng.standard_normal()),
            })
    return pl.DataFrame(rows)


def test_persist_full_v1_writes_extra_artifacts(tmp_path: Path) -> None:
    cfg = AlphaEqConfig(mode=TrainingMode.FULL_V1)
    persist_full_v1_run(
        run_dir=tmp_path,
        config=cfg,
        feature_cols=["f1", "f2", "f3"],
        dev_panel=_toy_panel(),
        target="y_xs",
        enable_sequence=False,
    )
    for art in REQUIRED_FULL_V1_ARTIFACTS:
        assert (tmp_path / art).exists(), f"missing artifact: {art}"
