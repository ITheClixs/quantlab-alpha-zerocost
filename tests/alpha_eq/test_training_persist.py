"""Refit-on-full + artifact persistence — required artifacts under run_dir."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode
from quant_research_stack.alpha_eq.training.persist import (
    REQUIRED_FAST_V1_ARTIFACTS,
    persist_fast_v1_run,
)


def _toy_panel() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(50):
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


def test_persist_fast_v1_run_writes_required_artifacts(tmp_path: Path) -> None:
    cfg = AlphaEqConfig(mode=TrainingMode.FAST_V1)
    persist_fast_v1_run(
        run_dir=tmp_path,
        config=cfg,
        feature_cols=["f1", "f2"],
        dev_panel=_toy_panel(),
        target="y_xs",
    )
    for art in REQUIRED_FAST_V1_ARTIFACTS:
        assert (tmp_path / art).exists(), f"missing artifact: {art}"
    sha_blob = json.loads((tmp_path / "_artifact_sha256.json").read_text())
    for art in REQUIRED_FAST_V1_ARTIFACTS:
        if art != "_artifact_sha256.json":
            assert art in sha_blob
