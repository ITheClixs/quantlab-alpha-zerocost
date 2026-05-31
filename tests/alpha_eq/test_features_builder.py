"""Feature builder composition + sha256-locked feature_cols.json."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.features.builder import (
    FeatureBuildConfig,
    build_features,
    write_feature_cols_json,
)


def _toy_panel(n: int = 80) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    dates_full = pl.date_range(
        date(2020, 1, 2), date(2020, 12, 31), interval="1d", eager=True
    )
    dates = dates_full.filter(dates_full.dt.weekday() < 6).head(n).to_list()
    rows = []
    for s in ["A", "B", "C"]:
        c = 100.0
        for d in dates:
            r = float(rng.standard_normal()) * 0.01
            c *= (1 + r)
            rows.append(
                {
                    "date": d, "symbol": s,
                    "open": c * (1 + float(rng.standard_normal()) * 0.005),
                    "high": c * (1 + abs(float(rng.standard_normal())) * 0.01),
                    "low": c * (1 - abs(float(rng.standard_normal())) * 0.01),
                    "close": c,
                    "volume": int(1_000_000 + abs(float(rng.standard_normal())) * 100_000),
                    "in_universe": True,
                }
            )
    return pl.DataFrame(rows)


def test_build_features_returns_expected_columns_and_no_meta() -> None:
    df = build_features(panel=_toy_panel(), config=FeatureBuildConfig())
    must_have = {
        "feature_as_of_date", "execution_date",
        "log_return_1", "realized_vol_20", "amihud_illiq_20",
        "dollar_volume", "rank_log_return_1", "vix_close", "vix_is_proxy",
        "gaussian_noise_seed42",
    }
    assert must_have.issubset(set(df.columns))


def test_write_feature_cols_json_sha256(tmp_path: Path) -> None:
    cols = ["log_return_1", "realized_vol_20", "gaussian_noise_seed42"]
    out = tmp_path / "feature_cols.json"
    write_feature_cols_json(out, cols)
    blob = json.loads(out.read_text())
    assert blob["feature_columns"] == cols
    assert len(blob["feature_cols_sha256"]) == 64
