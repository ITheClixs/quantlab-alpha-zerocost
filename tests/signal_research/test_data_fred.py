"""FRED loader (spec §2.4)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl

from quant_research_stack.signal_research.data.fred import (
    FredConfig,
    fetch_fred_series,
    save_fred_panel,
)


def _fake_fred_series(series_id: str, *, api_key=None, start=None, end=None) -> pd.Series:
    idx = pd.date_range("2024-01-01", "2024-06-01", freq="D")
    s = pd.Series(range(len(idx)), index=idx, name=series_id, dtype=float)
    return s


def test_fetch_fred_series_returns_polars_df() -> None:
    with patch("quant_research_stack.signal_research.data.fred._fred_get_series", _fake_fred_series):
        df = fetch_fred_series("DGS10", config=FredConfig(start=date(2024, 1, 1), end=date(2024, 6, 1)))
    assert "date" in df.columns
    assert "DGS10" in df.columns
    assert df.height > 100


def test_save_fred_panel_emits_manifest(tmp_signal_research_root: Path) -> None:
    with patch("quant_research_stack.signal_research.data.fred._fred_get_series", _fake_fred_series):
        save_fred_panel(
            series_ids=["DGS10", "T10Y2Y"],
            config=FredConfig(start=date(2024, 1, 1), end=date(2024, 6, 1)),
            root=tmp_signal_research_root / "macro",
        )
    p = tmp_signal_research_root / "macro" / "fred_features.parquet"
    m = tmp_signal_research_root / "macro" / "fred_features.manifest.json"
    assert p.exists() and m.exists()
    df = pl.read_parquet(p)
    assert {"date", "DGS10", "T10Y2Y"}.issubset(df.columns)
