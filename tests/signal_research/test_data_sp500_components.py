"""Current SP500 list loader (spec §2.3, §6.1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl

from quant_research_stack.signal_research.data.sp500_components import (
    load_or_fetch_sp500,
    save_sp500_manifest,
)


def _fake_wikipedia_sp500() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
                       "BRK.B", "JPM", "V"] * 50 + ["XOM"],
            "name": ["..."] * 501,
            "sector": ["Tech"] * 501,
        }
    )


def test_sp500_loader_returns_at_least_500_symbols(tmp_signal_research_root: Path) -> None:
    out = tmp_signal_research_root / "sp500" / "sp500_current.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    with patch(
        "quant_research_stack.signal_research.data.sp500_components._fetch_from_wikipedia",
        return_value=_fake_wikipedia_sp500(),
    ):
        df = load_or_fetch_sp500(parquet_path=out)
    assert df.height >= 500


def test_sp500_manifest_flags_survivorship(tmp_signal_research_root: Path) -> None:
    out = tmp_signal_research_root / "sp500" / "sp500_current.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    with patch(
        "quant_research_stack.signal_research.data.sp500_components._fetch_from_wikipedia",
        return_value=_fake_wikipedia_sp500(),
    ):
        save_sp500_manifest(parquet_path=out)
    assert (out.parent / "sp500_current.manifest.json").exists()
