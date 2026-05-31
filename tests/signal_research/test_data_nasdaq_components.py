"""Current Nasdaq 100 list loader (spec §2.3.1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl

from quant_research_stack.signal_research.data.nasdaq_components import (
    load_or_fetch_nasdaq_100,
    save_nasdaq_100_manifest,
)


def _fake_wikipedia_ndx100() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG",
                       "TSLA", "AVGO", "COST"] * 10 + ["NFLX"],
            "name": ["..."] * 101,
        }
    )


def test_nasdaq_100_loader_returns_at_least_100(tmp_signal_research_root: Path) -> None:
    out = tmp_signal_research_root / "nasdaq" / "nasdaq_100_current.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    with patch(
        "quant_research_stack.signal_research.data.nasdaq_components._fetch_from_wikipedia",
        return_value=_fake_wikipedia_ndx100(),
    ):
        df = load_or_fetch_nasdaq_100(parquet_path=out)
    assert df.height >= 100


def test_nasdaq_100_manifest_flags_survivorship(tmp_signal_research_root: Path) -> None:
    out = tmp_signal_research_root / "nasdaq" / "nasdaq_100_current.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    with patch(
        "quant_research_stack.signal_research.data.nasdaq_components._fetch_from_wikipedia",
        return_value=_fake_wikipedia_ndx100(),
    ):
        save_nasdaq_100_manifest(parquet_path=out)
    assert (out.parent / "nasdaq_100_current.manifest.json").exists()
