"""M1 integration: long-history loader emits manifest end-to-end."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from quant_research_stack.signal_research.data.long_history import (
    LongHistoryConfig,
    fetch_one_ticker,
    save_with_manifest,
)


def test_long_history_save_to_signal_research_root(tmp_signal_research_root: Path) -> None:
    df = fetch_one_ticker(
        "SPY", config=LongHistoryConfig(start=date(2024, 1, 1), end=date(2024, 3, 1))
    )
    save_with_manifest(
        df,
        ticker="SPY",
        root=tmp_signal_research_root / "long_history",
        constituent_survivorship_applicable=False,
    )
    assert (tmp_signal_research_root / "long_history" / "SPY.parquet").exists()
    assert (tmp_signal_research_root / "long_history" / "SPY.manifest.json").exists()
