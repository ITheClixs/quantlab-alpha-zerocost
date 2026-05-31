"""Long-history yfinance loader (spec §2.1)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from quant_research_stack.signal_research.data.long_history import (
    LongHistoryConfig,
    fetch_one_ticker,
    save_with_manifest,
)
from quant_research_stack.signal_research.data.manifest import (
    DataQualityTier,
    load_and_verify_manifest,
    sha256_of_file,
)


def test_fetch_one_ticker_returns_required_columns() -> None:
    cfg = LongHistoryConfig(start=date(2024, 1, 1), end=date(2024, 6, 1))
    df = fetch_one_ticker("SPY", config=cfg)
    for col in ("date", "symbol", "open", "high", "low", "close", "volume"):
        assert col in df.columns
    assert df.height > 50
    assert df["symbol"].unique().to_list() == ["SPY"]


def test_save_with_manifest_writes_parquet_and_manifest(tmp_signal_research_root: Path) -> None:
    cfg = LongHistoryConfig(start=date(2024, 1, 1), end=date(2024, 6, 1))
    df = fetch_one_ticker("SPY", config=cfg)
    out_root = tmp_signal_research_root / "long_history"
    save_with_manifest(
        df,
        ticker="SPY",
        root=out_root,
        constituent_survivorship_applicable=False,
    )
    parquet = out_root / "SPY.parquet"
    manifest_json = out_root / "SPY.manifest.json"
    assert parquet.exists()
    assert manifest_json.exists()
    sha = sha256_of_file(parquet)
    m = load_and_verify_manifest(manifest_json, expected_sha256={"SPY": sha})
    assert m.data_quality_tier == DataQualityTier.PUBLIC_SNAPSHOT_NOT_PIT
    assert m.constituent_survivorship_applicable is False
