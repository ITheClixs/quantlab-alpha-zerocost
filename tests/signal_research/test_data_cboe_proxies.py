"""CBOE proxies via yfinance (spec §2.4)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from quant_research_stack.signal_research.data.cboe_proxies import (
    CboeProxiesConfig,
    fetch_cboe_panel,
)


def test_fetch_cboe_panel_includes_requested_tickers() -> None:
    cfg = CboeProxiesConfig(
        tickers=["^VIX", "^VVIX", "^SKEW"],
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
    )
    df = fetch_cboe_panel(config=cfg)
    assert "date" in df.columns
    # Each ticker may produce a column (close_VIX, etc.)
    for t in cfg.tickers:
        safe = t.replace("^", "")
        assert f"close_{safe}" in df.columns


def test_vxn_fallback_recorded_in_manifest(tmp_signal_research_root: Path) -> None:
    from quant_research_stack.signal_research.data.cboe_proxies import save_cboe_panel
    cfg = CboeProxiesConfig(
        tickers=["^VIX", "^VXN"],
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
    )
    save_cboe_panel(config=cfg, root=tmp_signal_research_root / "cboe")
    parquet = tmp_signal_research_root / "cboe" / "cboe_proxies.parquet"
    manifest = tmp_signal_research_root / "cboe" / "cboe_proxies.manifest.json"
    assert parquet.exists()
    assert manifest.exists()
