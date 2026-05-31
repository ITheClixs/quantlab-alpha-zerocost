"""Crypto minimal v1 — BTCUSDT + ETHUSDT daily (spec §6.6)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import polars as pl


def _fake_binance_klines(ticker: str, start: str, end: str) -> pl.DataFrame:
    dates = pl.date_range(date(2024, 1, 1), date(2024, 6, 1), interval="1d", eager=True)
    n = dates.len()
    return pl.DataFrame({
        "date": dates,
        "symbol": [ticker] * n,
        "open": [40000.0 + i for i in range(n)],
        "high": [40100.0 + i for i in range(n)],
        "low": [39900.0 + i for i in range(n)],
        "close": [40050.0 + i for i in range(n)],
        "volume": [1000.0 + i for i in range(n)],
    })


def test_crypto_minimal_loader_persists_to_disk(tmp_signal_research_root: Path) -> None:
    from quant_research_stack.signal_research.data.crypto_minimal import (
        CryptoMinimalConfig,
        save_crypto_minimal,
    )
    with patch(
        "quant_research_stack.signal_research.data.crypto_minimal._fetch_binance_klines",
        side_effect=_fake_binance_klines,
    ):
        save_crypto_minimal(
            config=CryptoMinimalConfig(
                tickers=["BTCUSDT", "ETHUSDT"],
                start=date(2024, 1, 1),
                end=date(2024, 6, 1),
            ),
            root=tmp_signal_research_root / "crypto",
        )
    for t in ("BTCUSDT", "ETHUSDT"):
        assert (tmp_signal_research_root / "crypto" / f"{t}_daily.parquet").exists()
        assert (tmp_signal_research_root / "crypto" / f"{t}_daily.manifest.json").exists()
