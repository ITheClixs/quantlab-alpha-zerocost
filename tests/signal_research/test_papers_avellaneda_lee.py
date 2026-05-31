"""Avellaneda-Lee (2010) — rolling-PCA residual MR (spec §5.5)."""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.signal_research.papers.avellaneda_lee import (
    AvellanedaLeeConfig,
    AvellanedaLeeStrategy,
)


def _toy_cs_panel(n_dates: int = 300, n_symbols: int = 30, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n_dates) * 0.01
    rows = []
    for s in range(n_symbols):
        beta = 0.5 + 0.5 * rng.standard_normal()
        idiosyncratic = rng.standard_normal(n_dates) * 0.005
        returns = beta * factor + idiosyncratic
        price = 100.0 * np.cumprod(1.0 + returns)
        for t in range(n_dates):
            rows.append({"date": t, "symbol": f"S{s}", "close": float(price[t])})
    return pl.DataFrame(rows)


def test_avellaneda_lee_produces_predictions_per_date_symbol() -> None:
    panel = _toy_cs_panel()
    cfg = AvellanedaLeeConfig(pca_window=120, n_components=3, z_entry=1.5)
    strat = AvellanedaLeeStrategy(cfg)
    preds = strat.positions(panel)
    assert "y_xs_pred" in preds.columns


def test_avellaneda_lee_uses_only_past_data_for_pca() -> None:
    panel = _toy_cs_panel()
    cfg = AvellanedaLeeConfig(pca_window=120, n_components=3, z_entry=1.5)
    strat = AvellanedaLeeStrategy(cfg)
    preds = strat.positions(panel)
    early = (
        preds.filter(pl.col("date") < cfg.pca_window)
             .filter(pl.col("y_xs_pred").is_not_null())
    )
    assert early.height == 0
