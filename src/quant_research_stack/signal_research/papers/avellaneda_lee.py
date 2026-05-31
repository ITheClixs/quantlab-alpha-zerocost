"""Avellaneda-Lee (2010) cross-sectional residual MR with rolling PCA.

Spec §3.3 #4, §5.5. Invariants:
- PCA fit on PAST data only (rolling window).
- Residuals standardised cross-sectionally per date.
- Z-score entry threshold predeclared.
- Exit / rebalance cadence predeclared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from quant_research_stack.signal_research.papers.base import StandaloneStrategy


@dataclass(frozen=True)
class AvellanedaLeeConfig:
    pca_window: int = 252
    n_components: int = 5
    z_entry: float = 1.5
    z_exit: float = 0.5
    rebalance_cadence: str = "daily"


class AvellanedaLeeStrategy(StandaloneStrategy):
    def __init__(self, config: AvellanedaLeeConfig) -> None:
        self.config = config

    def positions(self, panel: pl.DataFrame) -> pl.DataFrame:
        df = panel.sort(["symbol", "date"])
        df = df.with_columns(
            (
                pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()
            ).alias("ret")
        )
        wide = (
            df.pivot(values="ret", index="date", on="symbol").sort("date").fill_null(0.0)
        )
        symbols = [c for c in wide.columns if c != "date"]
        R = wide.select(symbols).to_numpy().astype(np.float64)
        T = R.shape[0]
        preds = np.full(R.shape, np.nan, dtype=np.float64)
        for t in range(self.config.pca_window, T):
            window = R[t - self.config.pca_window : t]
            mean = window.mean(axis=0, keepdims=True)
            centred = window - mean
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
            comps = Vt[: self.config.n_components]
            today = R[t] - mean.flatten()
            factor_loadings = today @ comps.T
            reconstruction = factor_loadings @ comps
            residual = today - reconstruction
            window_residuals = centred - centred @ comps.T @ comps
            std = float(np.std(window_residuals, ddof=1)) or 1.0
            z = residual / std
            preds[t] = -np.clip(z / self.config.z_entry, -1.0, 1.0)

        date_list = wide["date"].to_list()
        date_col: list[object] = []
        sym_col: list[str] = []
        pred_col: list[float | None] = []
        for ti, d in enumerate(date_list):
            for si, sym in enumerate(symbols):
                v = preds[ti, si]
                date_col.append(d)
                sym_col.append(sym)
                pred_col.append(float(v) if not np.isnan(v) else None)
        return pl.DataFrame(
            {"date": date_col, "symbol": sym_col, "y_xs_pred": pred_col},
            schema={"date": wide.schema["date"], "symbol": pl.Utf8, "y_xs_pred": pl.Float64},
        )
