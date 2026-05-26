"""GKX-style OHLCV-characteristic subset (spec §3.3 #5, §5.6).

This is NOT a replication of Gu, Kelly, Xiu 2020. It uses ONLY OHLCV-derived
characteristics. The full GKX paper uses ~94 firm characteristics including
fundamentals; this subset is a transparent v1 approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from quant_research_stack.signal_research.papers.base import ModelFamily

GKX_FEATURE_LIST: list[str] = [
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m_skip_1m",
    "reversal_1d",
    "reversal_5d",
    "reversal_1m",
    "realized_vol_20",
    "realized_vol_60",
    "beta_to_spy_60",
    "beta_to_spy_252",
    "idiosyncratic_vol_60",
    "dollar_volume_20d",
    "amihud_illiq_20",
    "max_daily_return_20",
    "drawdown_60",
    "drawdown_252",
    "volume_shock_zscore_20",
    "close_location_20",
]


@dataclass(frozen=True)
class GKXOHLCVSubsetConfig:
    n_estimators: int = 500
    num_leaves: int = 31
    learning_rate: float = 0.05
    seed: int = 42


class GKXOHLCVSubsetModelFamily(ModelFamily):
    """GKX-style OHLCV-characteristic subset cross-sectional model."""

    def __init__(self, config: GKXOHLCVSubsetConfig) -> None:
        self.config = config
        self._booster: lgb.Booster | None = None

    def fit(self, x: pl.DataFrame, y: pl.Series) -> None:
        x_np = x.select(GKX_FEATURE_LIST).to_numpy().astype(np.float64)
        y_np = y.to_numpy().astype(np.float64)
        ds = lgb.Dataset(x_np, label=y_np)
        self._booster = lgb.train(
            params={
                "objective": "regression",
                "num_leaves": self.config.num_leaves,
                "learning_rate": self.config.learning_rate,
                "seed": self.config.seed,
                "verbose": -1,
            },
            train_set=ds,
            num_boost_round=self.config.n_estimators,
        )

    def predict(self, x: pl.DataFrame) -> pl.Series:
        if self._booster is None:
            raise RuntimeError("GKX model not fit")
        x_np = x.select(GKX_FEATURE_LIST).to_numpy().astype(np.float64)
        preds = self._booster.predict(x_np)
        return pl.Series("y_xs_pred", np.asarray(preds, dtype=np.float64))
