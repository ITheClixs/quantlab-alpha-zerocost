"""HMM regime feature (spec §3.3 #7).

Wraps `methodology.regime_conditional.fit_hmm_regimes` as a FeatureGenerator.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_research_stack.signal_research.methodology.regime_conditional import (
    fit_hmm_regimes,
)
from quant_research_stack.signal_research.papers.base import FeatureGenerator


@dataclass(frozen=True)
class HMMRegimeConfig:
    n_states: int = 2
    seed: int = 42


class HMMRegimeFeature(FeatureGenerator):
    def __init__(self, config: HMMRegimeConfig) -> None:
        self.config = config

    def features(self, panel: pl.DataFrame) -> pl.DataFrame:
        df = panel.sort("date").with_columns(
            (
                pl.col("market_close").log() - pl.col("market_close").shift(1).log()
            ).alias("_r")
        )
        rets = df["_r"].fill_null(0.0).to_numpy().astype(float)
        states = fit_hmm_regimes(
            rets, n_states=self.config.n_states, seed=self.config.seed
        )
        return df.with_columns(pl.Series("regime_id", states.tolist())).drop("_r")
