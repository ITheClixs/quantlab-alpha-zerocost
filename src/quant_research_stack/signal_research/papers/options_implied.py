"""Options-implied features (spec §3.3 #9).

For Nasdaq, prefer ^VXN; fall back to ^VIX with an imperfect-proxy label.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_research_stack.signal_research.papers.base import FeatureGenerator


@dataclass(frozen=True)
class OptionsImpliedConfig:
    nasdaq_vix_fallback_to_vix: bool = True


class OptionsImpliedFeatures(FeatureGenerator):
    def __init__(self, config: OptionsImpliedConfig) -> None:
        self.config = config

    def features(self, panel: pl.DataFrame) -> pl.DataFrame:
        df = panel
        if "vix9d" in df.columns and "vix" in df.columns:
            df = df.with_columns(
                (pl.col("vix9d") / pl.col("vix")).alias("vix_term_structure")
            )
        if "vvix" in df.columns and "vix" in df.columns:
            df = df.with_columns(
                (pl.col("vvix") / pl.col("vix")).alias("vol_of_vol_ratio")
            )
        if "skew" in df.columns:
            df = df.with_columns(pl.col("skew").alias("cboe_skew"))
        if "vxn" in df.columns:
            df = df.with_columns(pl.col("vxn").alias("nasdaq_iv"))
        elif self.config.nasdaq_vix_fallback_to_vix and "vix" in df.columns:
            df = df.with_columns(
                pl.col("vix").alias("nasdaq_iv"),
                pl.lit(True).alias("nasdaq_iv_is_vix_fallback"),
            )
        return df
