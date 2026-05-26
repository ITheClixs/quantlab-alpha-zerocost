"""Vol-Risk-Premium (Bondarenko 2014).

Spec §3.3 #6: distinguishes implied-vol FEATURE from tradable strategy.
v1 ships the feature variant; tradable variant requires a real instrument.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_research_stack.signal_research.papers.base import (
    FeatureGenerator,
    StandaloneStrategy,
)


@dataclass(frozen=True)
class VRPFeatureConfig:
    realized_vol_window: int = 20


class VRPFeature(FeatureGenerator):
    """Implied-vol feature: ^VIX/100 - realised_vol_20 (annualised)."""

    def __init__(self, config: VRPFeatureConfig) -> None:
        self.config = config

    def features(self, panel: pl.DataFrame) -> pl.DataFrame:
        return panel.with_columns(
            (
                (pl.col("vix") / 100.0)
                - (pl.col("close").log() - pl.col("close").shift(1).log())
                .rolling_std(
                    window_size=self.config.realized_vol_window,
                    min_samples=self.config.realized_vol_window,
                )
                * (252 ** 0.5)
            ).alias("vrp")
        )


class VRPTradableNotConfiguredError(RuntimeError):
    pass


class VRPTradableStrategy(StandaloneStrategy):
    """Tradable VRP: short-vol via a real instrument. Refuses if no instrument
    is provided (the VIX index itself is NOT tradable; per spec §3.3 #6,
    pure VIX-index strategies are diagnostic-only)."""

    def __init__(self, tradable_instrument: str | None) -> None:
        if tradable_instrument is None:
            raise VRPTradableNotConfiguredError(
                "VRP tradable strategy requires a real instrument "
                "(SVXY, VIXM, VIX futures, etc.). The VIX index itself is NOT tradable."
            )
        self.instrument = tradable_instrument

    def positions(self, panel: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame({"position": [0.0] * panel.height})
