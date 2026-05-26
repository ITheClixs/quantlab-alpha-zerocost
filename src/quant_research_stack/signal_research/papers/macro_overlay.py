"""Macro overlay features (spec §3.3 #10).

FRED series broadcast onto the panel as features/filters. Not a tuned rule
library — features only in v1.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_research_stack.signal_research.papers.base import FeatureGenerator


@dataclass(frozen=True)
class MacroOverlayConfig:
    series_to_attach: tuple[str, ...] = (
        "DGS10", "T10Y2Y", "DTWEXBGS", "DCOILWTICO", "GOLDAMGBD228NLBM",
    )


class MacroOverlayFeatures(FeatureGenerator):
    def __init__(self, config: MacroOverlayConfig) -> None:
        self.config = config

    def features(self, panel: pl.DataFrame) -> pl.DataFrame:
        present = [c for c in self.config.series_to_attach if c in panel.columns]
        if not present:
            return panel
        return panel
