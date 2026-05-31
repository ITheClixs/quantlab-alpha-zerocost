"""Paper-signal base classes (spec §3.1)."""

from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.papers.base import (
    FeatureGenerator,
    StandaloneStrategy,
    Wrapper,
)


def test_standalone_strategy_subclass_returns_dataframe() -> None:
    class Trivial(StandaloneStrategy):
        def positions(self, panel: pl.DataFrame) -> pl.DataFrame:
            return pl.DataFrame({"position": [0.0] * panel.height})

    s = Trivial()
    df = pl.DataFrame({"date": [1, 2, 3], "close": [1.0, 1.0, 1.0]})
    p = s.positions(df)
    assert p.height == 3


def test_feature_generator_subclass_returns_panel() -> None:
    class TrivialFeat(FeatureGenerator):
        def features(self, panel: pl.DataFrame) -> pl.DataFrame:
            return panel.with_columns(pl.lit(0.0).alias("zero_feature"))

    f = TrivialFeat()
    df = pl.DataFrame({"date": [1, 2]})
    out = f.features(df)
    assert "zero_feature" in out.columns


def test_wrapper_subclass_modifies_primary() -> None:
    class PassThrough(Wrapper):
        def apply(self, positions: pl.Series) -> pl.Series:
            return positions

    w = PassThrough()
    p = pl.Series("position", [0.5, -0.5, 0.0])
    out = w.apply(p)
    assert out.to_list() == [0.5, -0.5, 0.0]
