"""OOF aggregation + stacker fit."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.stacking import LinearStackerEq


def collect_oof(fold_outputs: Sequence[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(list(fold_outputs))


def fit_stacker_on_oof(
    *,
    oof: pl.DataFrame,
    feature_order: Sequence[str],
    target: str,
    alpha: float,
    prefer_non_negative: bool,
) -> LinearStackerEq:
    cols = [f"pred_{n}" for n in feature_order]
    keep = oof.drop_nulls(subset=[*cols, target])
    x = keep.select(cols).to_numpy().astype(np.float64)
    y = keep[target].to_numpy().astype(np.float64)
    s = LinearStackerEq(
        alpha=alpha, prefer_non_negative=prefer_non_negative, feature_order=feature_order
    )
    s.fit(oof_predictions=x, y=y)
    return s
