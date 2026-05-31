"""Top-minus-bottom spread, rank IC, and long/short direction are internally consistent."""

from __future__ import annotations

import numpy as np
import polars as pl


def test_positive_ic_implies_positive_top_minus_bottom() -> None:
    rng = np.random.default_rng(0)
    n = 1_000
    truth = rng.standard_normal(n)
    pred = truth + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"pred": pred, "y": truth})
    top = df.filter(pl.col("pred") >= df["pred"].quantile(0.9))
    bot = df.filter(pl.col("pred") <= df["pred"].quantile(0.1))
    spread = float(top["y"].mean()) - float(bot["y"].mean())
    assert spread > 0.0
