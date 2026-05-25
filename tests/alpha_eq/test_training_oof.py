"""OOF aggregation + stacker fit."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.training.oof import (
    collect_oof,
    fit_stacker_on_oof,
)


def test_collect_and_fit_stacker() -> None:
    fold_outputs = [
        pl.DataFrame(
            {
                "date": [date(2020, 1, k)] * 5,
                "symbol": list("ABCDE"),
                "y_xs": np.linspace(-1, 1, 5),
                "pred_ridge": np.linspace(-0.9, 0.9, 5),
                "pred_lightgbm": np.linspace(-0.8, 0.8, 5),
                "pred_xgboost": np.linspace(-0.7, 0.7, 5),
            }
        )
        for k in (2, 3, 6)
    ]
    oof = collect_oof(fold_outputs)
    assert oof.height == 15
    stacker = fit_stacker_on_oof(
        oof=oof,
        feature_order=("ridge", "lightgbm", "xgboost"),
        target="y_xs",
        alpha=1e-3,
        prefer_non_negative=True,
    )
    assert stacker.weights.shape == (3,)
