"""Per-fold base-learner training loop produces OOF preds."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode
from quant_research_stack.alpha_eq.training.cv import Fold
from quant_research_stack.alpha_eq.training.loop import run_fold_loop


def _toy_dataset(n_dates: int = 200, n_symbols: int = 10) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for s in range(n_symbols):
            f1 = float(rng.standard_normal())
            f2 = float(rng.standard_normal())
            y = 0.3 * f1 - 0.2 * f2 + float(rng.standard_normal()) * 0.1
            rows.append({"date": d, "symbol": f"S{s}", "f1": f1, "f2": f2, "y_xs": y})
    return pl.DataFrame(rows)


def test_run_fold_loop_returns_oof_rows() -> None:
    df = _toy_dataset()
    cfg = AlphaEqConfig(mode=TrainingMode.FAST_V1)
    unique_dates = df["date"].unique().sort().to_list()
    fold = Fold(
        fold_id=0,
        train_dates=tuple(unique_dates[:100]),
        validation_dates=tuple(unique_dates[100:150]),
    )
    feature_cols = ["f1", "f2"]
    oof = run_fold_loop(
        panel=df, feature_cols=feature_cols, target="y_xs", fold=fold, config=cfg
    )
    assert oof.height == 50 * 10
    for m in cfg.active_models():
        assert f"pred_{m}" in oof.columns
