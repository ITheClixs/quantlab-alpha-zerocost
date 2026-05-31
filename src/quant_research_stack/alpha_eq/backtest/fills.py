"""Fill-price selection (spec §5.3).  HLC3 is ALWAYS labelled vwap_proxy_hlc3
in any artifact column; never called real VWAP."""

from __future__ import annotations

import enum

import polars as pl


class FillModel(enum.StrEnum):
    OPEN = "open"
    HLC3_PROXY = "vwap_proxy_hlc3"
    CLOSE = "close"


def pick_fill_prices(bars: pl.DataFrame, *, model: FillModel) -> pl.DataFrame:
    if model is FillModel.OPEN:
        out = bars.with_columns(pl.col("open").alias("fill_price"))
    elif model is FillModel.HLC3_PROXY:
        out = bars.with_columns(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("fill_price")
        )
    elif model is FillModel.CLOSE:
        out = bars.with_columns(pl.col("close").alias("fill_price"))
    else:  # pragma: no cover
        raise ValueError(f"unknown fill model: {model}")
    return out.with_columns(pl.lit(model.value).alias("fill_model"))
