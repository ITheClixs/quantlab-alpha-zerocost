"""Survivorship-safe daily return-panel construction SCAFFOLD.

Infrastructure only — NOT used for strategy validation in this scaffold. Builds a
date x instrument return panel from Sharadar SEP (+ optional TICKERS for stable
permaticker indexing, + optional ACTIONS for delisting-tail handling).

Hard invariants (tested):
- delisted names are NEVER dropped;
- the universe is NEVER filtered by future survival (no reference to the panel's
  max date / "still trading at end" when deciding inclusion).
"""

from __future__ import annotations

import polars as pl


def daily_returns(sep: pl.DataFrame) -> pl.DataFrame:
    """Per-ticker price / adjusted / dividend-inclusive daily returns (causal)."""
    if not {"ticker", "date", "close"}.issubset(sep.columns):
        raise ValueError("SEP needs at least ticker, date, close")
    df = sep.sort(["ticker", "date"])
    exprs = [(pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("ret_price")]
    if "closeadj" in df.columns:  # Sharadar closeadj is split+dividend adjusted -> total return
        exprs.append((pl.col("closeadj") / pl.col("closeadj").shift(1).over("ticker") - 1.0).alias("ret_adj"))
    if "dividends" in df.columns:  # dividend-inclusive price return when no closeadj
        exprs.append(
            ((pl.col("close") + pl.col("dividends").fill_null(0.0)) / pl.col("close").shift(1).over("ticker") - 1.0)
            .alias("ret_div_incl")
        )
    return df.with_columns(exprs)


def attach_permaticker(returns: pl.DataFrame, tickers: pl.DataFrame | None) -> pl.DataFrame:
    """Join the stable permaticker (survives ticker renames) for indexing."""
    if tickers is None or not {"ticker", "permaticker"}.issubset(tickers.columns):
        return returns.with_columns(pl.col("ticker").alias("permaticker"))
    key = tickers.select(["ticker", "permaticker"]).unique(subset=["ticker"], keep="first")
    return returns.join(key, on="ticker", how="left").with_columns(
        pl.col("permaticker").fill_null(pl.col("ticker"))
    )


def delisting_final_returns(sep: pl.DataFrame, actions: pl.DataFrame | None) -> pl.DataFrame:
    """SCAFFOLD: compute a final delisting-day return where ACTIONS supply a value.

    Returns an (empty-by-default) frame of [ticker, date, ret_delisting] to be
    unioned into the panel. Without a usable acquisition value the final return
    falls back to the last observed price path (handled upstream); this hook exists
    so the tail can be made tail-correct (bankruptcy -> ~ -1.0; acquisition -> cash
    value vs last close) once real ACTIONS fields are confirmed [VERIFY].
    """
    schema = {"ticker": pl.Utf8, "date": pl.Utf8, "ret_delisting": pl.Float64}
    if actions is None or not {"ticker", "action", "date"}.issubset(actions.columns):
        return pl.DataFrame(schema=schema)
    # Placeholder: real value/last-close join is wired after ACTIONS fields are verified.
    return pl.DataFrame(schema=schema)


def build_return_panel(sep: pl.DataFrame, *, tickers: pl.DataFrame | None = None,
                       actions: pl.DataFrame | None = None) -> pl.DataFrame:
    """Long-format date x instrument return panel. Keeps ALL tickers (incl delisted)."""
    n_in = sep["ticker"].n_unique()
    panel = attach_permaticker(daily_returns(sep), tickers)
    # invariant: do not drop any input ticker
    if panel["ticker"].n_unique() != n_in:
        raise AssertionError("return panel dropped tickers — survivorship-unsafe")
    keep = [c for c in ("permaticker", "ticker", "date", "ret_price", "ret_adj", "ret_div_incl")
            if c in panel.columns]
    return panel.select(keep).sort(["ticker", "date"])
