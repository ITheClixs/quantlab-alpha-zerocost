"""Cross-sectional ranks within the date-t tradable PIT universe (spec §3.3-5).

Out-of-universe rows are excluded from the per-date rank computation so that
their values cannot influence the ranks of in-universe symbols.  The rank
column is null for out-of-universe rows.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl


def build_cross_sectional_ranks(
    panel: pl.DataFrame,
    *,
    columns: Iterable[str],
    universe_col: str,
) -> pl.DataFrame:
    out = panel
    for col in columns:
        in_uv = out.filter(pl.col(universe_col)).select(["date", "symbol", col])
        n_per_date = in_uv.group_by("date").len().rename({"len": "_n"})
        ranks = (
            in_uv.with_columns(
                pl.col(col)
                .rank(method="ordinal")
                .over("date")
                .cast(pl.Float64)
                .alias("_rank_raw")
            )
            .join(n_per_date, on="date", how="left")
            .with_columns(
                pl.when(pl.col("_n") > 1)
                .then(
                    (pl.col("_rank_raw") - 1.0) / (pl.col("_n").cast(pl.Float64) - 1.0)
                    - 0.5
                )
                .otherwise(None)
                .alias(f"rank_{col}")
            )
            .select(["date", "symbol", f"rank_{col}"])
        )
        out = out.join(ranks, on=["date", "symbol"], how="left")
    return out
