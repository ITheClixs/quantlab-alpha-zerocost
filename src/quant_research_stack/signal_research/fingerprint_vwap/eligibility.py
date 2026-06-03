"""Primary-edge stats for the VWAP entry, feeding methodology.meta_labeling
.check_eligibility (spec §6 step 4; prior #2). Net-of-cost, forward-looking returns."""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.signal_research.methodology.meta_labeling import (
    PrimarySignalStats,
)


def primary_signal_stats(
    panel: pl.DataFrame, *, horizon_days: int = 3, cost_bps_one_way: float = 1.0
) -> PrimarySignalStats:
    """Per-entry net forward return over `horizon_days` for primary_position==1 rows,
    then summarize into PrimarySignalStats. Round-trip cost = 2 * cost_bps_one_way."""
    if "primary_position" not in panel.columns:
        raise ValueError("missing 'primary_position'; call vwap_primary_position first")
    cost = 2.0 * cost_bps_one_way / 1e4
    df = panel.sort(["symbol", "date"]).with_columns(
        (pl.col("close").shift(-horizon_days).over("symbol") / pl.col("close") - 1.0).alias("fwd")
    )
    entries = df.filter((pl.col("primary_position") == 1.0) & pl.col("fwd").is_finite())
    r = entries["fwd"].to_numpy().astype(np.float64) - cost
    n = int(r.size)
    if n == 0:
        return PrimarySignalStats(0.0, 0.0, 0.0, 0, "cross_sectional", False, False)
    mean, sd = float(np.mean(r)), float(np.std(r, ddof=1)) if n > 1 else 0.0
    sharpe = 0.0 if sd == 0.0 else mean / sd * np.sqrt(252.0 / horizon_days)
    hit = float(np.mean(r > 0.0))
    expectancy = mean
    inverted_sharpe = 0.0 if sd == 0.0 else (-mean) / sd * np.sqrt(252.0 / horizon_days)
    return PrimarySignalStats(
        validation_net_sharpe=float(sharpe),
        validation_hit_rate=float(hit),
        validation_expectancy=float(expectancy),
        event_count=n,
        single_asset_or_cross_sectional="cross_sectional",
        is_inverted_superior=bool(inverted_sharpe > sharpe),
        is_near_duplicate=False,
    )
