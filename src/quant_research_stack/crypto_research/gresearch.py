from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

FEATURE_COLUMNS: tuple[str, ...] = (
    "ret1",
    "ret5",
    "ret15",
    "ret60",
    "vwap_dev",
    "log_volume",
    "log_count",
    "vol15",
    "vol60",
)


@dataclass(frozen=True)
class ChronologicalSplit:
    development: pl.DataFrame
    validation: pl.DataFrame
    holdout: pl.DataFrame


@dataclass(frozen=True)
class PortfolioBacktestResult:
    metrics: dict[str, float | int | str | bool]
    trades: pl.DataFrame
    bars: pl.DataFrame


def build_gresearch_features(frame: pl.DataFrame, *, horizon_minutes: int = 15) -> pl.DataFrame:
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    required = {"timestamp", "Asset_ID", "Count", "Close", "Volume", "VWAP", "Target"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing G-Research columns: {sorted(missing)}")
    ordered = frame.sort(["Asset_ID", "timestamp"])
    return ordered.with_columns(
        [
            (pl.col("Close").log() - pl.col("Close").shift(1).over("Asset_ID").log()).alias("ret1"),
            (pl.col("Close").log() - pl.col("Close").shift(5).over("Asset_ID").log()).alias("ret5"),
            (pl.col("Close").log() - pl.col("Close").shift(15).over("Asset_ID").log()).alias("ret15"),
            (pl.col("Close").log() - pl.col("Close").shift(60).over("Asset_ID").log()).alias("ret60"),
            (pl.col("Close") / pl.col("VWAP") - 1.0).alias("vwap_dev"),
            pl.col("Volume").log1p().alias("log_volume"),
            pl.col("Count").log1p().alias("log_count"),
            (
                pl.col("Close").shift(-horizon_minutes).over("Asset_ID") / pl.col("Close") - 1.0
            ).alias(f"future_return_{horizon_minutes}"),
        ]
    ).with_columns(
        [
            pl.col("ret1").rolling_std(window_size=15, min_samples=15).over("Asset_ID").alias("vol15"),
            pl.col("ret1").rolling_std(window_size=60, min_samples=60).over("Asset_ID").alias("vol60"),
        ]
    )


def chronological_split(
    frame: pl.DataFrame,
    *,
    validation_fraction: float = 0.25,
    holdout_fraction: float = 0.15,
) -> ChronologicalSplit:
    if frame.height < 10:
        raise ValueError("not enough rows for chronological split")
    ordered = frame.sort("timestamp")
    timestamps = sorted(ordered.get_column("timestamp").unique().to_list())
    dev_end = int(len(timestamps) * (1.0 - validation_fraction - holdout_fraction))
    val_end = int(len(timestamps) * (1.0 - holdout_fraction))
    if dev_end <= 0 or val_end <= dev_end or val_end >= len(timestamps):
        raise ValueError("split fractions produce an empty period")
    dev_cut = timestamps[dev_end - 1]
    val_cut = timestamps[val_end - 1]
    return ChronologicalSplit(
        development=ordered.filter(pl.col("timestamp") <= dev_cut),
        validation=ordered.filter((pl.col("timestamp") > dev_cut) & (pl.col("timestamp") <= val_cut)),
        holdout=ordered.filter(pl.col("timestamp") > val_cut),
    )


def _safe_sharpe(values: np.ndarray, *, annualization: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 0.0
    std = float(np.std(finite, ddof=1))
    mean = float(np.mean(finite))
    if std <= 0.0 or not math.isfinite(std):
        return 0.0
    out = mean / std * math.sqrt(annualization)
    return out if math.isfinite(out) else 0.0


def _compound(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.prod(values + 1.0) - 1.0)


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(returns + 1.0)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    out = float(np.min(drawdowns))
    return out if math.isfinite(out) else 0.0


def _scalar_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _months_from_timestamps(timestamps: pl.Series) -> float:
    if timestamps.is_empty():
        return 0.0
    start = _scalar_float(timestamps.min())
    end = _scalar_float(timestamps.max())
    return max((end - start) / (30.0 * 24.0 * 60.0 * 60.0), 1e-9)


def portfolio_backtest(
    frame: pl.DataFrame,
    *,
    threshold: float,
    horizon_minutes: int,
    fee_bps: float,
    slippage_bps: float,
    prediction_column: str = "prediction",
    side_policy: str = "both",
    cost_multiplier: float = 1.0,
) -> PortfolioBacktestResult:
    target_column = f"future_return_{horizon_minutes}"
    required = {"timestamp", "Asset_ID", prediction_column, target_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing backtest columns: {sorted(missing)}")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    if fee_bps < 0.0 or slippage_bps < 0.0:
        raise ValueError("costs must be non-negative")
    if side_policy not in {"both", "long_only", "short_only"}:
        raise ValueError("side_policy must be one of: both, long_only, short_only")
    if cost_multiplier < 0.0:
        raise ValueError("cost_multiplier must be non-negative")
    cost = 2.0 * (fee_bps + slippage_bps) * 1e-4 * cost_multiplier
    clean = frame.drop_nulls([prediction_column, target_column])
    if clean.is_empty():
        clean = clean.with_columns(pl.lit(False).alias("__rebalance_bar"))
        return PortfolioBacktestResult(
            metrics={
                "trade_count": 0,
                "bar_count": 0,
                "gross_total_return": 0.0,
                "net_total_return": 0.0,
                "average_monthly_net_return": 0.0,
                "net_sharpe": 0.0,
                "max_drawdown": 0.0,
                "hit_rate": 0.0,
                "gross_hit_rate": 0.0,
                "avg_net_return": 0.0,
                "threshold": threshold,
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "side_policy": side_policy,
                "cost_multiplier": cost_multiplier,
            },
            trades=clean,
            bars=pl.DataFrame(),
        )
    first_timestamp = int(_scalar_float(clean.get_column("timestamp").min()))
    tradable = (
        clean
        .filter(pl.col(prediction_column).abs() > threshold)
        .with_columns(
            [
                pl.when(pl.col(prediction_column) > 0.0).then(1.0).otherwise(-1.0).alias("side"),
                (((pl.col("timestamp") - first_timestamp) % (horizon_minutes * 60)) == 0).alias(
                    "__rebalance_bar"
                ),
            ]
        )
        .filter(
            (pl.lit(side_policy) == "both")
            | ((pl.lit(side_policy) == "long_only") & (pl.col("side") > 0.0))
            | ((pl.lit(side_policy) == "short_only") & (pl.col("side") < 0.0))
        )
        .filter(pl.col("__rebalance_bar"))
        .with_columns(
            [
                (pl.col("side") * pl.col(target_column)).alias("gross_return"),
                pl.lit(cost).alias("cost_return"),
            ]
        )
        .with_columns((pl.col("gross_return") - pl.col("cost_return")).alias("net_return"))
    )
    if tradable.is_empty():
        empty_metrics: dict[str, float | int | str | bool] = {
            "trade_count": 0,
            "bar_count": 0,
            "gross_total_return": 0.0,
            "net_total_return": 0.0,
            "average_monthly_net_return": 0.0,
            "net_sharpe": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
            "gross_hit_rate": 0.0,
            "avg_net_return": 0.0,
            "threshold": threshold,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "side_policy": side_policy,
            "cost_multiplier": cost_multiplier,
        }
        return PortfolioBacktestResult(metrics=empty_metrics, trades=tradable, bars=pl.DataFrame())
    bars = (
        tradable.group_by("timestamp")
        .agg(
            [
                pl.col("gross_return").mean().alias("gross_return"),
                pl.col("net_return").mean().alias("net_return"),
                pl.len().alias("active_trade_count"),
            ]
        )
        .sort("timestamp")
    )
    net = bars.get_column("net_return").to_numpy().astype(np.float64)
    gross = bars.get_column("gross_return").to_numpy().astype(np.float64)
    net_total = _compound(net)
    months = _months_from_timestamps(bars.get_column("timestamp"))
    monthly = (1.0 + net_total) ** (1.0 / months) - 1.0 if net_total > -1.0 and months > 0.0 else -1.0
    trade_net = tradable.get_column("net_return").to_numpy().astype(np.float64)
    trade_gross = tradable.get_column("gross_return").to_numpy().astype(np.float64)
    metrics: dict[str, float | int | str | bool] = {
        "trade_count": tradable.height,
        "bar_count": bars.height,
        "gross_total_return": _compound(gross),
        "net_total_return": net_total,
        "average_monthly_net_return": float(monthly),
        "net_sharpe": _safe_sharpe(net, annualization=365.0 * 24.0 * 60.0 / horizon_minutes),
        "max_drawdown": _max_drawdown(net),
        "hit_rate": float(np.mean(trade_net > 0.0)) if trade_net.size else 0.0,
        "gross_hit_rate": float(np.mean(trade_gross > 0.0)) if trade_gross.size else 0.0,
        "avg_net_return": float(np.mean(trade_net)) if trade_net.size else 0.0,
        "threshold": threshold,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "side_policy": side_policy,
        "cost_multiplier": cost_multiplier,
    }
    return PortfolioBacktestResult(metrics=metrics, trades=tradable, bars=bars)


def rows_to_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=max(len(rows), 1)) if rows else pl.DataFrame()
