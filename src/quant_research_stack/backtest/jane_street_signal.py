from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.alpha.metrics import weighted_zero_mean_r2


@dataclass(frozen=True)
class JaneStreetSignalBacktestResult:
    daily_curve: pl.DataFrame
    metrics: dict[str, float | int]


def _clean_frame(
    frame: pl.DataFrame,
    prediction_column: str,
    *,
    target_column: str,
    weight_column: str,
) -> pl.DataFrame:
    required = {prediction_column, target_column, weight_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing Jane Street signal columns: {sorted(missing)}")
    return (
        frame.drop_nulls([prediction_column, target_column, weight_column])
        .with_columns(
            [
                pl.col(prediction_column).cast(pl.Float64, strict=False).alias(prediction_column),
                pl.col(target_column).cast(pl.Float64, strict=False).alias(target_column),
                pl.col(weight_column).cast(pl.Float64, strict=False).alias(weight_column),
            ]
        )
        .filter(
            pl.col(prediction_column).is_finite()
            & pl.col(target_column).is_finite()
            & pl.col(weight_column).is_finite()
            & (pl.col(weight_column) >= 0.0)
        )
    )


def _weighted_mean(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(values * weights) / denom)


def _weighted_corr(a: NDArray[np.float64], b: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    denom = float(np.sum(weights))
    if a.size < 2 or denom <= 0.0:
        return 0.0
    mean_a = _weighted_mean(a, weights)
    mean_b = _weighted_mean(b, weights)
    da = a - mean_a
    db = b - mean_b
    var_a = float(np.sum(weights * da * da) / denom)
    var_b = float(np.sum(weights * db * db) / denom)
    if var_a <= 0.0 or var_b <= 0.0:
        return 0.0
    return float(np.sum(weights * da * db) / denom / math.sqrt(var_a * var_b))


def _max_drawdown_from_cumulative(values: list[float]) -> float:
    peak = 0.0
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return worst


def evaluate_prediction_column(
    frame: pl.DataFrame,
    prediction_column: str,
    *,
    target_column: str = "target_actual",
    weight_column: str = "weight",
) -> dict[str, float | int]:
    clean = _clean_frame(
        frame,
        prediction_column,
        target_column=target_column,
        weight_column=weight_column,
    )
    if clean.is_empty():
        return {
            "rows": 0,
            "weighted_zero_mean_r2": 0.0,
            "weighted_directional_accuracy": 0.0,
            "positive_precision": 0.0,
            "negative_precision": 0.0,
            "weighted_sign_capture": 0.0,
            "weighted_information_coefficient": 0.0,
            "positive_signal_share": 0.0,
        }

    y = clean[target_column].to_numpy().astype(np.float64)
    pred = clean[prediction_column].to_numpy().astype(np.float64)
    weight = clean[weight_column].to_numpy().astype(np.float64)
    total_weight = float(np.sum(weight))
    same_sign = (pred > 0.0) == (y > 0.0)
    positive_mask = pred > 0.0
    negative_mask = pred < 0.0

    def weighted_share(mask: NDArray[np.bool_]) -> float:
        if total_weight <= 0.0:
            return 0.0
        return float(np.sum(weight[mask]) / total_weight)

    return {
        "rows": clean.height,
        "weighted_zero_mean_r2": float(weighted_zero_mean_r2(y, pred, weight)),
        "weighted_directional_accuracy": weighted_share(same_sign),
        "positive_precision": (
            _weighted_mean((y[positive_mask] > 0.0).astype(np.float64), weight[positive_mask])
            if np.any(positive_mask)
            else 0.0
        ),
        "negative_precision": (
            _weighted_mean((y[negative_mask] < 0.0).astype(np.float64), weight[negative_mask])
            if np.any(negative_mask)
            else 0.0
        ),
        "weighted_sign_capture": _weighted_mean(np.sign(pred) * y, weight),
        "weighted_information_coefficient": _weighted_corr(pred, y, weight),
        "positive_signal_share": weighted_share(positive_mask),
    }


def run_grouped_long_short_backtest(
    frame: pl.DataFrame,
    prediction_column: str,
    *,
    target_column: str = "target_actual",
    weight_column: str = "weight",
    group_column: str = "date_id",
    selection_fraction: float = 0.10,
) -> JaneStreetSignalBacktestResult:
    if not 0.0 < selection_fraction <= 0.5:
        raise ValueError("selection_fraction must be in (0, 0.5]")
    clean = _clean_frame(
        frame,
        prediction_column,
        target_column=target_column,
        weight_column=weight_column,
    )
    if group_column not in clean.columns:
        raise ValueError(f"missing group column: {group_column}")

    rows: list[dict[str, float | int]] = []
    cumulative = 0.0
    for group in clean.sort(group_column).partition_by(group_column, maintain_order=True):
        if group.height < 2:
            continue
        n_select = max(1, int(math.floor(group.height * selection_fraction)))
        n_select = min(n_select, max(1, group.height // 2))
        ranked = group.sort(prediction_column, descending=True)
        longs = ranked.head(n_select)
        shorts = ranked.tail(n_select)
        long_y = longs[target_column].to_numpy().astype(np.float64)
        long_w = longs[weight_column].to_numpy().astype(np.float64)
        short_y = shorts[target_column].to_numpy().astype(np.float64)
        short_w = shorts[weight_column].to_numpy().astype(np.float64)
        long_capture = _weighted_mean(long_y, long_w)
        short_capture = _weighted_mean(short_y, short_w)
        spread = long_capture - short_capture
        daily_pnl = 0.5 * spread
        cumulative += daily_pnl
        rows.append(
            {
                group_column: int(cast(int, group[group_column][0])),
                "long_count": int(longs.height),
                "short_count": int(shorts.height),
                "long_weighted_responder": long_capture,
                "short_weighted_responder": short_capture,
                "long_short_spread": spread,
                "pnl_units": daily_pnl,
                "cumulative_pnl_units": cumulative,
            }
        )

    if not rows:
        empty = pl.DataFrame(
            {
                group_column: [],
                "long_count": [],
                "short_count": [],
                "long_weighted_responder": [],
                "short_weighted_responder": [],
                "long_short_spread": [],
                "pnl_units": [],
                "cumulative_pnl_units": [],
            }
        )
        return JaneStreetSignalBacktestResult(
            daily_curve=empty,
            metrics={
                "n_groups": 0,
                "total_pnl_units": 0.0,
                "mean_pnl_units": 0.0,
                "mean_long_short_spread": 0.0,
                "sharpe_like": 0.0,
                "hit_rate": 0.0,
                "max_drawdown_units": 0.0,
            },
        )

    curve = pl.DataFrame(rows)
    pnl = curve["pnl_units"].to_numpy().astype(np.float64)
    sigma = float(np.std(pnl, ddof=1)) if pnl.size > 1 else 0.0
    mean_pnl = float(np.mean(pnl))
    sharpe_like = 0.0 if sigma == 0.0 else float(mean_pnl / sigma * math.sqrt(252.0))
    metrics: dict[str, float | int] = {
        "n_groups": int(curve.height),
        "total_pnl_units": float(cast(float, curve["pnl_units"].sum())),
        "mean_pnl_units": mean_pnl,
        "mean_long_short_spread": float(cast(float, curve["long_short_spread"].mean())),
        "sharpe_like": sharpe_like,
        "hit_rate": float(np.mean(pnl > 0.0)),
        "max_drawdown_units": _max_drawdown_from_cumulative(curve["cumulative_pnl_units"].to_list()),
    }
    return JaneStreetSignalBacktestResult(daily_curve=curve, metrics=metrics)
