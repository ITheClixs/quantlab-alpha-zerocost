from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass(frozen=True)
class SignalModelArtifact:
    model: Any
    feature_columns: list[str]
    target_column: str
    artifact_path: Path


@dataclass(frozen=True)
class EquitySignalBacktestResult:
    daily_curve: pl.DataFrame
    metrics: dict[str, float | int]


def _date_expression(column: str, dtype: pl.DataType) -> pl.Expr:
    if dtype == pl.Date:
        return pl.col(column).alias("date")
    if isinstance(dtype, pl.Datetime):
        return pl.col(column).dt.date().alias("date")
    return pl.col(column).cast(pl.Utf8).str.slice(0, 10).str.to_date(strict=False).alias("date")


def normalize_equity_ohlcv(
    frame: pl.DataFrame,
    *,
    dataset_id: str,
    date_column: str,
    symbol_column: str,
    open_column: str = "open",
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "close",
    volume_column: str = "volume",
    volatility_windows: tuple[int, ...] = (5, 20, 60),
    forward_horizons: tuple[int, ...] = (1,),
) -> pl.DataFrame:
    """Normalize daily equity OHLCV rows and add strictly past-looking features.

    The only forward-looking output is `future_return_1`, the next-row close-to-close
    return per symbol. All predictor features use current or historical rows only.
    """
    required = {
        date_column,
        symbol_column,
        open_column,
        high_column,
        low_column,
        close_column,
        volume_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")

    date_dtype = frame.schema[date_column]
    out = (
        frame.select(
            [
                pl.lit(dataset_id).alias("dataset_id"),
                _date_expression(date_column, date_dtype),
                pl.col(symbol_column).cast(pl.Utf8, strict=False).alias("symbol"),
                pl.col(open_column).cast(pl.Float64, strict=False).alias("open"),
                pl.col(high_column).cast(pl.Float64, strict=False).alias("high"),
                pl.col(low_column).cast(pl.Float64, strict=False).alias("low"),
                pl.col(close_column).cast(pl.Float64, strict=False).alias("close"),
                pl.col(volume_column).cast(pl.Float64, strict=False).alias("volume"),
            ]
        )
        .drop_nulls(["date", "symbol", "open", "high", "low", "close", "volume"])
        .filter((pl.col("close") > 0.0) & (pl.col("open") > 0.0))
        .sort(["symbol", "date"])
        .with_columns(
            [
                pl.col("date").cast(pl.Datetime("us")).alias("timestamp_utc"),
                (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias("return_1"),
                (pl.col("open") / pl.col("close").shift(1).over("symbol") - 1.0).alias("gap_return_1"),
                (pl.col("close") / pl.col("open") - 1.0).alias("open_close_return"),
                pl.col("close").log().diff().over("symbol").alias("log_return_1"),
                ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("high_low_range"),
                (pl.col("volume") * pl.col("close")).alias("dollar_volume"),
                (pl.col("volume") / pl.col("volume").shift(1).over("symbol") - 1.0).alias("volume_change_1"),
            ]
        )
    )
    for window in volatility_windows:
        out = out.with_columns(
            [
                pl.col("log_return_1")
                .rolling_std(window_size=window, min_samples=max(2, min(window, 5)))
                .over("symbol")
                .alias(f"realized_vol_{window}"),
                pl.col("return_1")
                .rolling_mean(window_size=window, min_samples=max(2, min(window, 5)))
                .over("symbol")
                .alias(f"return_{window}_mean"),
            ]
        )
    forward_exprs = []
    direction_exprs = []
    for horizon in sorted(set(forward_horizons)):
        if horizon <= 0:
            raise ValueError(f"forward horizons must be positive; got {horizon}")
        target_name = f"future_return_{horizon}"
        forward_exprs.append((pl.col("close").shift(-horizon).over("symbol") / pl.col("close") - 1.0).alias(target_name))
        direction_exprs.append((pl.col(target_name) > 0.0).cast(pl.Int8).alias(f"direction_up_{horizon}"))
    out = out.with_columns(forward_exprs)
    return out.with_columns(direction_exprs)


def load_signal_model(path: str | Path) -> SignalModelArtifact:
    artifact_path = Path(path)
    payload = joblib.load(artifact_path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"unsupported signal model artifact: {artifact_path}")
    feature_columns = payload.get("features") or payload.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"artifact lacks a non-empty feature list: {artifact_path}")
    target_column = str(payload.get("target") or payload.get("target_column") or "future_return_1")
    return SignalModelArtifact(
        model=payload["model"],
        feature_columns=[str(col) for col in feature_columns],
        target_column=target_column,
        artifact_path=artifact_path,
    )


def _feature_matrix(frame: pl.DataFrame, feature_columns: list[str]) -> NDArray[np.float32]:
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing model features: {sorted(missing)}")
    selected = frame.select([pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in feature_columns])
    selected = selected.fill_null(0.0).fill_nan(0.0)
    x = selected.to_numpy().astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def predict_signal_frame(
    frame: pl.DataFrame,
    artifact: SignalModelArtifact,
    *,
    prediction_column: str = "prediction",
) -> pl.DataFrame:
    x = _feature_matrix(frame, artifact.feature_columns)
    pred = np.asarray(artifact.model.predict(x), dtype=np.float64).reshape(-1)
    return frame.with_columns(
        [
            pl.Series(prediction_column, pred),
            pl.Series("prediction_abs", np.abs(pred)),
            pl.Series("signal_side", np.sign(pred).astype(np.int8)),
        ]
    )


def _clean_signal_frame(
    frame: pl.DataFrame,
    *,
    prediction_column: str,
    target_column: str,
) -> pl.DataFrame:
    required = {prediction_column, target_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing signal columns: {sorted(missing)}")
    return (
        frame.drop_nulls([prediction_column, target_column])
        .with_columns(
            [
                pl.col(prediction_column).cast(pl.Float64, strict=False).alias(prediction_column),
                pl.col(target_column).cast(pl.Float64, strict=False).alias(target_column),
            ]
        )
        .filter(pl.col(prediction_column).is_finite() & pl.col(target_column).is_finite())
    )


def _zero_mean_r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    denom = float(np.sum(np.square(y_true)))
    if denom <= 0.0:
        return 0.0
    return float(1.0 - np.sum(np.square(y_true - y_pred)) / denom)


def _safe_corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    if a.size < 2 or b.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if math.isnan(value) else value


def _ordinal_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _rank_ic_by_date(
    frame: pl.DataFrame,
    *,
    prediction_column: str,
    target_column: str,
    date_column: str,
) -> list[float]:
    if date_column not in frame.columns:
        return []
    values: list[float] = []
    for group in frame.partition_by(date_column, maintain_order=True):
        if group.height < 3:
            continue
        pred = group[prediction_column].to_numpy().astype(np.float64)
        target = group[target_column].to_numpy().astype(np.float64)
        values.append(_safe_corr(_ordinal_ranks(pred), _ordinal_ranks(target)))
    return values


def _top_bottom_returns(
    frame: pl.DataFrame,
    *,
    prediction_column: str,
    target_column: str,
    date_column: str,
    quantile: float,
) -> tuple[float, float, float]:
    if date_column not in frame.columns:
        return 0.0, 0.0, 0.0
    top_returns: list[float] = []
    bottom_returns: list[float] = []
    spreads: list[float] = []
    for group in frame.partition_by(date_column, maintain_order=True):
        if group.height < 2:
            continue
        n_select = max(1, int(math.floor(group.height * quantile)))
        n_select = min(n_select, max(1, group.height // 2))
        ordered = group.sort(prediction_column, descending=True)
        top = float(cast(float, ordered.head(n_select)[target_column].mean()))
        bottom = float(cast(float, ordered.tail(n_select)[target_column].mean()))
        top_returns.append(top)
        bottom_returns.append(bottom)
        spreads.append(top - bottom)
    if not spreads:
        return 0.0, 0.0, 0.0
    return float(np.mean(top_returns)), float(np.mean(bottom_returns)), float(np.mean(spreads))


def evaluate_signal_accuracy(
    frame: pl.DataFrame,
    *,
    prediction_column: str = "prediction",
    target_column: str = "future_return_1",
    date_column: str = "date",
    selection_quantile: float = 0.10,
) -> dict[str, float | int]:
    clean = _clean_signal_frame(frame, prediction_column=prediction_column, target_column=target_column)
    if clean.is_empty():
        return {
            "rows": 0,
            "directional_accuracy": 0.0,
            "positive_precision": 0.0,
            "negative_precision": 0.0,
            "zero_mean_r2": 0.0,
            "information_coefficient": 0.0,
            "rank_ic_mean": 0.0,
            "rank_ic_std": 0.0,
            "top_mean_forward_return": 0.0,
            "bottom_mean_forward_return": 0.0,
            "top_bottom_spread_return": 0.0,
            "positive_signal_share": 0.0,
        }
    y_pred = clean[prediction_column].to_numpy().astype(np.float64)
    y_true = clean[target_column].to_numpy().astype(np.float64)
    positive_mask = y_pred > 0.0
    negative_mask = y_pred < 0.0
    rank_ics = _rank_ic_by_date(
        clean,
        prediction_column=prediction_column,
        target_column=target_column,
        date_column=date_column,
    )
    top, bottom, spread = _top_bottom_returns(
        clean,
        prediction_column=prediction_column,
        target_column=target_column,
        date_column=date_column,
        quantile=selection_quantile,
    )
    return {
        "rows": clean.height,
        "directional_accuracy": float(np.mean((y_pred > 0.0) == (y_true > 0.0))),
        "positive_precision": float(np.mean(y_true[positive_mask] > 0.0)) if np.any(positive_mask) else 0.0,
        "negative_precision": float(np.mean(y_true[negative_mask] < 0.0)) if np.any(negative_mask) else 0.0,
        "zero_mean_r2": _zero_mean_r2(y_true, y_pred),
        "information_coefficient": _safe_corr(y_pred, y_true),
        "rank_ic_mean": float(np.mean(rank_ics)) if rank_ics else 0.0,
        "rank_ic_std": float(np.std(rank_ics)) if rank_ics else 0.0,
        "top_mean_forward_return": top,
        "bottom_mean_forward_return": bottom,
        "top_bottom_spread_return": spread,
        "positive_signal_share": float(np.mean(positive_mask)),
    }


def _max_drawdown_from_returns(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for ret in returns:
        equity *= 1.0 + ret
        peak = max(peak, equity)
        worst = min(worst, (equity - peak) / peak if peak > 0.0 else 0.0)
    return worst


def run_long_short_signal_backtest(
    frame: pl.DataFrame,
    *,
    prediction_column: str = "prediction",
    target_column: str = "future_return_1",
    date_column: str = "date",
    starting_equity: float = 100_000.0,
    selection_fraction: float = 0.10,
    cost_bps: float = 5.0,
    max_symbols_per_side: int | None = None,
) -> EquitySignalBacktestResult:
    if not 0.0 < selection_fraction <= 0.5:
        raise ValueError("selection_fraction must be in (0, 0.5]")
    clean = _clean_signal_frame(frame, prediction_column=prediction_column, target_column=target_column)
    if date_column not in clean.columns:
        raise ValueError(f"missing date column: {date_column}")

    equity = float(starting_equity)
    gross_equity = float(starting_equity)
    rows: list[dict[str, float | int | str]] = []
    for group in clean.sort(date_column).partition_by(date_column, maintain_order=True):
        if group.height < 2:
            continue
        n_select = max(1, int(math.floor(group.height * selection_fraction)))
        n_select = min(n_select, max(1, group.height // 2))
        if max_symbols_per_side is not None:
            n_select = min(n_select, max_symbols_per_side)
        if n_select <= 0:
            continue
        ordered = group.sort(prediction_column, descending=True)
        longs = ordered.head(n_select)
        shorts = ordered.tail(n_select)
        long_ret = float(cast(float, longs[target_column].mean()))
        short_ret = float(cast(float, shorts[target_column].mean()))
        gross_return = 0.5 * long_ret - 0.5 * short_ret
        # Daily dollar-neutral rebalance: enter and exit one gross notional unit per day.
        turnover = 2.0
        cost_return = turnover * cost_bps * 1e-4
        net_return = gross_return - cost_return
        gross_equity *= 1.0 + gross_return
        equity *= 1.0 + net_return
        date_value = group[date_column][0]
        rows.append(
            {
                "date": str(date_value),
                "long_count": int(longs.height),
                "short_count": int(shorts.height),
                "long_mean_forward_return": long_ret,
                "short_mean_forward_return": short_ret,
                "gross_return": gross_return,
                "cost_return": cost_return,
                "net_return": net_return,
                "turnover": turnover,
                "gross_equity": gross_equity,
                "equity": equity,
            }
        )

    if not rows:
        empty = pl.DataFrame(
            {
                "date": [],
                "long_count": [],
                "short_count": [],
                "long_mean_forward_return": [],
                "short_mean_forward_return": [],
                "gross_return": [],
                "cost_return": [],
                "net_return": [],
                "turnover": [],
                "gross_equity": [],
                "equity": [],
            }
        )
        return EquitySignalBacktestResult(
            daily_curve=empty,
            metrics={
                "n_days": 0,
                "total_return": 0.0,
                "gross_total_return": 0.0,
                "cost_drag_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "hit_rate": 0.0,
                "avg_daily_turnover": 0.0,
            },
        )

    curve = pl.DataFrame(rows)
    net_returns = curve["net_return"].to_numpy().astype(np.float64)
    total = float(equity / starting_equity - 1.0)
    gross_total = float(gross_equity / starting_equity - 1.0)
    sigma = float(np.std(net_returns, ddof=1)) if net_returns.size > 1 else 0.0
    mu = float(np.mean(net_returns))
    sharpe = 0.0 if sigma == 0.0 else float(mu / sigma * math.sqrt(252.0))
    n_days = int(curve.height)
    annualized = float((1.0 + total) ** (252.0 / n_days) - 1.0) if n_days > 0 and total > -1.0 else 0.0
    metrics: dict[str, float | int] = {
        "n_days": n_days,
        "total_return": total,
        "gross_total_return": gross_total,
        "cost_drag_return": gross_total - total,
        "annualized_return": annualized,
        "sharpe_ratio": sharpe,
        "max_drawdown": _max_drawdown_from_returns(net_returns.tolist()),
        "hit_rate": float(np.mean(net_returns > 0.0)),
        "avg_daily_turnover": float(cast(float, curve["turnover"].mean())),
        "avg_daily_net_return": mu,
        "avg_daily_gross_return": float(cast(float, curve["gross_return"].mean())),
    }
    return EquitySignalBacktestResult(daily_curve=curve, metrics=metrics)
