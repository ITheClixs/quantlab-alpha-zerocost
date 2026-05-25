from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from quant_research_stack.crypto_research.strategies import StrategyVariant


@dataclass(frozen=True)
class BacktestConfig:
    fee_bps: float = 4.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 1.0
    execution_delay_bars: int = 1
    annualization_bars: int = 365 * 24 * 60
    notional_usd: float = 100_000.0
    max_position: float = 1.0
    cost_multiplier: float = 1.0
    invert_signal: bool = False


@dataclass(frozen=True)
class BacktestResult:
    strategy_id: str
    metrics: dict[str, float | int | str | bool]
    trades: pl.DataFrame
    pnl: pl.DataFrame


def _safe_sharpe(returns: np.ndarray, *, annualization: float) -> float:
    values = returns[np.isfinite(returns)]
    if values.size < 2:
        return 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if std <= 0.0 or not math.isfinite(std):
        return 0.0 if mean == 0.0 else math.copysign(1_000_000.0, mean)
    out = mean / std * math.sqrt(annualization)
    return out if math.isfinite(out) else 0.0


def _compound(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.prod(returns + 1.0) - 1.0)


def _max_drawdown(equity: np.ndarray) -> tuple[float, int]:
    if equity.size == 0:
        return 0.0, 0
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    max_dd = float(np.min(drawdowns))
    duration = 0
    max_duration = 0
    for value in drawdowns:
        if value < 0.0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0
    return max_dd if math.isfinite(max_dd) else 0.0, max_duration


def _rolling_z(expr: pl.Expr, window: int) -> pl.Expr:
    mean = expr.rolling_mean(window_size=window, min_samples=window)
    std = expr.rolling_std(window_size=window, min_samples=window)
    return (expr - mean) / (std + 1e-12)


def _score_expression(variant: StrategyVariant) -> pl.Expr:
    params = variant.parameters
    family = variant.family
    feature_set = variant.feature_set
    close = pl.col("close")
    returns = close / close.shift(1) - 1.0
    if family == "baseline" and feature_set == "always_long":
        return (close * 0.0) + 1.0
    if family == "baseline" and feature_set == "always_short":
        return (close * 0.0) - 1.0
    if family == "baseline" and feature_set == "deterministic_random":
        index = pl.int_range(0, pl.len()).cast(pl.Float64)
        return (index * 12.9898).sin()
    if family == "momentum":
        lookback = int(params["lookback"])
        return close / close.shift(lookback) - 1.0
    if family == "mean_reversion":
        lookback = int(params["lookback"])
        z = (close / close.rolling_mean(window_size=lookback, min_samples=lookback) - 1.0) / (
            returns.rolling_std(window_size=lookback, min_samples=lookback) + 1e-12
        )
        return -z
    if family == "breakout":
        window = int(params["window"])
        prior_high = close.rolling_max(window_size=window, min_samples=window).shift(1)
        prior_low = close.rolling_min(window_size=window, min_samples=window).shift(1)
        return pl.when(close > prior_high).then(1.0).when(close < prior_low).then(-1.0).otherwise(0.0)
    if family == "volatility":
        lookback = int(params["lookback"])
        vol_window = int(params["vol_window"])
        return (close / close.shift(lookback) - 1.0) / (
            returns.rolling_std(window_size=vol_window, min_samples=vol_window) + 1e-12
        )
    if family == "liquidity" and feature_set == "maker_ratio_zscore":
        window = int(params["window"])
        return _rolling_z(pl.col("maker_ratio"), window)
    if family == "liquidity" and feature_set == "volume_shock_reversal":
        window = int(params["window"])
        volume_col = pl.col("volume")
        return -_sign_expr(returns) * _rolling_z(volume_col, window)
    if family == "paper_derived" and feature_set == "time_series_momentum_vol_scaled":
        lookback = int(params["lookback"])
        vol_window = int(params["vol_window"])
        return (close / close.shift(lookback) - 1.0) / (
            returns.rolling_std(window_size=vol_window, min_samples=vol_window) + 1e-12
        )
    if family == "paper_derived" and feature_set == "bollinger_reversal":
        lookback = int(params["lookback"])
        return -_rolling_z(close, lookback)
    raise ValueError(f"unsupported strategy variant: {variant.strategy_id}")


def _sign_expr(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr > 0.0).then(1.0).when(expr < 0.0).then(-1.0).otherwise(0.0)


def build_score(frame: pl.DataFrame, variant: StrategyVariant) -> np.ndarray:
    score = (
        frame.select(_score_expression(variant).fill_null(0.0).fill_nan(0.0).alias("score"))
        .get_column("score")
        .to_numpy()
    )
    return np.nan_to_num(score.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _signal_from_score(frame: pl.DataFrame, variant: StrategyVariant, score: np.ndarray) -> np.ndarray:
    threshold = float(variant.parameters.get("threshold", 0.0))
    signal = np.where(score > threshold, 1.0, np.where(score < -threshold, -1.0, 0.0)).astype(np.float64)
    if variant.parameters.get("vol_filter") in {"low", "high"}:
        returns = pl.col("close") / pl.col("close").shift(1) - 1.0
        vol = returns.rolling_std(window_size=120, min_samples=60)
        vol_ref = vol.rolling_mean(window_size=1440, min_samples=120)
        if variant.parameters["vol_filter"] == "low":
            allowed_expr = vol <= vol_ref
        else:
            allowed_expr = vol > vol_ref
        allowed = (
            frame.select(allowed_expr.fill_null(False).alias("allowed"))
            .get_column("allowed")
            .to_numpy()
        )
        signal = np.where(allowed, signal, 0.0)
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)


def build_signal(frame: pl.DataFrame, variant: StrategyVariant) -> np.ndarray:
    return _signal_from_score(frame, variant, build_score(frame, variant))


def _shift(values: np.ndarray, periods: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if periods <= 0:
        return values.astype(np.float64)
    if periods < values.size:
        out[periods:] = values[:-periods]
    return out


def _apply_holding_horizon(signal: np.ndarray, horizon: int) -> np.ndarray:
    if horizon <= 1:
        return signal.astype(np.float64)
    out = np.zeros_like(signal, dtype=np.float64)
    current = 0.0
    remaining = 0
    for index, value in enumerate(signal):
        if value != 0.0:
            current = float(value)
            remaining = horizon
        elif remaining <= 0:
            current = 0.0
        out[index] = current
        if remaining > 0:
            remaining -= 1
    return out


def run_variant_backtest(
    frame: pl.DataFrame,
    variant: StrategyVariant,
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    if frame.height < 3:
        raise ValueError("backtest frame must have at least 3 rows")
    ordered = frame.sort("timestamp")
    close = ordered.get_column("close").to_numpy().astype(np.float64)
    timestamps = ordered.get_column("timestamp")
    score = build_score(ordered, variant)
    signal = _signal_from_score(ordered, variant, score)
    if cfg.invert_signal:
        signal = -signal
    raw_position = np.clip(_apply_holding_horizon(signal, variant.horizon), -cfg.max_position, cfg.max_position)
    delayed_score = _shift(score, cfg.execution_delay_bars)
    delayed_signal = _shift(signal, cfg.execution_delay_bars)
    position = _shift(raw_position, cfg.execution_delay_bars)
    returns = np.zeros_like(close, dtype=np.float64)
    returns[1:] = close[1:] / close[:-1] - 1.0
    prior_position = np.zeros_like(position)
    prior_position[1:] = position[:-1]
    turnover = np.abs(position - prior_position)
    per_unit_cost = (cfg.fee_bps + cfg.half_spread_bps + cfg.slippage_bps) * 1e-4 * cfg.cost_multiplier
    cost_returns = turnover * per_unit_cost
    gross_returns = prior_position * returns
    net_returns = gross_returns - cost_returns
    equity = np.cumprod(1.0 + net_returns)
    trade_mask = turnover > 0.0
    trade_net_returns = net_returns[trade_mask]
    pnl = pl.DataFrame(
        {
            "timestamp": timestamps,
            "strategy_id": [variant.strategy_id] * ordered.height,
            "signal": signal,
            "position": prior_position,
            "gross_return": gross_returns,
            "turnover": turnover,
            "cost_return": cost_returns,
            "net_return": net_returns,
            "equity": equity,
        }
    )
    trade_indices = np.flatnonzero(trade_mask)
    exit_indices = np.minimum(trade_indices + max(variant.horizon, 1), close.size - 1)
    half_spread = cfg.half_spread_bps * 1e-4 * cfg.cost_multiplier
    entry_mid = close[trade_indices]
    exit_mid = close[exit_indices]
    side = np.where(position[trade_mask] > 0.0, "long", np.where(position[trade_mask] < 0.0, "short", "flat"))
    side_sign = np.where(position[trade_mask] > 0.0, 1.0, np.where(position[trade_mask] < 0.0, -1.0, 0.0))
    realized_mid_return = side_sign * (exit_mid / np.maximum(entry_mid, 1e-12) - 1.0)
    estimated_round_trip_cost = (
        cfg.fee_bps + cfg.half_spread_bps + cfg.slippage_bps
    ) * 1e-4 * cfg.cost_multiplier * np.maximum(turnover[trade_mask], 1.0)
    trades = pl.DataFrame(
        {
            "timestamp": timestamps.filter(pl.Series(trade_mask)),
            "strategy_id": [variant.strategy_id] * int(np.sum(trade_mask)),
            "side": side,
            "prediction": position[trade_mask],
            "raw_score": delayed_score[trade_mask],
            "signal": delayed_signal[trade_mask],
            "position": position[trade_mask],
            "entry_mid": entry_mid,
            "entry_bid": entry_mid * (1.0 - half_spread),
            "entry_ask": entry_mid * (1.0 + half_spread),
            "exit_mid": exit_mid,
            "exit_bid": exit_mid * (1.0 - half_spread),
            "exit_ask": exit_mid * (1.0 + half_spread),
            "realized_mid_return": realized_mid_return,
            "gross_return": gross_returns[trade_mask],
            "spread_cost": turnover[trade_mask] * cfg.half_spread_bps * 1e-4 * cfg.cost_multiplier,
            "fee_cost": turnover[trade_mask] * cfg.fee_bps * 1e-4 * cfg.cost_multiplier,
            "slippage_cost": turnover[trade_mask] * cfg.slippage_bps * 1e-4 * cfg.cost_multiplier,
            "cost_return": cost_returns[trade_mask],
            "net_return": net_returns[trade_mask],
            "turnover": turnover[trade_mask],
            "holding_horizon": [variant.horizon] * int(np.sum(trade_mask)),
            "direction_correct": realized_mid_return > 0.0,
            "estimated_round_trip_cost": estimated_round_trip_cost,
            "edge_to_cost_ratio": np.abs(delayed_score[trade_mask]) / np.maximum(estimated_round_trip_cost, 1e-12),
        }
    )
    daily = (
        pnl.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg(((pl.col("net_return") + 1.0).product() - 1.0).alias("daily_return"))
        .sort("date")
    )
    daily_returns = daily.get_column("daily_return").to_numpy().astype(np.float64) if daily.height else np.asarray([])
    metrics = _metrics_from_arrays(
        strategy_id=variant.strategy_id,
        family=variant.family,
        gross_returns=gross_returns,
        net_returns=net_returns,
        positions=prior_position,
        turnover=turnover,
        trade_net_returns=trade_net_returns,
        daily_returns=daily_returns,
        capacity_estimate_usd=float(np.nanmean(ordered.get_column("liquidity_sum").to_numpy()))
        if "liquidity_sum" in ordered.columns
        else 0.0,
        config=cfg,
    )
    return BacktestResult(strategy_id=variant.strategy_id, metrics=metrics, trades=trades, pnl=pnl)


def _metrics_from_arrays(
    *,
    strategy_id: str,
    family: str,
    gross_returns: np.ndarray,
    net_returns: np.ndarray,
    positions: np.ndarray,
    turnover: np.ndarray,
    trade_net_returns: np.ndarray,
    daily_returns: np.ndarray,
    capacity_estimate_usd: float,
    config: BacktestConfig,
) -> dict[str, float | int | str | bool]:
    equity = np.cumprod(1.0 + net_returns)
    max_drawdown, drawdown_duration = _max_drawdown(equity)
    active = positions != 0.0
    positive = net_returns[active & (net_returns > 0.0)]
    negative = net_returns[active & (net_returns < 0.0)]
    gross_active = gross_returns[active]
    net_active = net_returns[active]
    long_returns = net_returns[positions > 0.0]
    short_returns = net_returns[positions < 0.0]
    trade_count = int(np.count_nonzero(turnover > 0.0))
    return {
        "strategy_id": strategy_id,
        "family": family,
        "gross_total_return": _compound(gross_returns),
        "net_total_return": _compound(net_returns),
        "net_daily_sharpe": _safe_sharpe(daily_returns, annualization=365.0),
        "per_trade_sharpe": _safe_sharpe(trade_net_returns, annualization=max(float(trade_net_returns.size), 1.0)),
        "max_drawdown": max_drawdown,
        "drawdown_duration_bars": drawdown_duration,
        "gross_hit_rate": float(np.mean(gross_active > 0.0)) if gross_active.size else 0.0,
        "net_hit_rate": float(np.mean(net_active > 0.0)) if net_active.size else 0.0,
        "average_win": float(np.mean(positive)) if positive.size else 0.0,
        "average_loss": float(np.mean(negative)) if negative.size else 0.0,
        "profit_factor": float(np.sum(positive) / abs(np.sum(negative))) if negative.size and abs(np.sum(negative)) > 0.0 else 0.0,
        "turnover": float(np.sum(turnover)),
        "trade_count": trade_count,
        "average_holding_bars": float(np.sum(active) / max(trade_count, 1)),
        "capacity_estimate_usd": capacity_estimate_usd,
        "long_total_return": _compound(long_returns),
        "short_total_return": _compound(short_returns),
        "execution_delay_bars": config.execution_delay_bars,
        "cost_multiplier": config.cost_multiplier,
        "fee_bps": config.fee_bps,
        "half_spread_bps": config.half_spread_bps,
        "slippage_bps": config.slippage_bps,
    }


def summarize_backtest_frames(
    *,
    strategy_id: str,
    family: str,
    pnl: pl.DataFrame,
    trades: pl.DataFrame,
    config: BacktestConfig,
    period_name: str,
) -> dict[str, float | int | str | bool]:
    if pnl.is_empty():
        return {
            "period": period_name,
            "strategy_id": strategy_id,
            "family": family,
            "gross_total_return": 0.0,
            "net_total_return": 0.0,
            "net_daily_sharpe": 0.0,
            "per_trade_sharpe": 0.0,
            "max_drawdown": 0.0,
            "drawdown_duration_bars": 0,
            "gross_hit_rate": 0.0,
            "net_hit_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "turnover": 0.0,
            "trade_count": 0,
            "average_holding_bars": 0.0,
            "capacity_estimate_usd": 0.0,
            "long_total_return": 0.0,
            "short_total_return": 0.0,
            "execution_delay_bars": config.execution_delay_bars,
            "cost_multiplier": config.cost_multiplier,
            "fee_bps": config.fee_bps,
            "half_spread_bps": config.half_spread_bps,
            "slippage_bps": config.slippage_bps,
        }
    daily = (
        pnl.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg(((pl.col("net_return") + 1.0).product() - 1.0).alias("daily_return"))
        .sort("date")
    )
    metrics = _metrics_from_arrays(
        strategy_id=strategy_id,
        family=family,
        gross_returns=pnl.get_column("gross_return").to_numpy().astype(np.float64),
        net_returns=pnl.get_column("net_return").to_numpy().astype(np.float64),
        positions=pnl.get_column("position").to_numpy().astype(np.float64),
        turnover=pnl.get_column("turnover").to_numpy().astype(np.float64),
        trade_net_returns=trades.get_column("net_return").to_numpy().astype(np.float64)
        if "net_return" in trades.columns
        else np.asarray([]),
        daily_returns=daily.get_column("daily_return").to_numpy().astype(np.float64) if daily.height else np.asarray([]),
        capacity_estimate_usd=0.0,
        config=config,
    )
    return {"period": period_name, **metrics}


def metrics_by_period(
    frame: pl.DataFrame,
    variant: StrategyVariant,
    *,
    config: BacktestConfig,
    period_name: str,
) -> dict[str, float | int | str | bool]:
    result = run_variant_backtest(frame, variant, config=config)
    return {"period": period_name, **result.metrics}
