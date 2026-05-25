from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyVariant:
    strategy_id: str
    family: str
    feature_set: str
    parameters: dict[str, Any]
    horizon: int
    entry_rule: str
    exit_rule: str
    execution_assumption: str
    cost_assumption: str
    source: str = "internal"

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["parameters_json"] = json.dumps(self.parameters, sort_keys=True)
        row.pop("parameters")
        return row


def _strategy_id(family: str, feature_set: str, params: dict[str, Any], horizon: int) -> str:
    raw = json.dumps(
        {"family": family, "feature_set": feature_set, "params": params, "horizon": horizon},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{family}_{feature_set}_h{horizon}_{digest}"


def _variant(
    *,
    family: str,
    feature_set: str,
    params: dict[str, Any],
    horizon: int,
    entry_rule: str,
    source: str = "internal",
) -> StrategyVariant:
    return StrategyVariant(
        strategy_id=_strategy_id(family, feature_set, params, horizon),
        family=family,
        feature_set=feature_set,
        parameters=params,
        horizon=horizon,
        entry_rule=entry_rule,
        exit_rule="rebalance each bar; flat when signal is zero",
        execution_assumption="one-bar delayed taker execution unless stress test overrides delay",
        cost_assumption="per-unit turnover cost = fee + half spread + slippage; funding is zero unless dataset supplies funding",
        source=source,
    )


def generate_strategy_variants(*, target_count: int = 1500) -> list[StrategyVariant]:
    if target_count < 1:
        raise ValueError("target_count must be positive")
    variants: list[StrategyVariant] = []
    horizons = [1, 3, 5, 10, 15, 30, 60]
    lookbacks = [5, 10, 15, 30, 60, 120, 240, 480, 720, 1440]
    thresholds = [0.0, 0.0001, 0.00025, 0.0005, 0.001]
    vol_filters = ["none", "low", "high"]

    for lookback in lookbacks:
        for threshold in thresholds:
            for vol_filter in vol_filters:
                for horizon in horizons:
                    variants.append(
                        _variant(
                            family="momentum",
                            feature_set="close_return",
                            params={"lookback": lookback, "threshold": threshold, "vol_filter": vol_filter},
                            horizon=horizon,
                            entry_rule="sign(close / close.shift(lookback) - 1) when abs(score) > threshold",
                        )
                    )
                    variants.append(
                        _variant(
                            family="mean_reversion",
                            feature_set="rolling_zscore",
                            params={"lookback": lookback, "threshold": threshold, "vol_filter": vol_filter},
                            horizon=horizon,
                            entry_rule="-sign((close / rolling_mean(close, lookback) - 1) / rolling_std(return, lookback))",
                        )
                    )

    breakout_windows = [20, 30, 60, 120, 240, 480, 720, 1440]
    for window in breakout_windows:
        for threshold in thresholds:
            for horizon in horizons:
                variants.append(
                    _variant(
                        family="breakout",
                        feature_set="rolling_channel",
                        params={"window": window, "threshold": threshold},
                        horizon=horizon,
                        entry_rule="long above prior rolling max, short below prior rolling min",
                    )
                )
                variants.append(
                    _variant(
                        family="volatility",
                        feature_set="vol_adjusted_trend",
                        params={"lookback": window, "vol_window": max(10, window // 2), "threshold": threshold},
                        horizon=horizon,
                        entry_rule="sign(lookback return / realized volatility) when abs(score) > threshold",
                    )
                )

    liquidity_windows = [30, 60, 120, 240, 480, 720, 1440]
    for window in liquidity_windows:
        for threshold in thresholds:
            for horizon in horizons:
                variants.append(
                    _variant(
                        family="liquidity",
                        feature_set="maker_ratio_zscore",
                        params={"window": window, "threshold": threshold},
                        horizon=horizon,
                        entry_rule="sign(zscore(maker_ratio, window)) as order-flow proxy",
                    )
                )
                variants.append(
                    _variant(
                        family="liquidity",
                        feature_set="volume_shock_reversal",
                        params={"window": window, "threshold": threshold},
                        horizon=horizon,
                        entry_rule="-sign(last return) when volume z-score exceeds threshold",
                    )
                )

    for lookback in lookbacks:
        for threshold in thresholds:
            for horizon in horizons:
                variants.append(
                    _variant(
                        family="paper_derived",
                        feature_set="time_series_momentum_vol_scaled",
                        params={"lookback": lookback, "vol_window": lookback, "threshold": threshold},
                        horizon=horizon,
                        entry_rule="time-series momentum scaled by realized volatility",
                        source="Moskowitz-Ooi-Pedersen-style time-series momentum translated to BTCUSDT bars",
                    )
                )
                variants.append(
                    _variant(
                        family="paper_derived",
                        feature_set="bollinger_reversal",
                        params={"lookback": lookback, "threshold": max(threshold, 0.5)},
                        horizon=horizon,
                        entry_rule="Bollinger-style reversal from rolling mean deviation",
                        source="Bollinger/z-score mean-reversion baseline",
                    )
                )
    return _select_balanced(_dedupe(variants), target_count=target_count)


def _dedupe(variants: list[StrategyVariant]) -> list[StrategyVariant]:
    seen: set[str] = set()
    out: list[StrategyVariant] = []
    for variant in variants:
        if variant.strategy_id in seen:
            continue
        seen.add(variant.strategy_id)
        out.append(variant)
    return out


def _select_balanced(variants: list[StrategyVariant], *, target_count: int) -> list[StrategyVariant]:
    by_family: dict[str, list[StrategyVariant]] = {}
    for variant in variants:
        by_family.setdefault(variant.family, []).append(variant)
    out: list[StrategyVariant] = []
    families = sorted(by_family)
    cursor = 0
    while len(out) < target_count and any(by_family.values()):
        family = families[cursor % len(families)]
        if by_family[family]:
            out.append(by_family[family].pop(0))
        cursor += 1
    return out
