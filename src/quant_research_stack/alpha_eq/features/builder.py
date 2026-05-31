"""Feature builder composition + sha256-locked feature_cols.json (spec §3)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from quant_research_stack.alpha_eq.features.cross_sectional_ranks import (
    build_cross_sectional_ranks,
)
from quant_research_stack.alpha_eq.features.market_regime import build_market_regime
from quant_research_stack.alpha_eq.features.microstructure_proxies import (
    build_microstructure_proxies,
)
from quant_research_stack.alpha_eq.features.noise_sentinel import attach_noise_sentinel
from quant_research_stack.alpha_eq.features.returns_momentum import (
    build_returns_momentum,
)
from quant_research_stack.alpha_eq.features.timestamps import (
    attach_execution_date,
    attach_feature_as_of_date,
)
from quant_research_stack.alpha_eq.features.volatility import build_volatility
from quant_research_stack.alpha_eq.features.volume_liquidity import (
    build_volume_liquidity,
)


@dataclass(frozen=True)
class FeatureBuildConfig:
    momentum_horizons: tuple[int, ...] = (1, 2, 5, 10, 20, 60, 120, 252)
    vol_windows: tuple[int, ...] = (5, 20, 60)
    micro_window: int = 20
    liquidity_window: int = 20
    rank_columns: tuple[str, ...] = (
        "log_return_1", "log_return_5", "log_return_20",
        "realized_vol_20", "dollar_volume", "amihud_illiq_20",
        "overnight_gap", "close_location_20",
    )
    noise_seed: int = 42
    universe_col: str = "in_universe"
    enable_meta_features: bool = False


def build_features(*, panel: pl.DataFrame, config: FeatureBuildConfig) -> pl.DataFrame:
    df = panel
    df = attach_feature_as_of_date(df, convention="after_close_t")
    df = attach_execution_date(df, convention="next_trading_day")
    df = build_returns_momentum(df, horizons=config.momentum_horizons)
    df = build_volatility(df, windows=config.vol_windows)
    df = build_microstructure_proxies(df, window=config.micro_window)
    df = build_volume_liquidity(df, window=config.liquidity_window)
    df = build_market_regime(panel=df, vix=None, spy_close=None)
    df = build_cross_sectional_ranks(
        df, columns=config.rank_columns, universe_col=config.universe_col
    )
    df = attach_noise_sentinel(df, seed=config.noise_seed)
    return df


def _canonical_sha256(columns: Iterable[str]) -> str:
    payload = json.dumps(list(columns), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_feature_cols_json(path: Path, columns: list[str]) -> None:
    blob = {
        "feature_columns": list(columns),
        "feature_cols_sha256": _canonical_sha256(columns),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(blob, separators=(",", ":"), sort_keys=True))
