"""S4.1α: TradingView paper-validation tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field


class _Window(BaseModel):
    model_config = {"frozen": True}
    min_trading_days: Annotated[int, Field(ge=1)]
    rolling_window_days: Annotated[int, Field(ge=1)]


class _Thresholds(BaseModel):
    model_config = {"frozen": True}
    hit_rate_min: Annotated[float, Field(gt=0.0, lt=1.0)]
    sharpe_min: float
    max_daily_dd_pct: Annotated[float, Field(gt=0.0, lt=1.0)]
    governor_block_rate_max: Annotated[float, Field(gt=0.0, le=1.0)]


class _Data(BaseModel):
    model_config = {"frozen": True}
    forward_return_source: Literal["alpaca_bars", "yfinance", "polygon"]
    horizon_alignment: Literal["ceil_to_next_bar", "floor_to_next_bar"]


class _Artifacts(BaseModel):
    model_config = {"frozen": True}
    daily_report_dir: str
    per_signal_parquet_dir: str


class ValidationConfig(BaseModel):
    model_config = {"frozen": True}
    window: _Window
    thresholds: _Thresholds
    data: _Data
    artifacts: _Artifacts


def load_validation_config(path: Path) -> ValidationConfig:
    with path.open() as h:
        return ValidationConfig.model_validate(yaml.safe_load(h))


__all__ = ["ValidationConfig", "load_validation_config"]
