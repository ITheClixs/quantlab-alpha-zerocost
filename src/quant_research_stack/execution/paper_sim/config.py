from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PaperSimConfig(BaseModel):
    model_config = {"frozen": True}
    symbols: list[str] = Field(min_length=1)
    total_notional_usd: float = Field(default=20_000.0, gt=0.0)
    starting_equity_usd: float = Field(default=100_000.0, gt=0.0)
    leverage: float = Field(default=1.0, gt=0.0, le=1.0)  # 1x only (spec §0)
    half_spread_bps: float = Field(default=1.0, ge=0.0)
    slippage_bps: float = Field(default=4.0, ge=0.0)
    commission_bps: float = Field(default=1.0, ge=0.0)
    rebalance_drift_bps: float = Field(default=25.0, ge=0.0)
    poll_interval_s: float = Field(default=10.0, gt=0.0)
    max_data_gap_seconds: int = Field(default=120, ge=1)


def load_paper_sim_config(path: Path | str) -> PaperSimConfig:
    data = yaml.safe_load(Path(path).read_text())
    return PaperSimConfig.model_validate(data)
