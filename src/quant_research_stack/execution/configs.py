from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, Field, model_validator


class RiskLimits(BaseModel):
    model_config = {"frozen": True}
    max_per_symbol_pct: Annotated[float, Field(gt=0.0, lt=1.0)]
    max_gross_exposure_pct: Annotated[float, Field(gt=0.0, le=1.0)]
    base_notional_per_trade_pct: Annotated[float, Field(gt=0.0, lt=1.0)]
    max_orders_per_minute: Annotated[int, Field(ge=1)]


class DrawdownLimits(BaseModel):
    model_config = {"frozen": True}
    daily_realized_dd_kill_pct: Annotated[float, Field(gt=0.0, lt=1.0)]
    cumulative_dd_kill_pct: Annotated[float, Field(gt=0.0, lt=1.0)]


class Freshness(BaseModel):
    model_config = {"frozen": True}
    crypto_max_gap_seconds: Annotated[int, Field(ge=1)]
    equity_max_gap_seconds: Annotated[int, Field(ge=1)]


class Reconciliation(BaseModel):
    model_config = {"frozen": True}
    interval_seconds: Annotated[int, Field(ge=1)]
    max_diff_bps: Annotated[float, Field(gt=0.0)]


class StageOverrides(BaseModel):
    model_config = {"frozen": True}
    cap_multiplier_first_30d: Annotated[float, Field(gt=0.0, le=1.0)] = 0.50


class RiskConfig(BaseModel):
    model_config = {"frozen": True}
    limits: RiskLimits
    drawdown: DrawdownLimits
    freshness: Freshness
    reconciliation: Reconciliation
    stage_overrides: dict[str, StageOverrides] = {}

    @model_validator(mode="after")
    def _cumulative_above_daily(self) -> RiskConfig:
        if self.drawdown.cumulative_dd_kill_pct <= self.drawdown.daily_realized_dd_kill_pct:
            raise ValueError("cumulative_dd_kill_pct must exceed daily_realized_dd_kill_pct")
        return self


class GateRow(BaseModel):
    model_config = {"frozen": True}
    min_days_in_paper: int = 0
    min_days_in_live_shadow: int = 0
    min_sharpe: float | None = None
    max_daily_dd_pct: float | None = None
    no_kill_triggers_days: int | None = None
    max_audit_anomalies: int | None = None
    max_reconciliation_diff_bps: float | None = None
    max_feed_gap_violations: int | None = None
    kill_switch_drill_passed: bool | None = None
    required_artifacts: list[str] = []


class PromotionConfig(BaseModel):
    model_config = {"frozen": True}
    paper_to_live_shadow: GateRow
    live_shadow_to_live: GateRow


class IngestConfig(BaseModel):
    model_config = {"frozen": True}
    s1_predictions_dir: str
    s2_verdicts_dir: str
    poll_interval_seconds: Annotated[float, Field(gt=0.0)]
    pair_window_seconds: Annotated[int, Field(ge=1)]


class PositionBookConfig(BaseModel):
    model_config = {"frozen": True}
    snapshot_root: str
    snapshot_interval_seconds: Annotated[int, Field(ge=1)]


class AuditCfg(BaseModel):
    model_config = {"frozen": True}
    root: str
    rotation: str = "daily"
    chmod_after_close: bool = True


class KillSwitchCfg(BaseModel):
    model_config = {"frozen": True}
    repo_root_marker: str
    poll_interval_seconds: Annotated[float, Field(gt=0.0)]
    emergency_snapshot_root: str


class ExecConfig(BaseModel):
    model_config = {"frozen": True}
    ingest: IngestConfig
    position_book: PositionBookConfig
    audit: AuditCfg
    kill_switch: KillSwitchCfg


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as h:
        return yaml.safe_load(h)


def load_risk_config(path: Path) -> RiskConfig:
    return RiskConfig.model_validate(_load_yaml(path))


def load_promotion_config(path: Path) -> PromotionConfig:
    return PromotionConfig.model_validate(_load_yaml(path))


def load_exec_config(path: Path) -> ExecConfig:
    return ExecConfig.model_validate(_load_yaml(path))
