"""Compose VWAP primary + fingerprint features + eligibility + meta walk-forward,
then net-of-cost metrics and the lift-vs-baseline test (spec §6-7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from quant_research_stack.crypto_research.perps.validation import deflated_sharpe_payload
from quant_research_stack.signal_research.fingerprint_vwap.eligibility import primary_signal_stats
from quant_research_stack.signal_research.fingerprint_vwap.fingerprint import (
    build_fingerprint_features,
    fingerprint_columns,
)
from quant_research_stack.signal_research.fingerprint_vwap.vwap import daily_vwap_proxy, vwap_primary_position
from quant_research_stack.signal_research.methodology.meta_labeling import check_eligibility
from quant_research_stack.signal_research.papers.triple_barrier import TripleBarrierConfig
from quant_research_stack.signal_research.training.meta_label_walk_forward import (
    MetaLabelWalkForwardConfig,
    train_meta_label_walk_forward,
)


@dataclass(frozen=True)
class FingerprintVwapSpec:
    windows: tuple[int, ...] = (20, 60, 120, 252)
    vwap_window: int = 5
    band: float = 0.0
    horizon_days: int = 3
    cost_bps_one_way: float = 1.0
    train_window_days: int = 252
    test_window_days: int = 63
    step_days: int = 63
    min_train_events: int = 200


def _baseline_net_sharpe(prepared: pl.DataFrame, *, horizon: int, cost_bps_one_way: float) -> float:
    """Take EVERY eligible VWAP entry (no meta filter); net-of-cost annualized Sharpe."""
    cost = 2.0 * cost_bps_one_way / 1e4
    df = prepared.sort(["symbol", "date"]).with_columns(
        (pl.col("close").shift(-horizon).over("symbol") / pl.col("close") - 1.0).alias("fwd")
    )
    r = df.filter((pl.col("primary_position") == 1.0) & pl.col("fwd").is_finite())["fwd"].to_numpy().astype(np.float64) - cost
    if r.size < 2 or np.std(r, ddof=1) == 0.0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252.0 / horizon))


def run_fingerprint_vwap_meta(*, panel: pl.DataFrame, spec: FingerprintVwapSpec) -> dict[str, Any]:
    prepared = vwap_primary_position(
        build_fingerprint_features(daily_vwap_proxy(panel, window=spec.vwap_window), windows=spec.windows),
        band=spec.band,
    )
    stats = primary_signal_stats(prepared, horizon_days=spec.horizon_days, cost_bps_one_way=spec.cost_bps_one_way)
    elig = check_eligibility(stats)
    out: dict[str, Any] = {"eligibility": {"eligible": elig.eligible, "reason": elig.rejection_reason,
                                           "primary_net_sharpe": stats.validation_net_sharpe,
                                           "event_count": stats.event_count}}
    if not elig.eligible:
        out["status"] = "primary_ineligible"
        return out
    cfg = MetaLabelWalkForwardConfig(
        train_window_days=spec.train_window_days, test_window_days=spec.test_window_days,
        step_days=spec.step_days, min_train_events=spec.min_train_events,
        cost_bps_one_way=spec.cost_bps_one_way,
        triple_barrier=TripleBarrierConfig(vertical_barrier_days=spec.horizon_days),
        primary_position_col="primary_position",
        extra_feature_columns=fingerprint_columns(spec.windows),
    )
    result = train_meta_label_walk_forward(panel=prepared, config=cfg)
    meta_sharpe = float(result.summary.get("net_sharpe", 0.0))
    baseline = _baseline_net_sharpe(prepared, horizon=spec.horizon_days, cost_bps_one_way=spec.cost_bps_one_way)
    out.update({
        "status": "evaluated",
        "meta_net_sharpe": meta_sharpe,
        "baseline_net_sharpe": baseline,
        "lift": meta_sharpe - baseline,
        "summary": result.summary,
        "fold_metrics": result.fold_metrics,
        "predictions": result.predictions,
    })
    return out


def render_report(*, result: dict[str, Any], verdict: dict[str, Any], spec_repr: str) -> str:
    elig = result.get("eligibility", {})
    lines = [
        "# Fingerprint-VWAP Meta-Labeling v1 — Result",
        "",
        "**Status:** research_only. Not investment advice. No paper. No live.",
        "",
        f"**Verdict:** {verdict['verdict']}",
        "",
        "## Eligibility (primary VWAP entry)",
        f"- eligible: {elig.get('eligible')}  reason: {elig.get('reason') or 'n/a'}",
        f"- primary net Sharpe: {elig.get('primary_net_sharpe'):.3f}  events: {elig.get('event_count')}",
        "",
        "## Meta-labeling (net of cost)",
        f"- meta net Sharpe: {result.get('meta_net_sharpe', float('nan')):.3f}",
        f"- baseline (take-every-entry) net Sharpe: {result.get('baseline_net_sharpe', float('nan')):.3f}",
        f"- **lift**: {result.get('lift', float('nan')):.3f}",
        f"- deflated Sharpe: {verdict.get('deflated_sharpe')}",
        f"- failed gates: {verdict.get('failed') or 'none'}",
        "",
        "## Spec",
        f"`{spec_repr}`",
    ]
    return "\n".join(lines)


def gate_verdict(
    *,
    meta_net_sharpe: float,
    baseline_net_sharpe: float,
    lift_margin: float,
    daily_net_returns: list[float],
    trials: int,
) -> dict[str, Any]:
    """PASS only if net Sharpe>0, deflated-Sharpe prob>=0.95 at `trials`, and
    lift = meta - baseline > lift_margin. Otherwise DO_NOT_ADVANCE with reasons."""
    returns_arr = np.asarray(daily_net_returns, dtype=np.float64)
    dsr = deflated_sharpe_payload(returns_arr, trials=trials)
    dsr_prob = float(dsr.get("probability", 0.0))
    lift = meta_net_sharpe - baseline_net_sharpe
    failed: list[str] = []
    if meta_net_sharpe <= 0.0:
        failed.append("net_sharpe")
    if dsr_prob < 0.95:
        failed.append("deflated_sharpe")
    if lift <= lift_margin:
        failed.append("lift")
    return {
        "verdict": "PASS" if not failed else "DO_NOT_ADVANCE",
        "passed": not failed,
        "failed": failed,
        "net_sharpe": meta_net_sharpe,
        "lift": lift,
        "deflated_sharpe": dsr,
    }
