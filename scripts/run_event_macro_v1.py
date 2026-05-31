"""event_conditioned_macro_v1 — frozen variant pool + baselines + validation.

Pre-registration: docs/research/intake/2026-05-30-event-conditioned-macro-calendar-v1.md
Data gate: reports/signal_research/event_macro_v1/event_timestamp_audit.md (PASS).
FOMC-only v1 (CPI/NFP deferred). research_only; single-index risk timing.

Emits: event_strategy_registry.parquet, event_conditioned_validation_report.md,
event_placebo_report.md, event_failure_classification.md.

Usage:
    PYTHONPATH=src uv run python scripts/run_event_macro_v1.py
"""

from __future__ import annotations

import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from rich.console import Console

from quant_research_stack.crypto_research.perps.validation import (
    bootstrap_sharpe_payload,
    deflated_sharpe_payload,
    estimate_registry_pbo,
)
from quant_research_stack.signal_research.events.calendar import attach_event_features, load_fomc_dates
from quant_research_stack.signal_research.events.strategies import (
    backtest_positions,
    buy_and_hold,
    daily_returns,
    risk_off_gate,
    risk_on_gate,
    sma_gate,
    vol_regime_gate,
    vol_target_event,
    vol_target_position,
)

console = Console()
_BARS = "data/processed/vrp/bars/{sym}.parquet"
_SYMBOLS = ("SPY", "QQQ")
_OUT = Path("reports/signal_research/event_macro_v1")
_COST_BPS = 1.0
# Diagnostic-only strategies are reported but not eligible to be the promotable best.
_DIAGNOSTIC = ("fomc_riskoff_win5", "hmm_overlay")
_BASELINES = ("bah", "voltarget_bah", "sma_gate", "volregime_gate",
              "placebo_random_riskoff", "placebo_shifted_riskoff")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _ann_sharpe(net: np.ndarray) -> float:
    f = net[np.isfinite(net)]
    if f.size < 2:
        return 0.0
    sd = float(np.std(f, ddof=1))
    return float(np.mean(f) / sd * np.sqrt(252.0)) if sd > 0 else 0.0


def _shifted_dates(dates: list[date], fomc: set[date], offset: int) -> set[date]:
    pos = {d: i for i, d in enumerate(dates)}
    out: set[date] = set()
    for d in fomc:
        i = pos.get(d)
        if i is not None and 0 <= i + offset < len(dates):
            out.add(dates[i + offset])
    return out


def _build_symbol(sym: str, fomc: list[date], rng: np.random.Generator) -> dict[str, Any]:
    bars = pl.read_parquet(_BARS.format(sym=sym)).sort("date")
    feats = attach_event_features(bars, fomc_dates=fomc)
    dates = [d if isinstance(d, date) else d.date() for d in feats["date"].to_list()]
    close = feats["close"].to_numpy().astype(np.float64)
    r = daily_returns(close)
    fomc_in = set(d for d in fomc if d in set(dates))

    # placebo calendars (same count as real FOMC in-range)
    random_dates = set(rng.choice(np.array(dates, dtype=object), size=len(fomc_in), replace=False).tolist())
    shifted_dates = _shifted_dates(dates, fomc_in, 10)
    rnd_flag = attach_event_features(bars, fomc_dates=list(random_dates))["fomc_win2"].to_numpy()
    shf_flag = attach_event_features(bars, fomc_dates=list(shifted_dates))["fomc_win2"].to_numpy()

    flags = {w: feats[f"fomc_{w}"].to_numpy() for w in ("tm1", "t0", "tp1", "win2", "win5")}
    vt = vol_target_position(r)
    vr = vol_regime_gate(r)
    positions: dict[str, np.ndarray] = {
        "bah": buy_and_hold(len(r)),
        "voltarget_bah": vt,
        "sma_gate": sma_gate(close),
        "volregime_gate": vr,
        "fomc_riskoff_tm1": risk_off_gate(flags["tm1"]),
        "fomc_riskoff_t0": risk_off_gate(flags["t0"]),
        "fomc_riskoff_tp1": risk_off_gate(flags["tp1"]),
        "fomc_riskoff_win2": risk_off_gate(flags["win2"]),
        "fomc_riskon_tm1": risk_on_gate(flags["tm1"]),
        "fomc_riskon_tp1": risk_on_gate(flags["tp1"]),
        "voltarget_event": vol_target_event(r, flags["win2"], off_scale=0.0),
        "placebo_random_riskoff": risk_off_gate(rnd_flag),
        "placebo_shifted_riskoff": risk_off_gate(shf_flag),
        "fomc_riskoff_win5": risk_off_gate(flags["win5"]),
        "hmm_overlay": np.minimum(vr, risk_off_gate(flags["win2"])),
    }
    return {"dates": dates, "returns": r, "positions": positions}


def _crisis_sharpe(net: np.ndarray, years: np.ndarray, drop: set[int]) -> float:
    mask = ~np.isin(years, list(drop))
    return _ann_sharpe(net[mask])


def main() -> int:
    fomc = load_fomc_dates()
    rng = np.random.default_rng(20260530)
    registry: dict[str, dict[str, Any]] = {}
    per_strategy: list[dict[str, Any]] = []
    ret_matrix: dict[str, np.ndarray] = {}
    ref_dates: list[date] | None = None

    for sym in _SYMBOLS:
        built = _build_symbol(sym, fomc, rng)
        dates = built["dates"]
        years = np.array([d.year for d in dates])
        if ref_dates is None:
            ref_dates = dates
        for name, pos in built["positions"].items():
            base = backtest_positions(built["returns"], pos, cost_oneway_bps=_COST_BPS)
            c2x = backtest_positions(built["returns"], pos, cost_oneway_bps=2 * _COST_BPS)
            d1 = backtest_positions(built["returns"], pos, cost_oneway_bps=_COST_BPS, delay=1)
            key = f"{name}__{sym}"
            ret_matrix[key] = base.net_returns if dates == ref_dates else _align(base.net_returns, dates, ref_dates)
            per_strategy.append({
                "strategy": key, "symbol": sym, "family": name,
                "diagnostic": name in _DIAGNOSTIC, "baseline": name in _BASELINES,
                "sharpe": round(base.metrics["sharpe"], 4),
                "calmar": round(base.metrics["calmar"], 4),
                "max_drawdown": round(base.metrics["max_drawdown"], 4),
                "total_return": round(base.metrics["total_return"], 4),
                "exposure": round(base.metrics["exposure"], 4),
                "turnover": round(base.metrics["turnover"], 4),
                "sharpe_2x_cost": round(c2x.metrics["sharpe"], 4),
                "sharpe_delay1": round(d1.metrics["sharpe"], 4),
                "sharpe_ex2020": round(_crisis_sharpe(base.net_returns, years, {2020}), 4),
                "sharpe_ex2022": round(_crisis_sharpe(base.net_returns, years, {2022}), 4),
                "sharpe_ex2023_26": round(_crisis_sharpe(base.net_returns, years, {2023, 2024, 2025, 2026}), 4),
            })
            registry[key] = base.metrics

    # cross-strategy PBO + DSR over the full pool
    matrix_df = pl.DataFrame({k: v for k, v in ret_matrix.items()}).with_row_index("event_index")
    pbo = estimate_registry_pbo(matrix_df, strategy_columns=list(ret_matrix.keys()))
    # best promotable variant (exclude baselines + diagnostics)
    promotable = [row for row in per_strategy if not row["baseline"] and not row["diagnostic"]]
    best = max(promotable, key=lambda x: x["sharpe"]) if promotable else {}
    best_key = best.get("strategy", "")
    boot = bootstrap_sharpe_payload(ret_matrix.get(best_key, np.array([]))) if best_key else {}
    dsr = deflated_sharpe_payload(ret_matrix.get(best_key, np.array([])), trials=len(ret_matrix)) if best_key else {}

    by_key = {row["strategy"]: row for row in per_strategy}
    best_sym = best.get("symbol", "SPY")

    def metric(name: str, field: str = "sharpe") -> float:
        return float(by_key.get(f"{name}__{best_sym}", {}).get(field, 0.0))

    placebo_max = max(metric("placebo_random_riskoff"), metric("placebo_shifted_riskoff"))
    classification = _classify(best, metric, placebo_max, pbo, boot, dsr)

    _OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(per_strategy).write_parquet(_OUT / "event_strategy_registry.parquet")
    _write_reports(per_strategy, pbo, boot, dsr, best, best_sym, metric, placebo_max, classification, len(fomc))
    console.print(f"[bold]Best promotable:[/bold] {best_key} sharpe={best.get('sharpe')}")
    console.print(f"[bold]Classification:[/bold] {classification}")
    return 0


def _align(net: np.ndarray, dates: list[date], ref: list[date]) -> np.ndarray:
    m = dict(zip(dates, net, strict=False))
    return np.array([m.get(d, 0.0) for d in ref], dtype=np.float64)


def _classify(best: dict, metric, placebo_max: float, pbo: dict, boot: dict, dsr: dict) -> dict[str, Any]:
    if not best:
        return {"status": "none", "failure_class": "no_event_edge", "blockers": ["no_promotable_variant"]}
    s = best["sharpe"]
    bah = metric("bah")
    vtb = metric("voltarget_bah")
    blockers: list[str] = []
    if s < 1.5:
        blockers.append("below_1.5_sharpe_gate")
    if s <= bah and s <= vtb:
        blockers.append("no_improvement_over_baselines")
    if s <= vtb:
        blockers.append("subsumed_by_vol_targeting")
    if s <= placebo_max + 0.20:
        blockers.append("placebo_indistinguishable")
    pbo_p = pbo.get("pbo_probability")
    if pbo_p is None or pbo_p > 0.25:  # note: 0.0 is a PASS — avoid falsy-zero `or` trap
        blockers.append("high_pbo")
    ci_lower = boot.get("ci_lower_95")
    if ci_lower is None or ci_lower <= 0.0:
        blockers.append("bootstrap_ci_not_positive")
    if best["sharpe_2x_cost"] <= 0.0 or best["sharpe_delay1"] <= 0.0:
        blockers.append("fails_cost_or_delay_stress")
    if min(best["sharpe_ex2020"], best["sharpe_ex2022"], best["sharpe_ex2023_26"]) <= 0.0:
        blockers.append("regime_concentration")
    blockers.append("free_data_single_index_research_only")

    # primary failure class per the intake decision rule
    if "placebo_indistinguishable" in blockers:
        failure = "placebo_indistinguishable"
    elif "subsumed_by_vol_targeting" in blockers or "no_improvement_over_baselines" in blockers:
        failure = "already_subsumed_by_vol_or_regime"
    elif "regime_concentration" in blockers:
        failure = "regime_concentration"
    elif "fails_cost_or_delay_stress" in blockers:
        failure = "event_edge_too_small_after_costs"
    elif "below_1.5_sharpe_gate" in blockers:
        failure = "no_event_edge_clearing_gate"
    else:
        failure = "none"
    hard = [b for b in blockers if b != "free_data_single_index_research_only"]
    return {
        "strategy_id": best["strategy"],
        "status": "research_pass" if not hard else "none",
        "research_candidate": not hard,
        "promotion_eligible": False,
        "failure_class": failure,
        "blockers": blockers,
    }


def _write_reports(rows, pbo, boot, dsr, best, best_sym, metric, placebo_max, classification, n_fomc) -> None:
    def tbl(subset):
        lines = ["| strategy | sharpe | calmar | maxDD | total | expo | turn | s@2x | s@delay1 | ex2020 | ex2022 | ex23-26 |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in subset:
            lines.append(
                f"| `{r['strategy']}`{'*' if r['diagnostic'] else ''} | {r['sharpe']} | {r['calmar']} | "
                f"{r['max_drawdown']} | {r['total_return']} | {r['exposure']} | {r['turnover']} | "
                f"{r['sharpe_2x_cost']} | {r['sharpe_delay1']} | {r['sharpe_ex2020']} | {r['sharpe_ex2022']} | "
                f"{r['sharpe_ex2023_26']} |")
        return lines

    ordered = sorted(rows, key=lambda x: -x["sharpe"])
    val = [
        "# Event-Conditioned Macro/Calendar v1 — Validation Report",
        "",
        f"**Git SHA:** `{_git_sha()}`  **Built:** {datetime.now(UTC).isoformat()}",
        "**Intake:** `docs/research/intake/2026-05-30-event-conditioned-macro-calendar-v1.md`",
        "**Data gate:** `event_timestamp_audit.md` (FOMC PASS; CPI/NFP deferred).",
        f"**FOMC events:** {n_fomc} scheduled (manifest); SPY/QQQ daily 2010-2026. Cost {_COST_BPS} bps/side.",
        "**Promotion intent:** research_only (single-index risk timing). `*` = diagnostic-only.",
        "",
        "## Per-strategy metrics (full sample, base cost)",
        "",
        *tbl(ordered),
        "",
        "## Pool-level multiple-testing control",
        f"- PBO probability (CSCV over {len(rows)} strategies): **{pbo.get('pbo_probability')}**",
        f"- Best promotable variant: `{best.get('strategy')}` (Sharpe {best.get('sharpe')})",
        f"- Deflated-Sharpe probability (trials={len(rows)}): **{dsr.get('probability')}**",
        f"- Bootstrap Sharpe CI lower (95%): **{boot.get('ci_lower_95')}**",
        "",
        "## Classification",
        f"- status: **{classification['status']}**  research_candidate: {classification['research_candidate']}  "
        f"promotion_eligible: {classification['promotion_eligible']}",
        f"- failure_class: **{classification['failure_class']}**",
        f"- blockers: `{', '.join(classification['blockers'])}`",
    ]
    (_OUT / "event_conditioned_validation_report.md").write_text("\n".join(val) + "\n")

    plc = [
        "# Event-Conditioned Macro/Calendar v1 — Placebo Report",
        "",
        "The decisive test: does the real FOMC calendar beat fake calendars matched by frequency?",
        "",
        f"- Best real FOMC risk-off variant Sharpe ({best_sym}): **{best.get('sharpe')}**",
        f"- Random-calendar risk-off Sharpe: **{metric('placebo_random_riskoff')}**",
        f"- Shifted-calendar (+10d) risk-off Sharpe: **{metric('placebo_shifted_riskoff')}**",
        f"- Buy-and-hold Sharpe: **{metric('bah')}**  | Vol-targeted BAH Sharpe: **{metric('voltarget_bah')}**",
        f"- Vol-regime gate (HMM-only proxy) Sharpe: **{metric('volregime_gate')}**",
        "",
        f"**Real beats max(placebo) by >0.20 Sharpe:** "
        f"{'YES' if best.get('sharpe', 0) > placebo_max + 0.20 else 'NO'} "
        f"(real {best.get('sharpe')} vs placebo_max {round(placebo_max, 4)}).",
        "",
        "If the real calendar is indistinguishable from random/shifted placebos, any apparent edge is not",
        "event-driven. This is the primary falsification for the event channel.",
    ]
    (_OUT / "event_placebo_report.md").write_text("\n".join(plc) + "\n")

    if classification["failure_class"] != "none":
        fail = [
            "# Event-Conditioned Macro/Calendar v1 — Failure Classification",
            "",
            f"**Primary failure class:** `{classification['failure_class']}`",
            f"**Best promotable variant:** `{best.get('strategy')}` (Sharpe {best.get('sharpe')})",
            "",
            "## Evidence (per the intake §8 decision rule)",
            f"- vs buy-and-hold ({best_sym}): {best.get('sharpe')} vs {metric('bah')}",
            f"- vs vol-targeted BAH: {best.get('sharpe')} vs {metric('voltarget_bah')} "
            f"({'subsumed' if best.get('sharpe', 0) <= metric('voltarget_bah') else 'beats'})",
            f"- vs placebos: real {best.get('sharpe')} vs max placebo {round(placebo_max, 4)} "
            f"({'indistinguishable' if best.get('sharpe', 0) <= placebo_max + 0.20 else 'distinguishable'})",
            f"- 1.5 Sharpe gate: {'cleared' if best.get('sharpe', 0) >= 1.5 else 'NOT cleared'}",
            f"- cost/delay stress Sharpe: 2x={best.get('sharpe_2x_cost')}, delay1={best.get('sharpe_delay1')}",
            f"- crisis removal Sharpe: ex2020={best.get('sharpe_ex2020')}, ex2022={best.get('sharpe_ex2022')}, "
            f"ex2023-26={best.get('sharpe_ex2023_26')}",
            f"- PBO={pbo.get('pbo_probability')}, DSR={dsr.get('probability')}, bootstrap_lower={boot.get('ci_lower_95')}",
            "",
            "## Decision",
            "No variant clears the gate. Event-conditioned macro/calendar v1 is closed at the stated failure",
            "class. Per the intake decision rule, a second iteration runs only if a surviving event family",
            "exists (none here). CPI/NFP remain deferred pending a timestamp-clean source.",
        ]
        (_OUT / "event_failure_classification.md").write_text("\n".join(fail) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
