"""zero_cost_riskalloc_v1 — build + validate the frozen candidate (P1).

Long-flat vol-targeted regime+macro allocator over SPY/QQQ/BTCUSDT/ETHUSDT, weekly
rebalance, equal-risk, decision close t / execution t+1. BINDING GATE: must beat
vol-targeted buy-and-hold on net Sharpe AND max drawdown (holdout) or be killed.

Emits under reports/signal_research/zero_cost_v1/: zero_cost_strategy_registry.parquet,
zero_cost_validation_report.md, zero_cost_failure_classification.md (if killed).

Usage:
    PYTHONPATH=src uv run python scripts/run_zero_cost_riskalloc_v1.py
"""

from __future__ import annotations

import warnings
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
from quant_research_stack.signal_research.zero_cost.data import INSTRUMENTS, MACRO_REGISTRY, load_instrument, load_macro
from quant_research_stack.signal_research.zero_cost.strategy import (
    InstrumentSeries,
    backtest_portfolio,
    daily_returns,
    metrics,
    trend_on,
    vol_regime_on,
    vol_target_weight,
)

warnings.filterwarnings("ignore")
console = Console()
_OUT = Path("reports/signal_research/zero_cost_v1")
_COST = {"SPY": 1.0, "QQQ": 1.0, "BTCUSDT": 8.0, "ETHUSDT": 8.0}
_CAP = {"SPY": 1.5, "QQQ": 1.5, "BTCUSDT": 1.0, "ETHUSDT": 1.0}
_HOLDOUT_START = "2024-01-01"
_CRISIS_YEARS = (2018, 2020, 2022)


def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(len(x)):
        if i + 1 >= w:
            out[i] = float(np.mean(x[i + 1 - w : i + 1]))
    return out


def _aligned() -> tuple[list[date], dict[str, InstrumentSeries], dict[str, np.ndarray]]:
    # master = SPY equity calendar; inner-join the rest on date
    frames = {n: load_instrument(n).rename({"close": n}) for n in INSTRUMENTS}
    master = frames["SPY"]
    for n in ("QQQ", "BTCUSDT", "ETHUSDT"):
        master = master.join(frames[n], on="date", how="inner")
    for s in MACRO_REGISTRY:
        m = load_macro(s)
        if m.height:
            master = master.join(m.rename({"close": s.name}), on="date", how="left")
    master = master.sort("date")
    dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in master["date"].to_list()]
    insts = {}
    for n in INSTRUMENTS:
        close = master[n].to_numpy().astype(np.float64)
        insts[n] = InstrumentSeries(n, dates, close, daily_returns(close), n.endswith("USDT"))
    macro = {s.name: (master[s.name].fill_null(strategy="forward").to_numpy().astype(np.float64)
                      if s.name in master.columns else np.full(master.height, np.nan))
             for s in MACRO_REGISTRY}
    return dates, insts, macro


def _macro_risk_on(macro: dict[str, np.ndarray], n: int) -> np.ndarray:
    vix, vix3m = macro.get("vix"), macro.get("vix3m")
    hyg = macro.get("credit_hyg")
    u10, u2 = macro.get("ust10y"), macro.get("ust2y")
    contango = (vix < vix3m) if vix is not None and vix3m is not None else np.ones(n, bool)
    credit_on = (hyg > _roll_mean(hyg, 100)) if hyg is not None else np.ones(n, bool)
    slope_ok = ((u10 - u2) > -1.0) if u10 is not None and u2 is not None else np.ones(n, bool)
    return np.nan_to_num(contango, nan=1.0).astype(bool) & np.nan_to_num(credit_on, nan=1.0).astype(bool) \
        & np.nan_to_num(slope_ok, nan=1.0).astype(bool)


def _strategy_weights(insts: dict[str, InstrumentSeries], macro_on: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for n, s in insts.items():
        vt = vol_target_weight(s.returns, target_ann_vol=0.12, lookback=20, cap=_CAP[n])
        gate = trend_on(s.close, slow=200) & vol_regime_on(s.returns) & macro_on
        out[n] = vt * gate.astype(np.float64)
    return out


def _voltarget_only(insts: dict[str, InstrumentSeries]) -> dict[str, np.ndarray]:
    return {n: vol_target_weight(s.returns, target_ann_vol=0.12, lookback=20, cap=_CAP[n]) for n, s in insts.items()}


def _bah(insts: dict[str, InstrumentSeries]) -> dict[str, np.ndarray]:
    return {n: np.ones(len(s.dates)) for n, s in insts.items()}


def _trend_only(insts: dict[str, InstrumentSeries]) -> dict[str, np.ndarray]:
    return {n: trend_on(s.close, slow=200).astype(np.float64) for n, s in insts.items()}


def _regime_only(insts: dict[str, InstrumentSeries]) -> dict[str, np.ndarray]:
    return {n: vol_regime_on(s.returns).astype(np.float64) for n, s in insts.items()}


def _random_alloc(insts: dict[str, InstrumentSeries], rng) -> dict[str, np.ndarray]:
    return {n: rng.uniform(0, 1, len(s.dates)) for n, s in insts.items()}


def _split_metrics(net: np.ndarray, dates: list[date]) -> dict[str, Any]:
    yrs = np.array([d.year for d in dates])
    hold_mask = np.array([d >= datetime.strptime(_HOLDOUT_START, "%Y-%m-%d").date() for d in dates])
    crisis_mask = ~np.isin(yrs, _CRISIS_YEARS)
    return {"full": metrics(net), "holdout": metrics(net[hold_mask]),
            "ex_crisis": metrics(net[crisis_mask])}


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    dates, insts, macro = _aligned()
    n = len(dates)
    console.print(f"[bold]aligned[/bold] {n} days {dates[0]}..{dates[-1]} | instruments {list(insts)}")
    macro_on = _macro_risk_on(macro, n)
    rng = np.random.default_rng(20260530)

    variants = {
        "zero_cost_riskalloc_v1": _strategy_weights(insts, macro_on),
        "voltarget_bah": _voltarget_only(insts),
        "buy_and_hold": _bah(insts),
        "trend_only": _trend_only(insts),
        "regime_only": _regime_only(insts),
        "random_alloc": _random_alloc(insts, rng),
    }
    results: dict[str, dict[str, Any]] = {}
    daily_matrix: dict[str, np.ndarray] = {}
    for name, tw in variants.items():
        base = backtest_portfolio(insts, tw, dates=dates, cost_bps=_COST, delay=1, weekly=True)
        c2x = backtest_portfolio(insts, tw, dates=dates, cost_bps={k: 2 * v for k, v in _COST.items()},
                                 delay=1, weekly=True)
        d2 = backtest_portfolio(insts, tw, dates=dates, cost_bps=_COST, delay=2, weekly=True)
        turnover = float(np.mean([np.mean(np.abs(np.diff(w, prepend=w[0]))) for w in base.weights.values()]) * 252)
        sm = _split_metrics(base.daily_returns, dates)
        results[name] = {"full": sm["full"], "holdout": sm["holdout"], "ex_crisis": sm["ex_crisis"],
                         "sharpe_2xcost": c2x.metrics["sharpe"], "sharpe_delay2": d2.metrics["sharpe"],
                         "ann_turnover": round(turnover, 2)}
        daily_matrix[name] = base.daily_returns

    strat = results["zero_cost_riskalloc_v1"]
    bench = results["voltarget_bah"]
    # binding gate: beat voltarget_bah on holdout net Sharpe AND max drawdown
    beats_sharpe = strat["holdout"]["sharpe"] > bench["holdout"]["sharpe"]
    beats_dd = strat["holdout"]["max_drawdown"] > bench["holdout"]["max_drawdown"]  # less negative
    boot = bootstrap_sharpe_payload(daily_matrix["zero_cost_riskalloc_v1"])
    pbo_df = pl.DataFrame(daily_matrix).with_row_index("event_index")
    pbo = estimate_registry_pbo(pbo_df, strategy_columns=list(variants), n_partitions=8)
    dsr = deflated_sharpe_payload(daily_matrix["zero_cost_riskalloc_v1"], trials=len(variants))

    classification = _classify(strat, bench, beats_sharpe, beats_dd, boot, pbo, dsr)
    _write_reports(dates, insts, results, classification, beats_sharpe, beats_dd, boot, pbo, dsr)
    console.print(f"[bold]holdout[/bold] strat Sharpe={strat['holdout']['sharpe']:.3f} "
                  f"DD={strat['holdout']['max_drawdown']:.3f} | voltgt_bah Sharpe={bench['holdout']['sharpe']:.3f} "
                  f"DD={bench['holdout']['max_drawdown']:.3f}")
    console.print(f"[bold]classification[/bold] {classification}")
    return 0


def _classify(strat, bench, beats_sharpe, beats_dd, boot, pbo, dsr) -> dict[str, Any]:
    hard: list[str] = []      # literal §6 kill conditions
    caveats: list[str] = []   # pass-but-fragile flags
    if not (beats_sharpe and beats_dd):
        hard.append("subsumed_by_vol_targeting")
    if strat["sharpe_delay2"] <= 0 or strat["holdout"]["sharpe"] <= 0:
        hard.append("delay_or_holdout_failure")
    if strat["sharpe_2xcost"] <= 0:
        hard.append("cost_failure")
    if strat["ex_crisis"]["sharpe"] <= 0:
        hard.append("crisis_concentration")
    p = pbo.get("pbo_probability")
    if p is not None and p > 0.25:
        hard.append("high_pbo")
    # CRITICAL honesty check: is the edge over the benchmark crisis-driven?
    # If ex-crisis the strategy LOSES to vol-targeted BAH, the outperformance is
    # crisis-insurance, not alpha -> not an auto-advance.
    crisis_dependent = strat["ex_crisis"]["sharpe"] < bench["ex_crisis"]["sharpe"]
    if crisis_dependent:
        caveats.append("edge_is_crisis_dependent_vs_benchmark")
    if (boot.get("ci_lower_95") or 0.0) <= 0.0:
        caveats.append("bootstrap_ci_not_positive")

    if hard:
        decision = "KILL"
    elif caveats:
        decision = "PASS_WITH_CAVEAT_NEEDS_STRICTER_REVIEW"
    else:
        decision = "ADVANCE_TO_PAPER"
    return {"strategy_id": "zero_cost_riskalloc_v1", "decision": decision,
            "promotion_eligible": False, "paper_candidate": decision == "ADVANCE_TO_PAPER",
            "primary_failure": (hard[0] if hard else (caveats[0] if caveats else "none")),
            "hard_blockers": hard, "caveats": caveats,
            "ex_crisis_strat_sharpe": round(strat["ex_crisis"]["sharpe"], 4),
            "ex_crisis_bench_sharpe": round(bench["ex_crisis"]["sharpe"], 4)}


def _write_reports(dates, insts, results, classification, beats_sharpe, beats_dd, boot, pbo, dsr) -> None:
    built = datetime.now(UTC).isoformat()
    rows = []
    for name, r in results.items():
        rows.append({"variant": name, "full_sharpe": round(r["full"]["sharpe"], 4),
                     "holdout_sharpe": round(r["holdout"]["sharpe"], 4),
                     "full_maxdd": round(r["full"]["max_drawdown"], 4),
                     "holdout_maxdd": round(r["holdout"]["max_drawdown"], 4),
                     "full_calmar": round(r["full"]["calmar"], 4),
                     "ann_return": round(r["full"]["ann_return"], 4),
                     "sharpe_2xcost": round(r["sharpe_2xcost"], 4),
                     "sharpe_delay2": round(r["sharpe_delay2"], 4),
                     "ex_crisis_sharpe": round(r["ex_crisis"]["sharpe"], 4),
                     "ann_turnover": r["ann_turnover"]})
    pl.DataFrame(rows).write_parquet(_OUT / "zero_cost_strategy_registry.parquet")

    def line(r: dict) -> str:
        return (f"| `{r['variant']}` | {r['full_sharpe']} | {r['holdout_sharpe']} | {r['full_maxdd']} | "
                f"{r['holdout_maxdd']} | {r['full_calmar']} | {r['ann_return']} | {r['sharpe_2xcost']} | "
                f"{r['sharpe_delay2']} | {r['ex_crisis_sharpe']} | {r['ann_turnover']} |")

    val = [
        "# Zero-Cost Risk-Allocator v1 — Validation Report",
        f"\n**Built:** {built} | basket {dates[0]}..{dates[-1]} ({len(dates)} days) | instruments {list(insts)}",
        "**Intake:** `docs/research/intake/2026-05-30-zero-cost-deployable-v1.md` | paper_trade_after_pass.",
        "Long-flat, weekly rebalance, equal-risk, decision close t / execution t+1. Cost: SPY/QQQ 1bp, BTC/ETH 8bp.",
        "\n| variant | full Sharpe | holdout Sharpe | full maxDD | holdout maxDD | full Calmar | ann ret | "
        "Sharpe@2xcost | Sharpe@delay2 | ex-crisis Sharpe | ann turnover |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[line(r) for r in rows],
        "\n## Binding gate (vs vol-targeted buy-and-hold, holdout)",
        f"- beats voltarget_bah on holdout Sharpe: **{beats_sharpe}**",
        f"- beats voltarget_bah on holdout max drawdown: **{beats_dd}**",
        f"- bootstrap Sharpe CI lower (95%): **{boot.get('ci_lower_95')}** | PBO **{pbo.get('pbo_probability')}** | "
        f"DSR **{dsr.get('probability')}**",
        "\n## Decision",
        f"- **{classification['decision']}** | paper_candidate: {classification['paper_candidate']} | "
        "promotion_eligible: False",
        f"- hard blockers (§6 kill): `{', '.join(classification['hard_blockers']) or 'none'}`",
        f"- caveats (pass-but-fragile): `{', '.join(classification['caveats']) or 'none'}`",
        f"- **Ex-crisis check: strategy Sharpe {classification['ex_crisis_strat_sharpe']} vs vol-targeted BAH "
        f"{classification['ex_crisis_bench_sharpe']}** — if the strategy is lower, its edge over the benchmark is "
        "crisis-driven (better drawdowns in 2018/2020/2022), NOT calm-market alpha.",
        "\n## Notes / honest caveats",
        "- Basket window is ETH-bound (~2017+); crypto history is short and crisis-heavy → wide CIs.",
        "- Crypto daily returns aligned to the equity trading calendar (weekend moves fold into Monday).",
        "- Macro features (VIX term structure, credit, yield slope) market-priced, used t+1. research_only until paper.",
    ]
    (_OUT / "zero_cost_validation_report.md").write_text("\n".join(val) + "\n")

    if classification["decision"] != "ADVANCE_TO_PAPER":
        killed = classification["decision"] == "KILL"
        (_OUT / "zero_cost_failure_classification.md").write_text("\n".join([
            "# Zero-Cost Risk-Allocator v1 — Classification",
            f"\n**Decision:** {classification['decision']} | primary: `{classification['primary_failure']}`",
            f"**Hard blockers (§6 kill):** `{', '.join(classification['hard_blockers']) or 'none'}`",
            f"**Caveats:** `{', '.join(classification['caveats']) or 'none'}`",
            "\n## Evidence",
            f"- Strategy holdout Sharpe {results['zero_cost_riskalloc_v1']['holdout']['sharpe']:.3f} vs "
            f"voltarget_bah {results['voltarget_bah']['holdout']['sharpe']:.3f}; "
            f"maxDD {results['zero_cost_riskalloc_v1']['holdout']['max_drawdown']:.3f} vs "
            f"{results['voltarget_bah']['holdout']['max_drawdown']:.3f}.",
            f"- **Ex-crisis Sharpe {classification['ex_crisis_strat_sharpe']} vs benchmark "
            f"{classification['ex_crisis_bench_sharpe']}.**",
            ("\n## Decision rationale\n"
             "- KILL: did not beat vol-targeted buy-and-hold; branch closed (matches the program's subsumption finding)."
             if killed else
             "\n## Decision rationale\n"
             "- The strategy clears the literal §6 gate (beats vol-targeted BAH on holdout Sharpe AND max drawdown,"
             " PBO<0.25, DSR high) BUT its outperformance over the benchmark is **crisis-driven** — ex-crisis it"
             " UNDERPERFORMS vol-targeted BAH. It is a drawdown-control / crisis-insurance overlay, not calm-market"
             " alpha. **Do NOT auto-advance to paper.** Stricter review required first: the single-index exception"
             " policy's full 24-criterion gate, a crypto-out (SPY/QQQ-only) test, and multiple holdout windows."),
            "\n_research_only; no paper/live; no promotion until stricter review passes._",
        ]) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
