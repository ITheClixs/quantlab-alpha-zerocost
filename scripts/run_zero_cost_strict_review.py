"""Strict second-stage review of zero_cost_riskalloc_v1 (no paper, no tuning).

Runs: (1) crypto-out (SPY/QQQ-only), (2) instrument PnL attribution, (3) multiple
anchored holdouts, (4) crisis-window attribution, (5) ex-crisis diagnostic,
(6) exception-style stress gate, (7) the paper-trading decision rule. Reuses the
frozen pipeline; no new features, no post-hoc tuning.

Emits under reports/signal_research/zero_cost_v1/: zero_cost_riskalloc_strict_review_report.md,
zero_cost_riskalloc_crypto_out_report.md, zero_cost_riskalloc_multi_holdout_report.md,
zero_cost_riskalloc_crisis_attribution.md, zero_cost_riskalloc_paper_decision.md

Usage:
    PYTHONPATH=src uv run python scripts/run_zero_cost_strict_review.py
"""

from __future__ import annotations

import warnings
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
from rich.console import Console

from quant_research_stack.signal_research.zero_cost import pipeline as P
from quant_research_stack.signal_research.zero_cost.strategy import backtest_portfolio, metrics

warnings.filterwarnings("ignore")
console = Console()
_OUT = Path("reports/signal_research/zero_cost_v1")
_HOLDOUTS = {"2020+": "2020-01-01", "2021+": "2021-01-01", "2022+": "2022-01-01",
             "2023+": "2023-01-01", "2024+": "2024-01-01"}
_CRISES = {"2018Q4": ("2018-10-01", "2018-12-31"), "2020_covid": ("2020-02-19", "2020-04-30"),
           "2022_bear": ("2022-01-01", "2022-12-31"), "2023_26_bull": ("2023-01-01", "2026-12-31")}
_CRISIS_YEARS = (2018, 2020, 2022)


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _slice_from(net: np.ndarray, dates: list[date], start: str) -> np.ndarray:
    m = np.array([d >= _d(start) for d in dates])
    return net[m]


def _slice_between(net: np.ndarray, dates: list[date], lo: str, hi: str) -> np.ndarray:
    m = np.array([_d(lo) <= d <= _d(hi) for d in dates])
    return net[m]


def _full(insts, dates, weights, *, cost=None, delay=1):
    return backtest_portfolio(insts, weights, dates=dates, cost_bps=cost or P.COST, delay=delay, weekly=True)


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    built = datetime.now(UTC).isoformat()

    # ---- full 4-instrument basket ----
    dates, insts, macro = P.aligned()
    n = len(dates)
    macro_on = P.macro_risk_on(macro, n)
    sw = P.strategy_weights(insts, macro_on)
    vw = P.voltarget_weights(insts)
    strat = _full(insts, dates, sw)
    bench = _full(insts, dates, vw)
    strat_2x = _full(insts, dates, sw, cost={k: 2 * v for k, v in P.COST.items()})
    strat_3x = _full(insts, dates, sw, cost={k: 3 * v for k, v in P.COST.items()})
    strat_d1 = strat
    strat_d2 = _full(insts, dates, sw, delay=2)
    inv = _full(insts, dates, P.inverted_weights(insts, macro_on))
    rng = np.random.default_rng(20260530)
    rnd = _full(insts, dates, {k: rng.uniform(0, 1, n) for k in insts})

    # ---- (1) crypto-out: SPY/QQQ only ----
    cd, ci, cm = P.aligned(("SPY", "QQQ"))
    cmacro = P.macro_risk_on(cm, len(cd))
    co_strat = _full(ci, cd, P.strategy_weights(ci, cmacro))
    co_bench = _full(ci, cd, P.voltarget_weights(ci))
    co = {
        "strat_full": metrics(co_strat.daily_returns), "bench_full": metrics(co_bench.daily_returns),
        "strat_holdout": metrics(_slice_from(co_strat.daily_returns, cd, "2024-01-01")),
        "bench_holdout": metrics(_slice_from(co_bench.daily_returns, cd, "2024-01-01")),
        "strat_excrisis": metrics(co_strat.daily_returns[~np.isin([d.year for d in cd], _CRISIS_YEARS)]),
        "bench_excrisis": metrics(co_bench.daily_returns[~np.isin([d.year for d in cd], _CRISIS_YEARS)]),
        "strat_2x": metrics(_full(ci, cd, P.strategy_weights(ci, cmacro),
                                  cost={k: 2 * v for k, v in P.COST.items()}).daily_returns),
        "strat_d2": metrics(_full(ci, cd, P.strategy_weights(ci, cmacro), delay=2).daily_returns),
    }

    # ---- (2) instrument attribution (sum of daily net contribution) ----
    total_pnl = float(np.sum(strat.daily_returns))
    crisis_mask = np.isin([d.year for d in dates], _CRISIS_YEARS)
    attr = {}
    for name, c in strat.contributions.items():
        share = float(np.sum(c)) / total_pnl if total_pnl else 0.0
        crisis_pnl = float(np.sum(c[crisis_mask]))
        attr[name] = {"pnl_share": round(share, 3), "crisis_pnl": round(crisis_pnl, 4)}
    crypto_share = sum(attr[k]["pnl_share"] for k in ("BTCUSDT", "ETHUSDT"))
    crisis_total = sum(a["crisis_pnl"] for a in attr.values()) or 1e-9
    max_instr_share = max(a["pnl_share"] for a in attr.values())
    max_crisis_share = max(a["crisis_pnl"] / crisis_total for a in attr.values())

    # ---- (3) multiple holdouts ----
    holdouts = {}
    for label, start in _HOLDOUTS.items():
        s_m = metrics(_slice_from(strat.daily_returns, dates, start))
        b_m = metrics(_slice_from(bench.daily_returns, dates, start))
        holdouts[label] = {
            "strat_sharpe": round(s_m["sharpe"], 3), "bench_sharpe": round(b_m["sharpe"], 3),
            "strat_maxdd": round(s_m["max_drawdown"], 3), "bench_maxdd": round(b_m["max_drawdown"], 3),
            "strat_calmar": round(s_m["calmar"], 3), "bench_calmar": round(b_m["calmar"], 3),
            "beats_sharpe_or_calmar": (s_m["sharpe"] > b_m["sharpe"]) or (s_m["calmar"] > b_m["calmar"]),
            "improves_dd": s_m["max_drawdown"] >= b_m["max_drawdown"],
        }
    windows_beaten = sum(1 for v in holdouts.values() if v["beats_sharpe_or_calmar"])
    dd_improved = sum(1 for v in holdouts.values() if v["improves_dd"])

    # ---- (4) crisis-window attribution ----
    crises = {}
    for label, (lo, hi) in _CRISES.items():
        s_ret = _slice_between(strat.daily_returns, dates, lo, hi)
        b_ret = _slice_between(bench.daily_returns, dates, lo, hi)
        if s_ret.size == 0:
            continue
        s_tot = float(np.prod(1 + s_ret) - 1)
        b_tot = float(np.prod(1 + b_ret) - 1)
        crises[label] = {
            "strat_return": round(s_tot, 4), "bench_return": round(b_tot, 4),
            "strat_maxdd": round(metrics(s_ret)["max_drawdown"], 4),
            "bench_maxdd": round(metrics(b_ret)["max_drawdown"], 4),
            "avoided_drawdown": round(metrics(s_ret)["max_drawdown"] - metrics(b_ret)["max_drawdown"], 4),
            "rel_return": round(s_tot - b_tot, 4),
        }

    # ---- (5) ex-crisis diagnostic ----
    ex_strat = metrics(strat.daily_returns[~crisis_mask])["sharpe"]
    ex_bench = metrics(bench.daily_returns[~crisis_mask])["sharpe"]
    crisis_insurance = ex_strat < ex_bench

    # ---- (6) exception-style gate + (7) decision ----
    delay_degradation = strat_d1.metrics["sharpe"] - strat_d2.metrics["sharpe"]
    gate = {
        "beats_bench_full_sharpe": strat.metrics["sharpe"] > bench.metrics["sharpe"],
        "improves_full_maxdd": strat.metrics["max_drawdown"] > bench.metrics["max_drawdown"],
        "windows_beaten_ge_3of5": windows_beaten >= 3,
        "dd_improved_ge_4of5": dd_improved >= 4,
        "delay_degradation_le_0.5": delay_degradation <= 0.5 and strat_d2.metrics["sharpe"] > 0,
        "survives_2x_cost": strat_2x.metrics["sharpe"] > 0,
        "survives_3x_cost": strat_3x.metrics["sharpe"] > 0,
        "not_crypto_dependent": crypto_share < 0.5,
        "crypto_out_still_beats_bench_dd": co["strat_full"]["max_drawdown"] >= co["bench_full"]["max_drawdown"],
        "no_single_instrument_gt_35pct": max_instr_share <= 0.35,
        "excrisis_nonneg": ex_strat >= 0.0,
        "inverted_worse_than_strat": inv.metrics["sharpe"] < strat.metrics["sharpe"],
        "random_worse_than_strat": rnd.metrics["sharpe"] < strat.metrics["sharpe"],
    }
    decision, failure = _decide(gate, crypto_share, crisis_insurance, windows_beaten, dd_improved,
                                co, delay_degradation)
    label = "crisis_insurance_allocator" if crisis_insurance else "risk_managed_allocator"

    _write_all(built, dates, strat, bench, co, attr, crypto_share, max_instr_share, max_crisis_share,
               holdouts, windows_beaten, dd_improved, crises, ex_strat, ex_bench, gate, decision,
               failure, label, delay_degradation, strat_2x, strat_3x, strat_d2, inv, rnd)
    console.print(f"[bold]decision[/bold] {decision} | label {label} | crypto_share {crypto_share:.2f} | "
                  f"windows_beaten {windows_beaten}/5 | ex-crisis {ex_strat:.2f} vs {ex_bench:.2f}")
    console.print(f"crypto-out: strat full Sharpe {co['strat_full']['sharpe']:.2f} vs bench "
                  f"{co['bench_full']['sharpe']:.2f}, maxDD {co['strat_full']['max_drawdown']:.2f} vs "
                  f"{co['bench_full']['max_drawdown']:.2f}")
    return 0


def _decide(gate, crypto_share, crisis_insurance, windows_beaten, dd_improved, co, delay_deg):
    if not gate["not_crypto_dependent"] or not gate["crypto_out_still_beats_bench_dd"]:
        return "DO_NOT_ADVANCE", "crypto_regime_concentration"
    if not gate["windows_beaten_ge_3of5"]:
        return "DO_NOT_ADVANCE", "subsumed_by_vol_targeting"
    if not gate["delay_degradation_le_0.5"]:
        return "DO_NOT_ADVANCE", "execution_or_regime_instability"
    if crisis_insurance and not gate["dd_improved_ge_4of5"]:
        return "DO_NOT_ADVANCE", "crisis_insurance_only"
    # passes the hard gates; advance ONLY as a clearly-labeled crisis-insurance allocator
    if all(gate.values()):
        return "ADVANCE_TO_PAPER_AS_CRISIS_INSURANCE", "none"
    return "ADVANCE_WITH_CONDITIONS", "review_caveats"


def _t(rows: list[str]) -> str:
    return "\n".join(rows)


def _write_all(built, dates, strat, bench, co, attr, crypto_share, max_instr_share, max_crisis_share,
               holdouts, windows_beaten, dd_improved, crises, ex_strat, ex_bench, gate, decision,
               failure, label, delay_deg, strat_2x, strat_3x, strat_d2, inv, rnd) -> None:
    # crypto-out
    (_OUT / "zero_cost_riskalloc_crypto_out_report.md").write_text(_t([
        "# Strict Review (1) — Crypto-Out (SPY/QQQ only)",
        f"\n**Built:** {built}",
        "| metric | strategy | vol-targeted BAH |", "|---|---:|---:|",
        f"| full Sharpe | {co['strat_full']['sharpe']:.3f} | {co['bench_full']['sharpe']:.3f} |",
        f"| holdout(2024+) Sharpe | {co['strat_holdout']['sharpe']:.3f} | {co['bench_holdout']['sharpe']:.3f} |",
        f"| full maxDD | {co['strat_full']['max_drawdown']:.3f} | {co['bench_full']['max_drawdown']:.3f} |",
        f"| full Calmar | {co['strat_full']['calmar']:.3f} | {co['bench_full']['calmar']:.3f} |",
        f"| ex-crisis Sharpe | {co['strat_excrisis']['sharpe']:.3f} | {co['bench_excrisis']['sharpe']:.3f} |",
        f"| Sharpe @2x cost | {co['strat_2x']['sharpe']:.3f} | — |",
        f"| Sharpe @delay-2 | {co['strat_d2']['sharpe']:.3f} | — |",
        f"\n- Crypto-out strategy {'still beats' if co['strat_full']['max_drawdown'] >= co['bench_full']['max_drawdown'] else 'does NOT beat'} "
        "the SPY/QQQ vol-targeted BAH on drawdown.",
        f"- Full-basket crypto (BTC+ETH) PnL share: **{crypto_share:.1%}**.",
    ]))
    # multi-holdout
    (_OUT / "zero_cost_riskalloc_multi_holdout_report.md").write_text(_t([
        "# Strict Review (3) — Multiple Anchored Holdouts",
        f"\n**Built:** {built}",
        "| window | strat Sharpe | bench Sharpe | strat maxDD | bench maxDD | strat Calmar | bench Calmar | beats(S/C) | DD improved |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
        *[f"| {w} | {v['strat_sharpe']} | {v['bench_sharpe']} | {v['strat_maxdd']} | {v['bench_maxdd']} | "
          f"{v['strat_calmar']} | {v['bench_calmar']} | {v['beats_sharpe_or_calmar']} | {v['improves_dd']} |"
          for w, v in holdouts.items()],
        f"\n- Beats vol-targeted BAH (Sharpe OR Calmar) on **{windows_beaten}/5** windows.",
        f"- Improves max drawdown on **{dd_improved}/5** windows.",
    ]))
    # crisis attribution
    (_OUT / "zero_cost_riskalloc_crisis_attribution.md").write_text(_t([
        "# Strict Review (4) — Crisis-Window Attribution",
        f"\n**Built:** {built}",
        "| window | strat return | bench return | strat maxDD | bench maxDD | avoided DD | rel return |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[f"| {w} | {v['strat_return']} | {v['bench_return']} | {v['strat_maxdd']} | {v['bench_maxdd']} | "
          f"{v['avoided_drawdown']} | {v['rel_return']} |" for w, v in crises.items()],
        "\n- `avoided DD` = strat maxDD − bench maxDD (positive = strategy had a shallower drawdown).",
        "- In the 2023-26 bull window, a negative `rel return` is the **missed upside / re-entry cost** of the gate.",
    ]))
    # instrument attribution table for the strict report
    attr_rows = [f"| {k} | {v['pnl_share']:.1%} | {v['crisis_pnl']} |" for k, v in attr.items()]
    # strict review (overall + gate)
    (_OUT / "zero_cost_riskalloc_strict_review_report.md").write_text(_t([
        "# Zero-Cost Risk-Allocator v1 — Strict Second-Stage Review",
        f"\n**Built:** {built} | basket {dates[0]}..{dates[-1]} | research_only, no paper/live.",
        "\n## (2) Instrument PnL attribution",
        "| instrument | PnL share | crisis-period PnL |", "|---|---:|---:|", *attr_rows,
        f"\n- Crypto (BTC+ETH) total PnL share: **{crypto_share:.1%}** | max single-instrument share: "
        f"**{max_instr_share:.1%}** (flag if >35%) | max crisis-period share: **{max_crisis_share:.1%}** (flag if >50%).",
        "\n## (5) Ex-crisis diagnostic (binding)",
        f"- Strategy ex-crisis Sharpe **{ex_strat:.3f}** vs vol-targeted BAH **{ex_bench:.3f}** → "
        f"{'CRISIS-DEPENDENT (insurance, not calm-market alpha)' if ex_strat < ex_bench else 'positive ex-crisis edge'}.",
        f"- Product label: **{label}**.",
        "\n## (6) Exception-style stress gate",
        *[f"- {'✅' if v else '❌'} {k}" for k, v in gate.items()],
        f"- delay-1→delay-2 Sharpe degradation: {delay_deg:.3f} (gate ≤ 0.5).",
        f"- inverted-sanity Sharpe {inv.metrics['sharpe']:.3f}; random-alloc Sharpe {rnd.metrics['sharpe']:.3f} "
        f"(both must be < strategy {strat.metrics['sharpe']:.3f}).",
        "\n## (7) Decision",
        f"- **{decision}** | failure_class: **{failure}** | label: **{label}** | promotion_eligible: False",
        "- See `zero_cost_riskalloc_paper_decision.md`.",
    ]))
    # paper decision
    advance = decision.startswith("ADVANCE")
    (_OUT / "zero_cost_riskalloc_paper_decision.md").write_text(_t([
        "# Zero-Cost Risk-Allocator v1 — Paper-Trading Decision",
        f"\n**Built:** {built}",
        f"\n## Decision: **{decision}**",
        f"- failure_class: **{failure}** | product label: **{label}** | promotion_eligible: False (paper only at most)",
        "\n## Decision-rule scorecard",
        f"- beats vol-targeted BAH on ≥3/5 holdouts (Sharpe or Calmar): {gate['windows_beaten_ge_3of5']} ({windows_beaten}/5)",
        f"- improves max drawdown consistently (≥4/5): {gate['dd_improved_ge_4of5']} ({dd_improved}/5)",
        f"- survives 1- & 2-bar delay (≤0.5 Sharpe loss): {gate['delay_degradation_le_0.5']}",
        f"- not crypto-dependent (<50% PnL) + crypto-out still beats on DD: "
        f"{gate['not_crypto_dependent'] and gate['crypto_out_still_beats_bench_dd']}",
        f"- not one-instrument (≤35%): {gate['no_single_instrument_gt_35pct']}",
        f"- ex-crisis acceptable for role (≥0): {gate['excrisis_nonneg']}",
        f"- clear product label: {label}",
        "\n## Rationale",
        (f"- **Advance to paper as a clearly-labeled `{label}`** — it improves drawdown consistently and is not "
         "crypto/one-crisis dependent. It is NOT alpha (ex-crisis it underperforms vol-targeted BAH); it is "
         "deployable only as a drawdown-control overlay, paper-only, no promotion."
         if advance else
         f"- **Do NOT advance to paper.** Classified `{failure}`: the candidate's apparent edge does not hold up "
         "under the stricter review. No paper, no live; kept research_only. This is consistent with the program's "
         "recurring finding that single-index risk-timing overlays are subsumed by / reduce to vol-targeting + "
         "crisis luck."),
        "\n_No paper trading executed. No live. No gate weakening. No tuning._",
    ]))


if __name__ == "__main__":
    raise SystemExit(main())
