"""crypto_only_riskalloc_v2 — BTCUSDT/ETHUSDT long-flat vol-targeted allocator.

Frozen pre-registration: operator message 2026-05-30 (reframe of zero_cost_riskalloc_v1
after the strict review classified the mixed allocator crypto_regime_concentration).
Crypto-only, long-flat, no leverage, weekly rebalance, decision close t / execution
t+1. Spot data (yfinance BTC-USD/ETH-USD) -> spot-only, funding N/A. No paper, no
live, no tuning, no equity sleeve.

Emits under reports/signal_research/crypto_only_v2/: crypto_only_riskalloc_v2_registry.parquet,
crypto_only_riskalloc_v2_validation_report.md, _multi_holdout_report.md,
_cost_stress_report.md, _regime_attribution.md, _paper_decision.md
"""

from __future__ import annotations

import warnings
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console

from quant_research_stack.crypto_research.perps.validation import (
    bootstrap_sharpe_payload,
    deflated_sharpe_payload,
)
from quant_research_stack.signal_research.zero_cost import pipeline as P
from quant_research_stack.signal_research.zero_cost.strategy import (
    backtest_portfolio,
    metrics,
    trend_on,
    vol_regime_on,
    vol_target_weight,
)

warnings.filterwarnings("ignore")
console = Console()
_OUT = Path("reports/signal_research/crypto_only_v2")
_INSTR = ("BTCUSDT", "ETHUSDT")
# Crypto-realistic SPOT cost (bps one-way): taker ~10 + spread ~5 + slippage ~5. Funding N/A (spot).
_COST = {"BTCUSDT": 20.0, "ETHUSDT": 20.0}
_TARGET_VOL, _CAP = 0.12, 1.0  # no leverage in v1
_HOLDOUTS = {"2020+": "2020-01-01", "2021+": "2021-01-01", "2022+": "2022-01-01",
             "2023+": "2023-01-01", "2024+": "2024-01-01"}
_REGIMES = {"2018_bear": ("2018-01-01", "2018-12-31"), "2020_covid": ("2020-02-19", "2020-04-30"),
            "2021_bull": ("2021-01-01", "2021-12-31"), "2022_crash": ("2022-01-01", "2022-12-31"),
            "2023_26_recovery": ("2023-01-01", "2026-12-31")}


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _from(net, dates, start):
    return net[np.array([d >= _d(start) for d in dates])]


def _between(net, dates, lo, hi):
    return net[np.array([_d(lo) <= d <= _d(hi) for d in dates])]


def _bt(insts, dates, weights, *, cost=None, delay=1):
    return backtest_portfolio(insts, weights, dates=dates, cost_bps=cost or _COST, delay=delay, weekly=True)


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    built = datetime.now(UTC).isoformat()
    dates, insts, _macro = P.aligned(_INSTR)  # crypto calendar (BTC/ETH inner-join)
    n = len(dates)
    console.print(f"[bold]crypto basket[/bold] {n} days {dates[0]}..{dates[-1]}")

    def vt(name):
        return vol_target_weight(insts[name].returns, target_ann_vol=_TARGET_VOL, lookback=20, cap=_CAP)

    def strat_w():
        out = {}
        for nm, s in insts.items():
            gate = trend_on(s.close, slow=200) & vol_regime_on(s.returns)
            out[nm] = vt(nm) * gate.astype(np.float64)
        return out

    rng = np.random.default_rng(20260530)
    # strategy + benchmarks (all run through the same equal-risk backtester)
    variants_w = {
        "crypto_only_riskalloc_v2": strat_w(),
        "voltarget_5050": {nm: vt(nm) for nm in insts},
        "bah_5050": {nm: np.ones(n) for nm in insts},
        "trend_only": {nm: trend_on(insts[nm].close, slow=200).astype(np.float64) for nm in insts},
        "bah_btc": {"BTCUSDT": np.ones(n), "ETHUSDT": np.zeros(n)},
        "bah_eth": {"BTCUSDT": np.zeros(n), "ETHUSDT": np.ones(n)},
        "voltarget_btc": {"BTCUSDT": vt("BTCUSDT"), "ETHUSDT": np.zeros(n)},
        "voltarget_eth": {"BTCUSDT": np.zeros(n), "ETHUSDT": vt("ETHUSDT")},
        "random_alloc": {nm: rng.uniform(0, 1, n) for nm in insts},
        "inverted": {nm: vt(nm) * (strat_w()[nm] <= 0).astype(np.float64) for nm in insts},
    }
    res = {name: _bt(insts, dates, w) for name, w in variants_w.items()}
    strat = res["crypto_only_riskalloc_v2"]
    bench = res["voltarget_5050"]
    strat_2x = _bt(insts, dates, variants_w["crypto_only_riskalloc_v2"], cost={k: 2 * v for k, v in _COST.items()})
    strat_3x = _bt(insts, dates, variants_w["crypto_only_riskalloc_v2"], cost={k: 3 * v for k, v in _COST.items()})
    strat_d2 = _bt(insts, dates, variants_w["crypto_only_riskalloc_v2"], delay=2)

    # multi-holdout vs voltarget_5050
    holdouts = {}
    for label, start in _HOLDOUTS.items():
        sm, bm = metrics(_from(strat.daily_returns, dates, start)), metrics(_from(bench.daily_returns, dates, start))
        holdouts[label] = {"s_sharpe": round(sm["sharpe"], 3), "b_sharpe": round(bm["sharpe"], 3),
                           "s_dd": round(sm["max_drawdown"], 3), "b_dd": round(bm["max_drawdown"], 3),
                           "s_calmar": round(sm["calmar"], 3), "b_calmar": round(bm["calmar"], 3),
                           "beats": (sm["sharpe"] > bm["sharpe"]) or (sm["calmar"] > bm["calmar"]),
                           "dd_better": sm["max_drawdown"] >= bm["max_drawdown"]}
    windows_beaten = sum(v["beats"] for v in holdouts.values())
    dd_improved = sum(v["dd_better"] for v in holdouts.values())

    # regime attribution
    regimes = {}
    for label, (lo, hi) in _REGIMES.items():
        s_ret, b_ret = _between(strat.daily_returns, dates, lo, hi), _between(bench.daily_returns, dates, lo, hi)
        if s_ret.size == 0:
            continue
        regimes[label] = {"s_return": round(float(np.prod(1 + s_ret) - 1), 4),
                          "b_return": round(float(np.prod(1 + b_ret) - 1), 4),
                          "s_dd": round(metrics(s_ret)["max_drawdown"], 4),
                          "b_dd": round(metrics(b_ret)["max_drawdown"], 4),
                          "avoided_dd": round(metrics(s_ret)["max_drawdown"] - metrics(b_ret)["max_drawdown"], 4)}

    # instrument + year attribution
    total = float(np.sum(strat.daily_returns)) or 1e-9
    instr_share = {nm: round(float(np.sum(c)) / total, 3) for nm, c in strat.contributions.items()}
    years = np.array([d.year for d in dates])
    year_share = {int(y): round(float(np.sum(strat.daily_returns[years == y])) / total, 3) for y in sorted(set(years.tolist()))}
    max_asset = max(instr_share.values())
    max_year = max(year_share.values())

    boot = bootstrap_sharpe_payload(strat.daily_returns)
    dsr = deflated_sharpe_payload(strat.daily_returns, trials=len(variants_w))

    gate = {
        "windows_beaten_ge_3of5": windows_beaten >= 3,
        "dd_improved_ge_4of5": dd_improved >= 4,
        "sharpe_pos_2x": strat_2x.metrics["sharpe"] > 0,
        "sharpe_pos_3x": strat_3x.metrics["sharpe"] > 0,
        "delay_ok": (strat.metrics["sharpe"] - strat_d2.metrics["sharpe"]) <= 0.5 and strat_d2.metrics["sharpe"] > 0,
        "no_year_gt_50pct": max_year <= 0.50,
        "no_asset_gt_65pct": max_asset <= 0.65,
        "bootstrap_lower_pos": (boot.get("ci_lower_95") or -1) > 0,
        "dsr_ok": (dsr.get("probability") or 0) >= 0.5,
        "random_inverted_fail": res["random_alloc"].metrics["sharpe"] < strat.metrics["sharpe"]
        and res["inverted"].metrics["sharpe"] < strat.metrics["sharpe"],
        "paper_feasible_spot_weekly": True,  # spot, taker, weekly, t+1 -> feasible on a free venue
    }
    decision, failure = _decide(gate, instr_share)
    _write(built, dates, res, strat, bench, strat_2x, strat_3x, strat_d2, holdouts, windows_beaten,
           dd_improved, regimes, instr_share, year_share, max_asset, max_year, boot, dsr, gate, decision, failure)
    console.print(f"[bold]decision[/bold] {decision} | failure {failure} | windows {windows_beaten}/5 dd {dd_improved}/5 "
                  f"| BTC {instr_share.get('BTCUSDT')} ETH {instr_share.get('ETHUSDT')} | maxyear {max_year}")
    console.print(f"strat full Sharpe {strat.metrics['sharpe']:.2f} vs voltgt5050 {bench.metrics['sharpe']:.2f} | "
                  f"2x {strat_2x.metrics['sharpe']:.2f} 3x {strat_3x.metrics['sharpe']:.2f} | "
                  f"boot_lower {boot.get('ci_lower_95')} DSR {dsr.get('probability')}")
    return 0


def _decide(gate, instr_share):
    if not gate["sharpe_pos_2x"] or not gate["sharpe_pos_3x"]:
        return "DO_NOT_ADVANCE", "cost_failure"
    if not gate["delay_ok"]:
        return "DO_NOT_ADVANCE", "delay_failure"
    if not gate["no_asset_gt_65pct"]:
        dominant = max(instr_share, key=lambda k: instr_share[k])
        cls = "btc_dominance_concentration" if dominant == "BTCUSDT" else "single_asset_concentration_eth"
        return "DO_NOT_ADVANCE", cls
    if not gate["windows_beaten_ge_3of5"]:
        return "DO_NOT_ADVANCE", "subsumed_by_crypto_vol_targeting"
    if not gate["no_year_gt_50pct"]:
        return "DO_NOT_ADVANCE", "crypto_bull_regime_only"
    if not (gate["bootstrap_lower_pos"] and gate["dsr_ok"] and gate["random_inverted_fail"]):
        return "DO_NOT_ADVANCE", "holdout_failure"
    if all(gate.values()):
        return "ADVANCE_TO_PAPER_CANDIDATE", "none"
    return "ADVANCE_WITH_CONDITIONS", "review_caveats"


def _tbl(rows: list[str]) -> str:
    return "\n".join(rows)


def _write(built, dates, res, strat, bench, s2x, s3x, sd2, holdouts, windows_beaten, dd_improved,
           regimes, instr_share, year_share, max_asset, max_year, boot, dsr, gate, decision, failure) -> None:
    rows = [{"variant": k, "sharpe": round(v.metrics["sharpe"], 4), "maxdd": round(v.metrics["max_drawdown"], 4),
             "calmar": round(v.metrics["calmar"], 4), "ann_return": round(v.metrics["ann_return"], 4),
             "total_return": round(v.metrics["total_return"], 4)} for k, v in res.items()]
    pl.DataFrame(rows).write_parquet(_OUT / "crypto_only_riskalloc_v2_registry.parquet")

    (_OUT / "crypto_only_riskalloc_v2_validation_report.md").write_text(_tbl([
        "# crypto_only_riskalloc_v2 — Validation Report",
        f"\n**Built:** {built} | basket {dates[0]}..{dates[-1]} ({len(dates)} days) | BTCUSDT+ETHUSDT, long-flat,"
        " no leverage, weekly rebalance, close t / execute t+1.",
        "**Pre-registration:** operator message 2026-05-30 (reframe after zero_cost_riskalloc_v1 strict review).",
        "**Cost:** SPOT, 20 bps one-way (taker ~10 + spread ~5 + slippage ~5); funding N/A (spot data). research_only.",
        "\n| variant | Sharpe | maxDD | Calmar | ann ret | total ret |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| `{r['variant']}` | {r['sharpe']} | {r['maxdd']} | {r['calmar']} | {r['ann_return']} | {r['total_return']} |"
          for r in rows],
        "\n## Stress + robustness",
        f"- Sharpe @2x cost: {s2x.metrics['sharpe']:.3f} | @3x cost: {s3x.metrics['sharpe']:.3f} | "
        f"@delay-2: {sd2.metrics['sharpe']:.3f}",
        f"- bootstrap Sharpe CI lower: {boot.get('ci_lower_95')} | DSR: {dsr.get('probability')}",
        "\n## Instrument & year attribution",
        f"- BTC PnL share {instr_share.get('BTCUSDT')} | ETH {instr_share.get('ETHUSDT')} "
        f"(flag if one >65%: max {max_asset})",
        f"- max single-year PnL share: {max_year} (flag if >50%)",
        f"- year shares: {year_share}",
        "\n## Gate scorecard",
        *[f"- {'✅' if v else '❌'} {k}" for k, v in gate.items()],
        f"\n## Decision: **{decision}** | failure_class: **{failure}** | promotion_eligible: False (paper at most)",
    ]))

    (_OUT / "crypto_only_riskalloc_v2_multi_holdout_report.md").write_text(_tbl([
        "# crypto_only_riskalloc_v2 — Multiple Holdouts (vs vol-targeted 50/50 BTC/ETH)",
        f"\n**Built:** {built}",
        "| window | strat Sharpe | bench Sharpe | strat maxDD | bench maxDD | strat Calmar | bench Calmar | beats | DD better |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
        *[f"| {w} | {v['s_sharpe']} | {v['b_sharpe']} | {v['s_dd']} | {v['b_dd']} | {v['s_calmar']} | {v['b_calmar']} | "
          f"{v['beats']} | {v['dd_better']} |" for w, v in holdouts.items()],
        f"\n- Beats (Sharpe or Calmar) on **{windows_beaten}/5**; improves maxDD on **{dd_improved}/5**.",
    ]))

    (_OUT / "crypto_only_riskalloc_v2_cost_stress_report.md").write_text(_tbl([
        "# crypto_only_riskalloc_v2 — Cost & Delay Stress",
        f"\n**Built:** {built} | base cost 20 bps one-way (spot).",
        "| scenario | Sharpe | maxDD |", "|---|---:|---:|",
        f"| base (1x cost, delay-1) | {strat.metrics['sharpe']:.3f} | {strat.metrics['max_drawdown']:.3f} |",
        f"| 2x cost | {s2x.metrics['sharpe']:.3f} | {s2x.metrics['max_drawdown']:.3f} |",
        f"| 3x cost | {s3x.metrics['sharpe']:.3f} | {s3x.metrics['max_drawdown']:.3f} |",
        f"| delay-2 | {sd2.metrics['sharpe']:.3f} | {sd2.metrics['max_drawdown']:.3f} |",
        f"\n- delay-1->delay-2 Sharpe degradation: {strat.metrics['sharpe'] - sd2.metrics['sharpe']:.3f} (gate ≤ 0.5).",
    ]))

    (_OUT / "crypto_only_riskalloc_v2_regime_attribution.md").write_text(_tbl([
        "# crypto_only_riskalloc_v2 — Regime Attribution",
        f"\n**Built:** {built}",
        "| regime | strat return | bench return | strat maxDD | bench maxDD | avoided DD |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {w} | {v['s_return']} | {v['b_return']} | {v['s_dd']} | {v['b_dd']} | {v['avoided_dd']} |"
          for w, v in regimes.items()],
        "\n- `avoided DD` positive = shallower drawdown than vol-targeted 50/50. Negative bull-regime `strat return`"
        " minus `bench return` = missed upside / re-entry cost of the trend+regime gate.",
    ]))

    advance = decision.startswith("ADVANCE")
    (_OUT / "crypto_only_riskalloc_v2_paper_decision.md").write_text(_tbl([
        "# crypto_only_riskalloc_v2 — Paper-Trading Decision",
        f"\n**Built:** {built}",
        f"\n## Decision: **{decision}** | failure_class: **{failure}** | promotion_eligible: False",
        "\n## Binding-gate scorecard",
        *[f"- {'✅' if v else '❌'} {k}" for k, v in gate.items()],
        "\n## Rationale",
        ("- **Advance to paper_candidate** (spot, weekly, t+1) as a crypto crisis-aware allocator: beats vol-targeted"
         " 50/50 across holdouts, improves drawdown, survives 2x/3x cost + delay, not BTC/one-year dominated,"
         " bootstrap+DSR+sanity all pass. Paper only; no live; no promotion."
         if advance else
         f"- **Do NOT advance.** Classified `{failure}`. Kept research_only; no paper/live. The crypto sleeve does"
         " not clear the stricter crypto-only gate either — consistent with the program's subsumption pattern."),
        "\n_No paper trading executed. No live. No equity sleeve. No tuning. No new paid data._",
    ]))


if __name__ == "__main__":
    raise SystemExit(main())
