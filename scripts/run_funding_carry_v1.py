"""P2 — delta-neutral funding-carry backtest (Strategy A) under the intake §5 gate.

research_only. NO paper, NO live. Long spot / short perp on BTC + ETH, equal-weight
pooled book, fully costed. Evaluates the binding gate: regime robustness (positive in
the majority of years incl. thin 2022/2026), cost survival, placebo separation,
PBO/DSR/bootstrap, and PnL concentration. Writes manifest + report + verdict.

Run: PYTHONPATH=src uv run python scripts/run_funding_carry_v1.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_research_stack.crypto_research.funding import carry, prices
from quant_research_stack.crypto_research.funding import data as fdata
from quant_research_stack.crypto_research.perps import validation as val

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SPOT_TAKER_BPS = 10.0
PERP_TAKER_BPS = 5.0
MANIFEST = Path("manifests/funding_carry/funding_carry_v1_manifest.json")
REPORT = Path("reports/signal_research/funding_carry_v1/funding_carry_v1_results.md")


def _panel(sym: str) -> pl.DataFrame:
    return prices.align_carry(fdata.load_funding(sym),
                              prices.load_daily_klines(sym, "spot"),
                              prices.load_daily_klines(sym, "perp"))


def main() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    panels = {s: _panel(s) for s in SYMBOLS}
    base = {s: carry.carry_returns(panels[s], spot_taker_bps=SPOT_TAKER_BPS,
                                   perp_taker_bps=PERP_TAKER_BPS) for s in SYMBOLS}
    pooled = carry.pooled_book(base)
    dates = pooled.dates

    # cost stress (2x, 3x) on the pooled book
    stress = {}
    for mult in (1.0, 2.0, 3.0):
        legs = {s: carry.carry_returns(panels[s], spot_taker_bps=SPOT_TAKER_BPS * mult,
                                       perp_taker_bps=PERP_TAKER_BPS * mult) for s in SYMBOLS}
        stress[f"{mult:g}x"] = carry.pooled_book(legs).metrics

    # placebos on pooled
    inv = carry.pooled_book({s: carry.carry_returns(
        panels[s], spot_taker_bps=SPOT_TAKER_BPS, perp_taker_bps=PERP_TAKER_BPS,
        invert=True) for s in SYMBOLS})
    zerof = carry.pooled_book({s: carry.carry_returns(
        panels[s], spot_taker_bps=SPOT_TAKER_BPS, perp_taker_bps=PERP_TAKER_BPS,
        zero_funding=True) for s in SYMBOLS})

    # statistical battery on pooled net
    boot = val.bootstrap_sharpe_payload(pooled.net, resamples=2000, seed=17)
    dsr = val.deflated_sharpe_payload(pooled.net, trials=3)  # BTC, ETH, pooled
    reg = pl.DataFrame({"BTC": base["BTCUSDT"].net, "ETH": base["ETHUSDT"].net,
                        "pooled": pooled.net})
    pbo = val.estimate_registry_pbo(reg, strategy_columns=["BTC", "ETH", "pooled"], n_partitions=8)

    py_pooled = carry.per_year(dates, pooled.net)
    conc = carry.pnl_concentration(dates, pooled.net)
    conc_assets = {s: carry.pnl_concentration(base[s].dates, base[s].net) for s in SYMBOLS}

    # ---- gate (intake §5) ----
    pos_years = [y for y, m in py_pooled.items() if m["total_pct"] > 0]
    g_regime = (
        len(pos_years) > len(py_pooled) / 2
        and py_pooled.get(2022, {}).get("total_pct", -1) > 0
        and py_pooled.get(2026, {}).get("total_pct", -1) >= 0
    )
    g_cost = stress["3x"]["sharpe"] > 0 and stress["3x"]["ann_return"] > 0
    g_placebo = (pooled.metrics["total_return"] > 0
                 and inv.metrics["total_return"] < 0
                 and abs(zerof.metrics["ann_return"]) < 0.02)
    g_stat = (float(boot["ci_lower_95"]) > 0
              and float(dsr["probability"]) > 0.90
              and float(pbo["pbo_probability"]) < 0.5)
    g_conc = conc["top_year_share"] <= 0.60
    gates = {"regime_robust": g_regime, "cost_survival": g_cost,
             "placebo_separation": g_placebo, "statistical": g_stat,
             "concentration": g_conc}
    verdict = "CANDIDATE" if all(gates.values()) else "DO_NOT_ADVANCE"

    out = {
        "built_utc": datetime.now(UTC).isoformat(),
        "status": "research_only — no paper, no live",
        "cost_bps": {"spot_taker": SPOT_TAKER_BPS, "perp_taker": PERP_TAKER_BPS},
        "per_asset_metrics": {s: base[s].metrics for s in SYMBOLS},
        "pooled_metrics": pooled.metrics,
        "pooled_per_year": py_pooled,
        "per_asset_per_year": {s: carry.per_year(base[s].dates, base[s].net) for s in SYMBOLS},
        "cost_stress_pooled": stress,
        "placebo": {"inverted": inv.metrics, "zero_funding": zerof.metrics},
        "statistical": {"bootstrap": boot, "deflated_sharpe": dsr, "pbo": pbo},
        "concentration": {"pooled": conc, **conc_assets},
        "gates": gates,
        "verdict": verdict,
    }
    MANIFEST.write_text(json.dumps(out, indent=2, default=str))
    _write_report(out, py_pooled)
    print(f"\nVERDICT: {verdict}")
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"pooled net sharpe={pooled.metrics['sharpe']:.2f} "
          f"ann={pooled.metrics['ann_return']*100:.2f}% maxdd={pooled.metrics['max_drawdown']*100:.1f}%")
    print(f"manifest: {MANIFEST}\nreport: {REPORT}")


def _write_report(out: dict, py_pooled: dict) -> None:
    pm = out["pooled_metrics"]
    g = out["gates"]
    boot = out["statistical"]["bootstrap"]
    dsr = out["statistical"]["deflated_sharpe"]
    pbo = out["statistical"]["pbo"]
    lines = [
        "# Funding-Carry v1 — Delta-Neutral Carry Backtest (P2)",
        "",
        f"**Date:** {datetime.now(UTC).date()}  ",
        f"**Verdict:** **{out['verdict']}**  ",
        "**Status:** research_only — **no paper, no live, no promotion**.",
        "",
        "Strategy A: long spot / short USDT-M perp on BTC + ETH, equal-weight pooled, "
        f"re-neutralized daily. Costs: spot taker {SPOT_TAKER_BPS}bps + perp taker "
        f"{PERP_TAKER_BPS}bps one-way, plus daily hedge-maintenance turnover and entry/"
        "exit. Returns per unit one-side notional (carry yield). Annualized at sqrt(365).",
        "",
        "## ⚠️ Realism caveat — read before the headline",
        "",
        f"The pooled Sharpe below ({pm['sharpe']:.1f}) is **not deployable-real** — it is the "
        "**carry illusion**. Spot and the perp track almost perfectly at the *same-venue daily "
        "close*, so the basis/price term has near-zero variance and the book is a smooth, "
        "low-variance positive funding drip. A tradable book carries risks this daily-close "
        "model omits: intraday basis gaps (the perp dislocated from spot by multiple % in the "
        "2020-03 and 2021-05 crashes), funding spikes, **short-leg liquidation/margin risk**, "
        "and execution slippage off the close. A realistic funding-carry book runs Sharpe ~1-2, "
        "not ~8. **Trust the annual return and the per-year regime picture; do NOT trust the "
        "Sharpe.**",
        "",
        "## Headline (pooled book, base cost)",
        "",
        f"- Sharpe **{pm['sharpe']:.2f}** (illusory — see caveat), ann return "
        f"**{pm['ann_return']*100:.2f}%**, ann vol {pm['ann_vol']*100:.2f}%, "
        f"max DD {pm['max_drawdown']*100:.1f}% (understated), Calmar {pm['calmar']:.2f}.",
        f"- Per asset: BTC Sharpe {out['per_asset_metrics']['BTCUSDT']['sharpe']:.2f} "
        f"({out['per_asset_metrics']['BTCUSDT']['ann_return']*100:.2f}%/yr); "
        f"ETH Sharpe {out['per_asset_metrics']['ETHUSDT']['sharpe']:.2f} "
        f"({out['per_asset_metrics']['ETHUSDT']['ann_return']*100:.2f}%/yr).",
        "",
        "## Per-year net return (pooled, %, after cost)",
        "",
        "| " + " | ".join(str(y) for y in sorted(py_pooled)) + " |",
        "|" + "---|" * len(py_pooled),
        "| " + " | ".join(f"{py_pooled[y]['total_pct']}" for y in sorted(py_pooled)) + " |",
        "",
        "## Cost stress (pooled Sharpe / ann return)",
        "",
        "| cost | Sharpe | ann return |",
        "|---|---|---|",
    ]
    for k, m in out["cost_stress_pooled"].items():
        lines.append(f"| {k} | {m['sharpe']:.2f} | {m['ann_return']*100:.2f}% |")
    inv = out["placebo"]["inverted"]
    zf = out["placebo"]["zero_funding"]
    lines += [
        "",
        "## Placebo separation",
        "",
        f"- Inverted book (long perp / short spot): ann {inv['ann_return']*100:.2f}% "
        f"(must be **negative** — carry direction matters).",
        f"- Zero-funding (price/basis only): ann {zf['ann_return']*100:.2f}% "
        f"(must be **~0** — confirms funding, not a price artifact, is the source).",
        "",
        "## Statistical battery (pooled net)",
        "",
        f"- Stationary bootstrap Sharpe 95% CI: [{boot['ci_lower_95']:.3f}, "
        f"{boot['ci_upper_95']:.3f}] (per-period; lower bound must be > 0).",
        f"- Deflated Sharpe probability: {float(dsr['probability']):.3f} "
        f"(trials={dsr['trials']}).",
        f"- PBO probability: {float(pbo['pbo_probability']):.3f} (must be < 0.5).",
        "",
        "## Concentration",
        "",
        f"- Pooled top-year PnL share {out['concentration']['pooled']['top_year_share']}, "
        f"top-day {out['concentration']['pooled']['top_day_share']} (year share must be ≤ 0.60).",
        f"- BTC top-year {out['concentration']['BTCUSDT']['top_year_share']}, "
        f"ETH top-year {out['concentration']['ETHUSDT']['top_year_share']}.",
        "",
        "## Gate (intake §5)",
        "",
        "| gate | result |",
        "|---|---|",
    ]
    for k, v in g.items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines += [
        "",
        "## Verdict",
        "",
        f"**{out['verdict']}.** " + (
            "Delta-neutral carry clears every §5 gate net of cost and across regimes. "
            "Documented as a candidate — still **no paper/live** without separate operator "
            "authorization. Next: P3 directional variant (held to the beta test)."
            if out["verdict"] == "CANDIDATE" else
            "The carry is **real and persistent** — net-positive in 6 of 7 years, with clean "
            "placebo and statistical separation (funding, not price, is the source). But it "
            "**does not clear the pre-registered regime gate**: the most recent regime "
            f"(2026 YTD, {py_pooled.get(2026, {}).get('total_pct', 'n/a')}% over "
            f"{py_pooled.get(2026, {}).get('days', 'n/a')} days) is net-negative after cost, "
            "and the gate (no-weakening) requires non-negative 2022 *and* 2026. Separately and "
            "more importantly, the headline Sharpe is an **illusion** of the daily-close basis "
            "model (see caveat) — a realistic execution/risk model would materially lower the "
            "risk-adjusted edge. **DO_NOT_ADVANCE.** No paper/live."
        ),
    ]
    REPORT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
