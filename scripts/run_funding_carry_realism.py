"""P2.5 — funding-carry realism upgrade (operator-requested).

Replaces the artificially-smooth daily-close basis with carry marked on the 8h
funding-settlement grid (restores true intraday basis variance -> honest Sharpe),
adds execution slippage, and models isolated-margin liquidation on the short perp.
Compares the daily illusion vs the 8h-honest book and reports the realistic
risk-adjusted edge. research_only — no paper, no live.

Run: PYTHONPATH=src uv run python scripts/run_funding_carry_realism.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.crypto_research.funding import carry, prices
from quant_research_stack.crypto_research.funding import data as fdata
from quant_research_stack.crypto_research.perps import validation as val

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SPOT_TAKER_BPS = 10.0
PERP_TAKER_BPS = 5.0
SLIP_BPS = 5.0
LEVERAGES = (3.0, 5.0, 10.0)
MANIFEST = Path("manifests/funding_carry/funding_carry_realism_manifest.json")
REPORT = Path("reports/signal_research/funding_carry_v1/funding_carry_realism_results.md")


def _panel_8h(sym: str) -> pl.DataFrame:
    return prices.align_carry_8h(fdata.load_funding(sym),
                                 prices.load_klines(sym, "spot", "8h"),
                                 prices.load_klines(sym, "perp", "8h"))


def _panel_daily(sym: str) -> pl.DataFrame:
    return prices.align_carry(fdata.load_funding(sym),
                              prices.load_klines(sym, "spot", "1d"),
                              prices.load_klines(sym, "perp", "1d"))


def main() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    p8 = {s: _panel_8h(s) for s in SYMBOLS}
    # coverage guard: the 8h panel must retain ~all funding settlements in range
    # (an exact-timestamp join silently dropped ~45% before the ms-jitter fix).
    for s in SYMBOLS:
        fund = fdata.load_funding(s)
        lo, hi = p8[s]["ts"].min(), p8[s]["ts"].max()
        expect = fund.filter((pl.col("funding_time") >= lo) & (pl.col("funding_time") <= hi)).height
        got = p8[s].height
        if got < 0.98 * expect:
            raise SystemExit(f"[{s}] 8h panel coverage {got}/{expect} (<98%) — join is dropping settlements")
        print(f"[{s}] 8h panel coverage {got}/{expect} settlements", flush=True)
    # common 8h settlement grid so the pooled book aligns exactly
    common = p8[SYMBOLS[0]].select("ts")
    for s in SYMBOLS[1:]:
        common = common.join(p8[s].select("ts"), on="ts", how="inner")
    p8 = {s: p8[s].join(common, on="ts", how="inner").sort("ts") for s in SYMBOLS}

    honest = {s: carry.carry_returns_8h(p8[s], spot_taker_bps=SPOT_TAKER_BPS,
                                        perp_taker_bps=PERP_TAKER_BPS, slip_bps=SLIP_BPS)
              for s in SYMBOLS}
    pooled = carry.pooled_book(honest, periods_per_year=carry.PPY_8H)

    # daily illusion (for the comparison)
    pd_ = {s: _panel_daily(s) for s in SYMBOLS}
    illus = {s: carry.carry_returns(pd_[s], spot_taker_bps=SPOT_TAKER_BPS,
                                    perp_taker_bps=PERP_TAKER_BPS) for s in SYMBOLS}
    illus_pooled = carry.pooled_book(illus)

    py = carry.per_year(pooled.dates, pooled.net, periods_per_year=carry.PPY_8H)
    boot = val.bootstrap_sharpe_payload(pooled.net, resamples=2000, seed=17)

    # liquidation tail (per asset) + stressed pooled metrics at each leverage
    liq_diag = {s: carry.liquidation_diag(p8[s], leverages=LEVERAGES) for s in SYMBOLS}
    rt = (SPOT_TAKER_BPS + PERP_TAKER_BPS) * 1e-4
    slip = SLIP_BPS * 1e-4
    liq_stress = {}
    for lev in LEVERAGES:
        nets = []
        n_liq = 0
        for s in SYMBOLS:
            n = honest[s].net.copy()
            mask = carry.adverse_dbasis(p8[s]) > (1.0 / lev)
            n[mask] -= (1.0 / lev) + (rt + 2.0 * slip)
            n_liq += int(mask.sum())
            nets.append(n)
        pooled_stressed = np.vstack(nets).mean(axis=0)
        liq_stress[f"{lev:g}x"] = {**carry.metrics_ann(pooled_stressed, carry.PPY_8H),
                                   "n_liquidations": float(n_liq)}

    out = {
        "built_utc": datetime.now(UTC).isoformat(),
        "status": "research_only — no paper, no live",
        "costs_bps": {"spot_taker": SPOT_TAKER_BPS, "perp_taker": PERP_TAKER_BPS, "slippage": SLIP_BPS},
        "comparison": {
            "daily_close_illusion": {
                "pooled_sharpe": round(illus_pooled.metrics["sharpe"], 2),
                "pooled_ann_return_pct": round(illus_pooled.metrics["ann_return"] * 100, 2),
                "pooled_ann_vol_pct": round(illus_pooled.metrics["ann_vol"] * 100, 2),
                "pooled_max_dd_pct": round(illus_pooled.metrics["max_drawdown"] * 100, 2),
            },
            "honest_8h": {
                "pooled_sharpe": round(pooled.metrics["sharpe"], 2),
                "pooled_ann_return_pct": round(pooled.metrics["ann_return"] * 100, 2),
                "pooled_ann_vol_pct": round(pooled.metrics["ann_vol"] * 100, 2),
                "pooled_max_dd_pct": round(pooled.metrics["max_drawdown"] * 100, 2),
            },
        },
        "honest_per_asset": {s: honest[s].metrics for s in SYMBOLS},
        "honest_pooled_per_year": py,
        "bootstrap_sharpe": {"ci_lower_95": round(float(boot["ci_lower_95"]), 3),
                             "ci_upper_95": round(float(boot["ci_upper_95"]), 3)},
        "liquidation_tail": liq_diag,
        "liquidation_stressed_pooled": liq_stress,
    }
    MANIFEST.write_text(json.dumps(out, indent=2, default=str))
    _write_report(out)
    di = illus_pooled.metrics
    ho = pooled.metrics
    print(f"\ndaily illusion pooled Sharpe = {di['sharpe']:.2f}")
    print(f"HONEST 8h pooled Sharpe      = {ho['sharpe']:.2f} "
          f"(ann {ho['ann_return']*100:.2f}%, vol {ho['ann_vol']*100:.2f}%, "
          f"maxDD {ho['max_drawdown']*100:.2f}%)")
    print("liquidation-stressed pooled:")
    for k, m in liq_stress.items():
        print(f"  {k}: Sharpe {m['sharpe']:.2f}, ann {m['ann_return']*100:.2f}%, "
              f"{int(m['n_liquidations'])} liq events")
    print(f"manifest: {MANIFEST}\nreport: {REPORT}")


def _write_report(out: dict) -> None:
    c = out["comparison"]
    py = out["honest_pooled_per_year"]
    lines = [
        "# Funding-Carry — Realism Upgrade (8h-marked, slippage, liquidation)",
        "",
        f"**Date:** {datetime.now(UTC).date()}  ",
        "**Status:** research_only — no paper, no live.",
        "",
        "Operator-requested realism pass. The P2 daily-close model smoothed away the "
        "intraday spot-perp basis variance, inflating the Sharpe to ~8.6. Here the carry "
        "is marked on the **8h funding-settlement grid** (true basis variance), with "
        f"execution slippage ({out['costs_bps']['slippage']}bps/leg) and an isolated-margin "
        "liquidation model on the short perp.",
        "",
        "## Daily illusion vs honest 8h (pooled BTC+ETH)",
        "",
        "| | daily-close (illusion) | 8h-marked (honest) |",
        "|---|---|---|",
        f"| Sharpe | {c['daily_close_illusion']['pooled_sharpe']} | "
        f"**{c['honest_8h']['pooled_sharpe']}** |",
        f"| ann return | {c['daily_close_illusion']['pooled_ann_return_pct']}% | "
        f"{c['honest_8h']['pooled_ann_return_pct']}% |",
        f"| ann vol | {c['daily_close_illusion']['pooled_ann_vol_pct']}% | "
        f"{c['honest_8h']['pooled_ann_vol_pct']}% |",
        f"| max DD | {c['daily_close_illusion']['pooled_max_dd_pct']}% | "
        f"{c['honest_8h']['pooled_max_dd_pct']}% |",
        "",
        f"Bootstrap Sharpe 95% CI (8h net): [{out['bootstrap_sharpe']['ci_lower_95']}, "
        f"{out['bootstrap_sharpe']['ci_upper_95']}] (per-bar).",
        "",
        "## Honest per-year net (pooled, %, 8h-marked)",
        "",
        "| " + " | ".join(str(y) for y in sorted(py)) + " |",
        "|" + "---|" * len(py),
        "| " + " | ".join(f"{py[y]['total_pct']}" for y in sorted(py)) + " |",
        "",
        "## Liquidation tail (intrabar adverse basis on the short perp)",
        "",
        "| asset | max adverse Δbasis | p99.9 | liq events 3x / 5x / 10x |",
        "|---|---|---|---|",
    ]
    for s, d in out["liquidation_tail"].items():
        ev = d["liq_events"]
        lines.append(f"| {s} | {d['max_adverse_dbasis_pct']}% | {d['p999_adverse_dbasis_pct']}% | "
                     f"{ev['3x']} / {ev['5x']} / {ev['10x']} |")
    lines += [
        "",
        "## Liquidation-stressed pooled (conservative: lose posted margin + re-hedge)",
        "",
        "| leverage | Sharpe | ann return | liq events |",
        "|---|---|---|---|",
    ]
    for k, m in out["liquidation_stressed_pooled"].items():
        lines.append(f"| {k} | {m['sharpe']:.2f} | {m['ann_return']*100:.2f}% | "
                     f"{int(m['n_liquidations'])} |")
    hs = c["honest_8h"]["pooled_sharpe"]
    ds = c["daily_close_illusion"]["pooled_sharpe"]
    lines += [
        "",
        "## Honest conclusion (corrects the P2 'illusion' hypothesis)",
        "",
        f"- **8h marking did NOT deflate the Sharpe** ({ds} -> {hs}). The spot-perp basis "
        "is genuinely tight even at 8h, so the daily-close model was *not* an illusion on "
        "the basis-variance axis. The P2 guess that the realistic Sharpe was ~1-2 was wrong "
        "on the mechanism.",
        "- The unlevered, fully-collateralized delta-neutral carry **genuinely** shows "
        f"Sharpe ~{hs} / ~14%/yr over 2020-2026 — real, because funding is a steady positive "
        "drip and the hedge tracks tightly (low daily/8h variance).",
        "- **The catch is the fat left tail, not the daily vol.** (1) At 1x it is "
        "capital-inefficient (100% margin on both legs). (2) Any leverage to fix that "
        "introduces short-perp **liquidation in crashes** — the stress table above goes "
        "negative at 3x (-17%/yr) and catastrophic at 10x (-90%). The high Sharpe does not "
        "price this tail.",
        "- The liquidation proxy uses non-simultaneous intrabar high/low so it **overstates** "
        "adverse basis; the 8h-close model **understates** intrabar dislocation. Pricing the "
        "tail precisely needs intraday simultaneous spot+perp (1m/tick) data.",
        "- **Verdict unchanged: DO_NOT_ADVANCE.** A real, free, market-neutral carry, but: "
        "(a) the P2 regime gate fails (2026 net-negative), (b) the edge is decaying with "
        "crowding, and (c) it is deployable only unlevered/capital-inefficient with an "
        "unpriced crash-liquidation tail. research_only — no paper, no live.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
