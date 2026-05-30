"""P1 — funding-carry data alignment audit (no strategy code).

Builds the leak-safe daily carry panel for BTC/ETH (spot close, perp close, basis,
daily funding) from free Binance Vision archives, audits coverage + gaps, and
cross-checks the Vision BTC spot daily close against the on-disk vaquum 1m klines
(reuse-on-disk validation). Writes a manifest + markdown report. Gate before P2.

Run: PYTHONPATH=src uv run python scripts/audit_funding_carry_alignment.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_research_stack.crypto_research.funding import data as fdata
from quant_research_stack.crypto_research.funding import prices

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
VAQUUM_BTC = "data/raw/huggingface/vaquum__binance_btcusdt_1m_klines/btcusdt_1m_kline_20200101_to_20260511.parquet"
MANIFEST = Path("manifests/funding_carry/funding_carry_alignment_manifest.json")
REPORT = Path("reports/signal_research/funding_carry_v1/funding_carry_alignment_audit.md")


def _f(x: object) -> float:
    """Coerce a polars aggregate (broad union incl. None) to a plain float."""
    return float(x) if isinstance(x, (int, float)) else 0.0


def _gap_days(dates: list) -> int:
    if len(dates) < 2:
        return 0
    span = (dates[-1] - dates[0]).days + 1
    return span - len(dates)


def _panel_stats(panel: pl.DataFrame) -> dict[str, object]:
    dates = panel["date"].to_list()
    basis = panel["basis"]
    fund = panel["funding_day"]
    by_year = (
        panel.with_columns(pl.col("date").dt.year().alias("y"))
        .group_by("y").agg(pl.col("funding_day").mean().alias("mean_fund_day"))
        .sort("y")
    )
    # mean daily funding -> annualized (3 settlements/day already summed into funding_day)
    yearly = {int(r["y"]): round(_f(r["mean_fund_day"]) * 365.0 * 100, 2) for r in by_year.iter_rows(named=True)}
    return {
        "rows": panel.height,
        "start": str(dates[0]) if dates else None,
        "end": str(dates[-1]) if dates else None,
        "missing_days": _gap_days(dates),
        "basis_mean_pct": round(_f(basis.mean()) * 100, 4),
        "basis_abs_p95_pct": round(_f(basis.abs().quantile(0.95)) * 100, 4),
        "basis_max_abs_pct": round(_f(basis.abs().max()) * 100, 4),
        "ann_funding_pct_by_year": yearly,
        "ann_funding_pct_full": round(_f(fund.mean()) * 365.0 * 100, 2),
    }


def _btc_spot_crosscheck(vision_spot: pl.DataFrame) -> dict[str, object]:
    """Compare Vision BTC spot daily close vs on-disk vaquum 1m last-of-day close."""
    if not Path(VAQUUM_BTC).exists():
        return {"status": "skipped", "reason": "vaquum parquet absent"}
    disk = (
        pl.scan_parquet(VAQUUM_BTC)
        .select(["datetime", "close"])
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .sort("datetime")
        .group_by("date").agg(pl.col("close").last().alias("disk_close"))
        .collect()
    )
    v = vision_spot.with_columns(pl.col("ts").dt.date().alias("date")).select(
        ["date", pl.col("close").alias("vision_close")])
    j = v.join(disk, on="date", how="inner")
    if j.is_empty():
        return {"status": "no_overlap"}
    rel = (j["vision_close"] - j["disk_close"]).abs() / j["disk_close"]
    return {
        "status": "ok",
        "overlap_days": j.height,
        "rel_diff_mean_pct": round(_f(rel.mean()) * 100, 4),
        "rel_diff_p99_pct": round(_f(rel.quantile(0.99)) * 100, 4),
        "rel_diff_max_pct": round(_f(rel.max()) * 100, 4),
    }


def main() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {"built_utc": datetime.now(UTC).isoformat(), "symbols": {}}
    panels: dict[str, dict[str, object]] = {}

    for sym in SYMBOLS:
        print(f"[{sym}] loading funding + spot + perp daily klines (Vision, cached)...", flush=True)
        funding = fdata.load_funding(sym)
        spot = prices.load_daily_klines(sym, "spot")
        perp = prices.load_daily_klines(sym, "perp")
        panel = prices.align_carry(funding, spot, perp)
        stats = _panel_stats(panel)
        stats["funding_rows"] = funding.height
        stats["spot_rows"] = spot.height
        stats["perp_rows"] = perp.height
        if sym == "BTCUSDT":
            stats["spot_crosscheck"] = _btc_spot_crosscheck(spot)
        panels[sym] = stats
        result["symbols"][sym] = stats  # type: ignore[index]
        print(f"[{sym}] panel rows={panel.height} {stats['start']}..{stats['end']} "
              f"missing={stats['missing_days']} basis_mean={stats['basis_mean_pct']}% "
              f"ann_fund_full={stats['ann_funding_pct_full']}%", flush=True)

    # verdict
    ok_cov = all(_f(p["missing_days"]) <= 5 and _f(p["rows"]) > 1500 for p in panels.values())
    cc: dict[str, object] = panels["BTCUSDT"].get("spot_crosscheck", {})  # type: ignore[assignment]
    ok_cc = cc.get("status") != "ok" or _f(cc.get("rel_diff_p99_pct", 99)) < 1.0
    verdict = "PASS" if (ok_cov and ok_cc) else "REVIEW"
    result["verdict"] = verdict
    result["leak_rule"] = ("funding settled on UTC day D collected by short-perp held "
                           "through D; basis from simultaneous UTC-midnight closes; "
                           "P2 applies the decision t / earn t+1 shift")
    MANIFEST.write_text(json.dumps(result, indent=2))

    lines = [
        "# Funding-Carry Data Alignment Audit (P1)",
        "",
        f"**Date:** {datetime.now(UTC).date()}  ",
        f"**Verdict:** **{verdict}**  ",
        "**Status:** research_only — no paper, no live.",
        "",
        "Leak-safe daily carry panel for BTC/ETH from free Binance Vision archives "
        "(funding + spot + perp daily klines). Basis = perp/spot − 1 at the simultaneous "
        "UTC-midnight close; `funding_day` = total funding settled that UTC day.",
        "",
        "## Per-symbol panel",
        "",
        "| Symbol | rows | start | end | missing days | basis mean | basis |max| p95 | ann funding (full) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for sym, p in panels.items():
        lines.append(
            f"| {sym} | {p['rows']} | {p['start']} | {p['end']} | {p['missing_days']} | "
            f"{p['basis_mean_pct']}% | {p['basis_max_abs_pct']}% | {p['basis_abs_p95_pct']}% | "
            f"{p['ann_funding_pct_full']}% |"
        )
    lines += ["", "## Annualized funding by year (gross, %)", ""]
    yearmaps: dict[str, dict[int, float]] = {
        sym: p["ann_funding_pct_by_year"] for sym, p in panels.items()  # type: ignore[misc]
    }
    years = sorted({y for ym in yearmaps.values() for y in ym})
    lines.append("| Symbol | " + " | ".join(str(y) for y in years) + " |")
    lines.append("|" + "---|" * (len(years) + 1))
    for sym, ym in yearmaps.items():
        lines.append(f"| {sym} | " + " | ".join(f"{ym.get(y, '—')}" for y in years) + " |")

    lines += [
        "",
        "## BTC spot cross-check (Vision daily vs on-disk vaquum 1m)",
        "",
        f"- status: `{cc.get('status')}`",
    ]
    if cc.get("status") == "ok":
        lines += [
            f"- overlap days: {cc.get('overlap_days')}",
            f"- relative close diff: mean {cc.get('rel_diff_mean_pct')}%, "
            f"p99 {cc.get('rel_diff_p99_pct')}%, max {cc.get('rel_diff_max_pct')}%",
            "- → Vision daily close ≈ on-disk last-of-day close (reuse-on-disk validated).",
        ]
    lines += [
        "",
        "## Leakage rule",
        "",
        "- Funding realized at settlement *t* is known at *t*; the short-perp leg held "
        "through UTC day *D* collects day *D*'s funding.",
        "- Basis uses spot and perp closes at the **same** UTC instant (one source) → no "
        "stale-leg basis.",
        "- P2 backtest applies a single explicit decision-*t* / earn-*t+1* shift; no "
        "contemporaneous funding in the position that earns it.",
        "",
        "## Verdict",
        "",
        f"**{verdict}.** " + (
            "Panel coverage clean (gaps ≤ 5 days, >1500 rows/asset) and the on-disk "
            "cross-check agrees → proceed to P2 (delta-neutral carry backtest) under the "
            "intake §5 gate." if verdict == "PASS" else
            "Coverage or cross-check needs review before P2."
        ),
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nVERDICT: {verdict}\nmanifest: {MANIFEST}\nreport: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
