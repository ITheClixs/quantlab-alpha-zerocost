"""Event timestamp audit — the hard data gate for event_conditioned_macro_v1.

Verifies the event calendar is ex-ante and timestamp-clean against the SPY/QQQ
trading index before any backtest. Emits event_timestamp_audit.md with a
PASS / DEFERRED verdict per family. If FOMC fails the structural checks, the
channel is rejected on data grounds.

Usage:
    PYTHONPATH=src uv run python scripts/audit_event_timestamps.py
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import polars as pl

from quant_research_stack.signal_research.events.calendar import attach_event_features, load_fomc_dates

_BARS = "data/processed/vrp/bars/SPY.parquet"
_MANIFEST = Path("manifests/event_calendar/event_calendar_manifest.json")
_OUT = Path("reports/signal_research/event_macro_v1/event_timestamp_audit.md")


def main() -> int:
    manifest = json.loads(_MANIFEST.read_text())
    fomc = load_fomc_dates()
    bars = pl.read_parquet(_BARS).sort("date")
    trading_dates = [d if isinstance(d, date) else d.date() for d in bars["date"].to_list()]
    t0, t1 = trading_dates[0], trading_dates[-1]
    trading_set = set(trading_dates)

    # --- structural FOMC checks ---
    monotonic = fomc == sorted(fomc)
    unique = len(fomc) == len(set(fomc))
    per_year = dict(sorted(Counter(d.year for d in fomc).items()))
    in_range = [d for d in fomc if t0 <= d <= t1]
    aligned = [d for d in in_range if d in trading_set]
    misaligned = [d for d in in_range if d not in trading_set]
    bad_years = {y: c for y, c in per_year.items() if 2006 <= y <= 2025 and c != 8}

    feats = attach_event_features(bars, fomc_dates=fomc)
    coverage = {
        "fomc_t0": int(feats["fomc_t0"].sum()),
        "fomc_tm1": int(feats["fomc_tm1"].sum()),
        "fomc_tp1": int(feats["fomc_tp1"].sum()),
        "fomc_win2": int(feats["fomc_win2"].sum()),
        "fomc_win5": int(feats["fomc_win5"].sum()),
        "is_month_end": int(feats["is_month_end"].sum()),
        "is_quarter_end": int(feats["is_quarter_end"].sum()),
        "in_earnings_season": int(feats["in_earnings_season"].sum()),
    }
    fomc_pass = monotonic and unique and not bad_years and not misaligned and len(aligned) >= 100

    lines = [
        "# Event Timestamp Audit — event_conditioned_macro_v1",
        "",
        f"**Manifest:** `{_MANIFEST}`  built `{manifest['built_utc']}`",
        f"**Trading index:** `{_BARS}` — {len(trading_dates):,} bars, {t0} → {t1}",
        "**Binding question:** are the event dates ex-ante and timestamp-clean against the trading calendar?",
        "",
        "## FOMC (active)",
        "",
        f"- Total scheduled dates in manifest: **{len(fomc)}**  (range {fomc[0]} → {fomc[-1]})",
        f"- Strictly monotonic: **{monotonic}**  |  unique: **{unique}**",
        f"- Per-year counts 2006-2025 all == 8: **{not bad_years}**" + (f" (anomalies: {bad_years})" if bad_years else ""),
        f"- FOMC dates within data range [{t0}..{t1}]: **{len(in_range)}**",
        f"- Of those, on a SPY trading day: **{len(aligned)}**  |  misaligned (holiday/gap): **{len(misaligned)}**"
        + (f" → {misaligned}" if misaligned else ""),
        "- Provenance: " + "; ".join(f"{k}: {v}" for k, v in manifest["families"]["fomc"]["provenance"].items()),
        "",
        "### Look-ahead controls",
        "- FOMC dates are published ~1 year ahead; conditioning on `days_to_next_fomc` / window flags uses only the ex-ante schedule.",
        "- Emergency 2020 cuts (03-03, 03-15) excluded — not ex-ante scheduled (they would be look-ahead).",
        "- Decision is ~14:00 ET; any daily position is set at the **prior close**, never intrabar.",
        "- `attach_event_features` derives every column from the date + schedule only — no price input, so no future-return leakage is structurally possible.",
        "",
        "### Feature coverage on the SPY index",
        "",
        "| feature | days flagged |",
        "|---|---:|",
        *[f"| `{k}` | {v:,} |" for k, v in coverage.items()],
        "",
        "## CPI / NFP (DEFERRED)",
        "",
        f"- CPI: {manifest['families']['cpi']['reason']}",
        f"- NFP: {manifest['families']['nfp']['reason']}",
        "- **Not run in v1.** Approximating release dates (CPI mid-month / NFP first-Friday) would violate the",
        "  timestamp-clean gate; the CPI/CPI-combined variants are deferred until a clean release-date source is secured.",
        "",
        "## Earnings-season / Period-end (deterministic)",
        "",
        f"- earnings_season: {manifest['families']['earnings_season']['rule']}",
        f"- period_end: {manifest['families']['period_end']['rule']}",
        "- Fully deterministic from the trading-date index; no external source, no look-ahead.",
        "",
        "## Verdict",
        "",
        f"- **FOMC: {'PASS' if fomc_pass else 'FAIL'}** — {len(aligned)} clean, trading-day-aligned, ex-ante scheduled events in range.",
        "- **Earnings-season / period-end: PASS** (deterministic).",
        "- **CPI / NFP: DEFERRED** (no timestamp-clean source in this environment).",
        "",
        f"v1 proceeds on **FOMC + earnings-season + period-end**. Data window {t0} → {t1} "
        f"(~{len([d for d in aligned])} FOMC events) — event count is modest; bootstrap CIs will be wide "
        "(pre-registered failure mode #5).",
    ]
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(lines) + "\n")
    print(f"FOMC verdict: {'PASS' if fomc_pass else 'FAIL'} | aligned={len(aligned)} misaligned={len(misaligned)} bad_years={bad_years}")
    print(f"Wrote {_OUT}")
    return 0 if fomc_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
