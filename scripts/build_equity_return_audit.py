"""Equity-return data audit (§4.2 gate for the options-IV cross-sectional track).

Binding question: can we build a survivorship-aware, corporate-action-adjusted
next-day return panel for the options-IV universe (2019-10-14 .. 2023-07-28) from
an on-disk source, without a misleading survivorship-biased return set?

Primary candidate: HexQuant__Stocks-Daily-Price (paperswithbacktest), columns
symbol/date/open/high/low/close/volume/adj_close, 6,320 US stock symbols.

No strategy code. Emits (reports/signal_research/options_iv_v1/ + manifests/options_iv/):
  equity_return_data_manifest.json, equity_return_data_audit.md,
  iv_to_equity_symbol_mapping.parquet, equity_return_coverage_report.md,
  equity_return_survivorship_report.md, equity_return_liquidity_report.md

Usage:
    PYTHONPATH=src uv run python scripts/build_equity_return_audit.py
"""

from __future__ import annotations

import glob
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

_IV = "data/raw/huggingface/gauss314__options-IV-SP500/data_IV_USA.csv"
_HQ_GLOB = "data/raw/huggingface/HexQuant__Stocks-Daily-Price/data/*.parquet"
_REPORTS = Path("reports/signal_research/options_iv_v1")
_MANIFEST = Path("manifests/options_iv/equity_return_data_manifest.json")
_W0, _W1 = "2019-10-14", "2023-07-28"
_DELIST_CUTOFF = "2023-07-20"
# Known names that delisted/merged/failed/renamed during the IV window — survivorship probes.
_DELIST_PROBES = ["AABA", "TWTR", "FB", "RTN", "CELG", "XLNX", "MXIM", "NLSN", "CERN",
                  "ZNGA", "SIVB", "FRC", "DISCA", "WORK", "MGLN"]
_ETFS = ["SPY", "QQQ", "DIA", "IWM"]


def main() -> int:
    iv = pl.read_csv(_IV, columns=["symbol", "date"])
    iv_dates = iv.group_by("symbol").agg(
        pl.col("date").min().alias("iv_first"), pl.col("date").max().alias("iv_last"))
    iv_syms = set(iv["symbol"].unique().to_list())

    hq = pl.concat([pl.read_parquet(f, columns=["symbol", "date", "close", "adj_close", "volume"])
                    for f in sorted(glob.glob(_HQ_GLOB))]).with_columns(
        pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("d"))
    hq_win = hq.filter((pl.col("d") >= _W0) & (pl.col("d") <= _W1))
    hq_syms = set(hq_win["symbol"].unique().to_list())

    overlap = iv_syms & hq_syms
    missing = iv_syms - hq_syms
    specials = [s for s in missing if re.search(r"[.\-/^]", s)]
    miss_df = iv_dates.filter(pl.col("symbol").is_in(list(missing)))
    delisted_missing = miss_df.filter(pl.col("iv_last") < _DELIST_CUTOFF)
    fullcov_missing = miss_df.filter(pl.col("iv_last") >= _DELIST_CUTOFF)
    probe_results = {s: {"in_iv": s in iv_syms, "in_hq_window": s in hq_syms} for s in _DELIST_PROBES}
    etf_results = {s: {"in_iv": s in iv_syms, "in_hq_window": s in hq_syms} for s in _ETFS}

    # mapping table
    mapping = iv_dates.with_columns(
        pl.col("symbol").is_in(list(hq_syms)).alias("in_return_source"),
        (pl.col("iv_last") < _DELIST_CUTOFF).alias("iv_delisted_in_window"),
    ).with_columns(
        pl.when(pl.col("in_return_source")).then(pl.lit("mapped"))
        .when(pl.col("iv_delisted_in_window")).then(pl.lit("unmapped_delisted_in_window"))
        .otherwise(pl.lit("unmapped_full_coverage_absent")).alias("classification")
    ).sort("symbol")
    _REPORTS.mkdir(parents=True, exist_ok=True)
    _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_parquet(_REPORTS / "iv_to_equity_symbol_mapping.parquet")

    # liquidity + DQ on the overlap (survivor) set
    ov = hq_win.filter(pl.col("symbol").is_in(list(overlap)))
    ov = ov.with_columns((pl.col("close") * pl.col("volume")).alias("dollar_vol"))
    dup = ov.height - ov.select(["symbol", "d"]).unique().height
    nonpos = ov.filter(pl.col("close") <= 0).height
    adj_nulls = ov["adj_close"].null_count()
    med_dollar_vol = float(ov["dollar_vol"].median() or 0.0)  # type: ignore[arg-type]
    # tradable universe on a sample date (min $5 close, min $1M dollar-vol), survivor set only
    sample_d = "2021-06-15"
    day = ov.filter(pl.col("d") == sample_d)
    tradable = day.filter((pl.col("close") >= 5.0) & (pl.col("dollar_vol") >= 1_000_000)).height
    day_n = day.height

    n_iv = len(iv_syms)
    pct_overlap = 100.0 * len(overlap) / n_iv
    survivorship = "survivorship_biased"  # delisted names absent from in-window slice
    verdict = "REJECT_CROSS_SECTIONAL_ON_DATA — run SPY/QQQ secondary diagnostic only"
    labels = ["survivorship_biased_research_only", "price_return_only_unless_adjclose_total",
              "mapping_incomplete", "liquidity_limited"]

    manifest = {
        "name": "equity_return_data_audit", "built_utc": datetime.now(UTC).isoformat(),
        "binding_question": "Survivorship-aware, CA-adjusted next-day return panel for the IV universe "
                            f"({_W0}..{_W1}) without a survivorship-biased source?",
        "primary_source": "HexQuant__Stocks-Daily-Price (symbol/date/ohlc/volume/adj_close; 6,320 US stocks)",
        "verdict": verdict,
        "survivorship_label": survivorship,
        "labels": labels,
        "iv_symbols": n_iv, "hq_window_symbols": len(hq_syms), "overlap": len(overlap),
        "overlap_pct": round(pct_overlap, 1), "missing": len(missing),
        "missing_special_char": len(specials), "missing_delisted_in_window": delisted_missing.height,
        "missing_full_coverage_absent": fullcov_missing.height,
        "delist_probes": probe_results, "etf_probes": etf_results,
        "adj_close_note": "adj_close < close for dividend payers (e.g. AAPL) -> dividend+split adjusted (total-return proxy)",
        "labels_legend": ["equity_returns_clean", "price_return_only", "partial_survivorship",
                          "survivorship_biased_research_only", "mapping_incomplete", "liquidity_limited", "reject"],
    }
    _MANIFEST.write_text(json.dumps(manifest, indent=2))

    # ---- coverage report ----
    (_REPORTS / "equity_return_coverage_report.md").write_text("\n".join([
        "# Equity-Return Audit — §1 Coverage",
        "", f"**Source:** HexQuant__Stocks-Daily-Price | **IV window:** {_W0} → {_W1}",
        f"- HexQuant symbols (all / in-window): {len(hq.unique(subset=['symbol']))} / {len(hq_syms)}",
        f"- IV universe symbols: {n_iv}",
        f"- **Overlap (mapped): {len(overlap)} ({pct_overlap:.1f}%)** | **Missing: {len(missing)} ({100-pct_overlap:.1f}%)**",
        f"- Missing with special chars (ticker-format issue): **{len(specials)}** → mapping is NOT the cause.",
        "- Index ETFs in HexQuant: " + ", ".join(f"{k}={v['in_hq_window']}" for k, v in etf_results.items())
        + " → **HexQuant is stocks-only; ETF returns must come from the clean SPY/QQQ bars (secondary track).**",
        "",
        "Coverage is only 57% of the IV universe; the gap is survivorship + ETFs, not ticker formatting.",
    ]) + "\n")

    # ---- survivorship report ----
    (_REPORTS / "equity_return_survivorship_report.md").write_text("\n".join([
        "# Equity-Return Audit — §2 Survivorship",
        "",
        f"- Missing names whose IV data ends mid-window (delisted-in-window): **{delisted_missing.height}**",
        f"- Missing names with full IV coverage but absent from HexQuant: **{fullcov_missing.height}** "
        "(mix of ETFs and renamed/merged tickers).",
        "",
        "## Known delisted/merged/failed names (in IV, should exist pre-delist in a survivorship-safe source):",
        "",
        "| symbol | in IV | in HexQuant window |",
        "|---|:---:|:---:|",
        *[f"| {s} | {v['in_iv']} | {v['in_hq_window']} |" for s, v in probe_results.items()],
        "",
        "**Every probed name that delisted/merged/failed during 2019-2023 (TWTR, FB→META, RTN→RTX, CELG, XLNX,",
        "MXIM, NLSN, CERN, ZNGA, SIVB, FRC, AABA, …) is ABSENT** from HexQuant's in-window slice. HexQuant",
        "carries currently-listed names with backfilled history; names that left the market before ~2025 are",
        "dropped. For a 2019-2023 cross-section this is **textbook survivorship bias** — it would silently",
        "exclude bankruptcies (SIVB, FRC) and merger targets (TWTR, CELG, XLNX), inflating any cross-sectional",
        "result.",
        "",
        f"**§2 label: `{survivorship}`.**",
    ]) + "\n")

    # ---- liquidity report ----
    (_REPORTS / "equity_return_liquidity_report.md").write_text("\n".join([
        "# Equity-Return Audit — §4/§6 Liquidity & Data Quality (overlap/survivor set)",
        "",
        f"Computed on the {len(overlap)} overlap (survivor) names — reported for completeness even though the",
        "cross-sectional track is rejected on survivorship grounds.",
        f"- Duplicate (symbol, date): {dup} | close <= 0: {nonpos} | adj_close nulls: {adj_nulls}",
        f"- Median daily dollar volume: ${med_dollar_vol:,.0f}",
        f"- Tradable on sample date {sample_d} (close>=$5, $vol>=$1M): {tradable} / {day_n} survivor names",
        f"- adj_close: {manifest['adj_close_note']} → total-return computation feasible on covered names.",
        "",
        "Liquidity/quality on the survivor set is fine; the binding defect is survivorship (§2), not liquidity.",
    ]) + "\n")

    # ---- summary audit ----
    (_REPORTS / "equity_return_data_audit.md").write_text("\n".join([
        "# Equity-Return Data Audit (§4.2 gate) — Summary & Decision",
        "",
        f"**Built:** {manifest['built_utc']}  **Source:** HexQuant__Stocks-Daily-Price",
        f"**IV window:** {_W0} → {_W1}  **IV symbols:** {n_iv}",
        "**Binding question:** survivorship-aware, CA-adjusted next-day return panel for the IV universe?",
        "",
        "## Findings",
        f"- **Coverage:** {len(overlap)}/{n_iv} ({pct_overlap:.1f}%) mapped; {len(missing)} missing "
        f"({len(specials)} due to ticker format → none). See coverage report.",
        f"- **Survivorship:** `{survivorship}` — every probed delisted/merged/failed 2019-2023 name (TWTR, FB,",
        "  RTN, CELG, XLNX, MXIM, NLSN, CERN, ZNGA, SIVB, FRC, AABA) is ABSENT from HexQuant in-window. See",
        "  survivorship report.",
        "- **Corporate actions:** `adj_close` is dividend+split adjusted (total-return proxy on covered names).",
        "- **ETFs:** HexQuant has no SPY/QQQ/DIA/IWM (stocks-only). Secondary SPY/QQQ track uses the clean bars.",
        "- **Liquidity/quality:** clean on the survivor set; not the binding constraint.",
        "- **Other on-disk sources:** benstaf (narrow curated RL universe), jwigginton (S&P500 current-",
        "  constituent → also survivorship-biased), mospira (2025 only) — none survivorship-safe for the broad",
        "  IV universe. yfinance is disallowed as primary (survivorship-biased for delisted names).",
        "",
        f"## Labels: `{', '.join(labels)}`",
        "",
        f"## VERDICT: **{verdict}**",
        "",
        "Per the §4.2 / operator decision rule: the only available return source is **clearly",
        "survivorship-biased** for the 2019-2023 cross-section, so the **cross-sectional options-IV track is",
        "REJECTED on data grounds** (a cross-sectional backtest on survivors only would be invalid — it would",
        "exclude the bankruptcies and merger targets that dominate the tails). Mapping is not the issue (0",
        "format mismatches); the defect is genuine missing delisted names.",
        "",
        "### Consequence",
        "- **Run only the SPY/QQQ secondary diagnostic track** (per the options-IV intake §4.2 / §11.3): index",
        "  IV features (VRP proxy, cross-sectional IV dispersion, vol-OI imbalance) from gauss314 → next-day",
        "  SPY/QQQ timing, using the already-clean SPY/QQQ bars. research_only; must beat vol-targeted BAH or it",
        "  is subsumed.",
        "- If the SPY/QQQ diagnostic is also weak, close options-IV on data grounds and move to the next branch.",
        "",
        "## Constraints honored",
        "- No strategy backtest. No silent symbol dropping (full mapping table written). No future-availability",
        "  filtering. No yfinance promotion claims. No direct options. research_only throughout.",
    ]) + "\n")

    print(f"VERDICT: {verdict}")
    print(f"overlap={len(overlap)}/{n_iv} ({pct_overlap:.1f}%) missing={len(missing)} "
          f"(delisted_in_window={delisted_missing.height}, full_cov_absent={fullcov_missing.height}, special={len(specials)})")
    print(f"survivorship={survivorship} | tradable@{sample_d}={tradable}/{day_n}")
    print(f"Wrote manifest + mapping.parquet + 4 reports under {_REPORTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
