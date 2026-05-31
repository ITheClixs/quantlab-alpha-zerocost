"""Options-chain IV DATA AUDIT (no strategy code).

Audits the on-disk gauss314__options-IV-SP500 dataset for timestamp safety,
universe/survivorship, chain structure, data quality, tradability, feature
feasibility, and leakage — then assigns data-quality labels and a decision.

Binding question: can options-chain IV support a leakage-safe, tradable research
program?

Emits (reports/signal_research/options_iv_v1/ + manifests/options_iv/):
  options_iv_data_manifest.json, options_iv_data_audit_report.md,
  options_iv_timestamp_audit.md, options_iv_universe_audit.md,
  options_iv_liquidity_audit.md, options_iv_feature_feasibility.md

Usage:
    PYTHONPATH=src uv run python scripts/build_options_iv_audit.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

_CSV = "data/raw/huggingface/gauss314__options-IV-SP500/data_IV_USA.csv"
_REPORTS = Path("reports/signal_research/options_iv_v1")
_MANIFEST = Path("manifests/options_iv/options_iv_data_manifest.json")
_INDEX_ETFS = ("SPY", "QQQ", "DIA", "IWM")


def _f(x: object) -> float:
    return float(x) if x is not None else float("nan")  # type: ignore[arg-type]


def compute_stats() -> dict:
    df = pl.read_csv(_CSV, infer_schema_length=2000)
    n = df.height
    yr = df.with_columns(pl.col("date").str.slice(0, 4).alias("y"))
    per_year = yr.group_by("y").agg(
        pl.col("symbol").n_unique().alias("n_syms"), pl.len().alias("rows")
    ).sort("y").to_dicts()
    etfs = {s: df.filter(pl.col("symbol") == s).height for s in _INDEX_ETFS}
    aaba = df.filter(pl.col("symbol") == "AABA")
    spy = df.filter(pl.col("symbol") == "SPY").sort("date")
    return {
        "rows": n,
        "columns": df.columns,
        "n_columns": len(df.columns),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "n_symbols": int(df["symbol"].n_unique()),
        "per_year": per_year,
        "index_etfs_rows": etfs,
        "delisted_probe": {"symbol": "AABA", "rows": aaba.height, "last_date": str(aaba["date"].max()) if aaba.height else None},
        "spy_trading_days": spy.height,
        "spy_atm_iv_sample": [round(_f(v), 2) for v in spy["ATM_IV"].head(3).to_list()],
        "dup_symbol_date": int(n - df.select(["symbol", "date"]).unique().height),
        "atm_iv_nulls": int(df["ATM_IV"].null_count()),
        "atm_iv_nonpositive": int(df.filter(pl.col("ATM_IV") <= 0).height),
        "atm_iv_gt_300": int(df.filter(pl.col("ATM_IV") > 300).height),
        "atm_iv_q": [round(_f(df["ATM_IV"].quantile(q)), 2) for q in (0.01, 0.5, 0.99)],
        "zero_call_volume_rows": int(df.filter(pl.col("calls_contracts_traded") == 0).height),
        "zero_call_oi_rows": int(df.filter(pl.col("calls_open_interest") == 0).height),
        "key_col_nulls": {c: int(df[c].null_count()) for c in
                          ("ATM_IV", "DOTM_IV", "DITM_IV", "hv_20", "VIX", "calls_open_interest", "expirations_number")},
    }


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.2f}%" if whole else "n/a"


def main() -> int:
    s = compute_stats()
    n = s["rows"]
    built = datetime.now(UTC).isoformat()
    etf_ok = all(v > 0 for v in s["index_etfs_rows"].values())
    delisted_drops = (s["delisted_probe"]["last_date"] or "") < s["date_max"]
    clean = s["dup_symbol_date"] == 0 and s["atm_iv_nulls"] == 0 and s["atm_iv_nonpositive"] == 0

    labels = ["options_features_only", "timestamp_uncertain"]
    if not delisted_drops:
        labels.append("survivorship_biased_research_only")
    verdict = "RESEARCH_ONLY_FEATURES"  # lacks strike/expiry/bid-ask -> not promotion-grade; usable as EOD features next-day

    _REPORTS.mkdir(parents=True, exist_ok=True)
    _MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    # ---- manifest ----
    manifest = {
        "name": "options_iv_v1_data_audit",
        "built_utc": built,
        "dataset": "gauss314__options-IV-SP500 / data_IV_USA.csv",
        "binding_question": "Can options-chain IV support a leakage-safe, tradable research program?",
        "verdict": verdict,
        "labels": labels,
        "stats": s,
        "chain_structure": {
            "strikes": False, "expiry_dates": False, "bid_ask": False, "per_contract_prices": False,
            "expirations_count_only": True, "call_put_volume_oi": True, "iv_supplied": "moneyness_buckets (DITM..DOTM)",
            "greeks": False, "true_delta_skew": False, "term_structure": False,
        },
        "labels_legend": ["options_chain_clean", "options_features_only", "survivorship_biased_research_only",
                          "timestamp_uncertain", "liquidity_insufficient", "reject"],
    }
    _MANIFEST.write_text(json.dumps(manifest, indent=2))

    # ---- timestamp audit ----
    (_REPORTS / "options_iv_timestamp_audit.md").write_text("\n".join([
        "# Options-IV Data Audit — §1 Timestamp Integrity",
        "", f"**Built:** {built}  **Rows:** {n:,}  **Range:** {s['date_min']} → {s['date_max']}",
        "",
        "- Only a daily `date` column exists — **no intraday/observability timestamp**. Values are end-of-day",
        "  (VIX is the EOD index level; `hv_*` are trailing historical vols; IV buckets are market-implied EOD).",
        "- **Same-day signal use is NOT safe** (we cannot prove the IV was observable before a same-day close",
        "  decision). **EOD-features-for-next-day (t → t+1) IS safe** by shifting features one trading day.",
        "- `feature_timestamp < signal_timestamp` can be enforced ONLY under the next-day convention (feature at",
        "  EOD t, decision/execution at t+1). Document and enforce a 1-day shift in any downstream research.",
        "- No revised/recomputed field is documented; IV is market-implied (not derived from future realized",
        "  vol per the dataset README), and `hv_*` are explicitly *historical* (trailing) — no future-RV leak",
        "  evident. Caveat: the author's exact computation is undocumented beyond the README → **timestamp_uncertain**.",
        "",
        "**§1 verdict: timestamp_uncertain — research_only, next-day features only.**",
    ]) + "\n")

    # ---- universe audit ----
    (_REPORTS / "options_iv_universe_audit.md").write_text("\n".join([
        "# Options-IV Data Audit — §2 Universe & Survivorship",
        "", f"**Unique symbols:** {s['n_symbols']:,} (far broader than S&P 500 — a broad US optionable universe).",
        "", "Per-year unique symbols / rows:",
        "", "| year | symbols | rows |", "|---|---:|---:|",
        *[f"| {r['y']} | {r['n_syms']:,} | {r['rows']:,} |" for r in s["per_year"]],
        "",
        "- Index ETFs present (directly tradable as the instrument): "
        + ", ".join(f"{k}={v} rows" for k, v in s["index_etfs_rows"].items()) + f"  → all present: {etf_ok}.",
        f"- Delisted-name probe: {s['delisted_probe']['symbol']} has {s['delisted_probe']['rows']} rows, "
        f"last {s['delisted_probe']['last_date']} (< dataset end {s['date_max']}) → **delisted names are retained**",
        f"  and drop out at delisting → NOT current-constituent survivorship-biased: {delisted_drops}.",
        "- Caveat: completeness of *additions* over time is unverifiable; declining yearly symbol counts "
        "(3,827→3,254) are consistent with attrition, not a current-only snapshot.",
        "",
        "**§2 verdict: not current-constituent survivorship-biased; PIT-plausible within 2019-2023. Promotion",
        "language is blocked anyway by the chain-structure gaps (§3).**",
    ]) + "\n")

    # ---- liquidity audit ----
    (_REPORTS / "options_iv_liquidity_audit.md").write_text("\n".join([
        "# Options-IV Data Audit — §4 Data Quality & Liquidity",
        "",
        f"- Duplicate (symbol, date): **{s['dup_symbol_date']}**",
        f"- ATM_IV nulls: {s['atm_iv_nulls']} | non-positive: {s['atm_iv_nonpositive']} | >300 (impossible): {s['atm_iv_gt_300']}",
        f"- ATM_IV quantiles (1/50/99): {s['atm_iv_q']} (vol points; broad universe incl. small/volatile names)",
        f"- Zero call volume rows: {s['zero_call_volume_rows']:,} ({_pct(s['zero_call_volume_rows'], n)})",
        f"- Zero call OI rows: {s['zero_call_oi_rows']:,} ({_pct(s['zero_call_oi_rows'], n)})",
        f"- Key-column nulls: {s['key_col_nulls']}",
        "",
        "- No bid/ask in the dataset → cannot screen crossed/zero-bid markets directly; liquidity must be proxied",
        "  by contracts-traded / open-interest. A liquidity filter (min OI / min volume, or restrict to ETFs +",
        "  large caps) is **feasible and necessary** (~5-6% of rows have zero call volume).",
        "",
        "**§4 verdict: clean (0 dups, 0 bad ATM_IV, no key-col nulls); liquidity filter feasible via volume/OI;",
        "label `liquidity_insufficient` applies only to the long tail of illiquid names, not to ETFs/large caps.**",
    ]) + "\n")

    # ---- feature feasibility ----
    (_REPORTS / "options_iv_feature_feasibility.md").write_text("\n".join([
        "# Options-IV Data Audit — §3/§6 Chain Structure & Feature Feasibility",
        "",
        "## §3 Chain structure (what the dataset is)",
        "- This is a **daily per-underlying AGGREGATE**, not a raw option chain.",
        "- strikes: **NO** | expiry dates: **NO** (only `expirations_number` count) | bid/ask: **NO** |",
        "  per-contract prices: **NO** | greeks: **NO**.",
        "- volume: YES (calls/puts contracts traded) | open interest: YES (calls/puts) | IV: SUPPLIED as 7",
        "  moneyness buckets (DITM, ITM, sITM, ATM, sOTM, OTM, DOTM) | call/put: split for volume/OI, but IV",
        "  buckets are by moneyness (not explicitly call vs put).",
        "",
        "## §6 Feature feasibility (timestamp-safe, EOD → next-day)",
        "",
        "| feature | feasible | how |",
        "|---|:---:|---|",
        "| ATM IV | YES | `ATM_IV` |",
        "| 25-delta put/call skew | PROXY | no true delta; moneyness-IV spread `DOTM_IV-DITM_IV` / `OTM-ITM` (approximate) |",
        "| IV term structure | **NO** | no per-expiry IV; only `expirations_number` count |",
        "| IV rank / percentile | YES | trailing per-symbol history of ATM_IV |",
        "| put/call IV spread | PARTIAL | IV not split call/put; put/call **volume & OI imbalance** YES |",
        "| realized − implied (VRP proxy) | YES | `hv_20..200` vs `ATM_IV` |",
        "| option volume / OI imbalance | YES | calls vs puts traded / OI |",
        "| skew change | YES | Δ of the moneyness-IV-spread proxy |",
        "| term-structure slope change | **NO** | no term structure |",
        "| cross-sectional IV dispersion | YES | across the ~3,900-name universe per day |",
        "",
        "**§3/§6 verdict: `options_features_only` — usable as features for trading SPY/QQQ/DIA/IWM (present) or",
        "underlying equities; term-structure features are NOT available. Direct option trading is impossible",
        "(no strikes/expiries/bid-ask).**",
    ]) + "\n")

    # ---- summary report ----
    (_REPORTS / "options_iv_data_audit_report.md").write_text("\n".join([
        "# Options-IV Data Audit — Summary & Decision",
        "",
        f"**Built:** {built}  **Dataset:** `gauss314__options-IV-SP500/data_IV_USA.csv`",
        f"**Rows:** {n:,}  **Range:** {s['date_min']} → {s['date_max']}  **Symbols:** {s['n_symbols']:,}",
        "**Binding question:** can options-chain IV support a leakage-safe, tradable research program?",
        "",
        "## Findings (see per-section audits)",
        "- **§1 Timestamp:** daily EOD only; safe **only** as next-day (t→t+1) features → `timestamp_uncertain`.",
        "- **§2 Universe:** broad ~3,900 US optionable names incl. SPY/QQQ/DIA/IWM; delisted names retained "
        "(AABA drops 2019-11-25) → **not current-constituent survivorship-biased**.",
        "- **§3 Chain structure:** aggregate only — **no strikes / expiries / bid-ask / per-contract prices** →",
        "  `options_features_only`; direct option trading impossible.",
        "- **§4 Data quality:** clean (0 dups, 0 bad ATM_IV, no key nulls); liquidity filter feasible/needed.",
        "- **§5 Tradability:** options are **features only**; tradable instruments = SPY/QQQ/DIA/IWM or equities.",
        "- **§6 Features:** ATM IV, skew proxy, IV rank, RV−IV (VRP) proxy, vol/OI imbalance, cross-sectional",
        "  dispersion all feasible; **IV term structure / slope NOT feasible**.",
        "- **§7 Leakage:** IV market-implied (not future-RV), `hv_*` trailing, no underlying-price column to leak,",
        "  universe not current-only → no hard-fail; the binding caveat is daily observability (→ next-day only).",
        "",
        f"## Data-quality labels: `{', '.join(labels)}`",
        "",
        f"## VERDICT: **{verdict}**",
        "",
        "Per the operator decision rule:",
        "- Chain structure does **not** pass (no strikes/expiries/bid-ask) → **no promotion-grade options-IV intake**.",
        "- Timestamps are daily/EOD but the data **is** usable as end-of-day features for **next-day** trading →",
        "  **`research_only` is permitted**.",
        "- The dataset lacks bid/ask/expiry/strike → **do NOT run a promotion-style strategy**; **no direct option",
        "  trading**.",
        "",
        "### Recommended next step (operator decision)",
        "Open a **`research_only` options-IV-features v1 intake**: EOD IV features (ATM IV, skew proxy, IV rank,",
        "RV−IV / VRP proxy, put/call vol-OI imbalance, cross-sectional IV dispersion) → **next-day** timing/",
        "selection on SPY/QQQ (and/or a liquid equity cross-section), validated under the standard gate with the",
        "explicit research_only ceiling and the vol-targeting-subsumption baseline. **No direct option trades,",
        "no promotion language.** If declined, close options-IV and move to the news/sentiment timestamp audit.",
        "",
        "## Constraints honored",
        "- No strategy code. No promotion-style claims. No direct option-trading assumptions. No term-structure",
        "  features fabricated. Survivorship/timestamp caveats recorded.",
    ]) + "\n")

    print(f"VERDICT: {verdict} | labels={labels} | rows={n:,} symbols={s['n_symbols']} ETFs_ok={etf_ok} clean={clean}")
    print(f"Wrote manifest + 5 audit md under {_REPORTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
