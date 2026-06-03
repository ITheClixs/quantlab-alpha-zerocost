# Fingerprint-VWAP Meta-Labeling v1 — Data Audit Checklist

**Status: PENDING**
The checks below must be executed against the real fetched panel before trusting
any pipeline run. The smoke run was DEFERRED (see note at the bottom).

---

## Critical Rule

> **If any check fails, STOP and write a negative-result note instead of running.**

Do not proceed with `run_fingerprint_vwap_meta` unless every check below is marked
PASS. A single FAIL invalidates any Sharpe or lift figure produced from the affected
panel.

---

## Checklist

### 1. Point-in-Time Universe Membership (No Survivorship Bias)

| Item | How to verify | Status |
|------|--------------|--------|
| Tickers included in the SP500 at the *start* of each sub-period, not only survivors to today | Compare `load_or_fetch_sp500` snapshot date against the earliest bar date; if the snapshot is today's constituent list, the backtest is survivorship-biased | PENDING |
| Tickers added after `--start` must not appear with data before their addition date | Join Wikipedia SP500 constituent history (or a point-in-time vendor source) against bar dates | PENDING |
| Delisted / acquired tickers present in the cache file must not be quietly excluded | Verify the fetch pool produces non-empty parquet for tickers that were in-index but are now gone (yfinance may return stale data or error; both are acceptable if handled) | PENDING |

**Mitigation if full PIT data is unavailable:** restrict the backtest universe to
tickers present in SP500 throughout the entire `--start` to `--end` window
(intersection, not union), accept the reduced universe, and document it.

---

### 2. No Missing or Duplicated Bars on the Trading Calendar

| Item | How to verify | Status |
|------|--------------|--------|
| No duplicate (date, symbol) pairs | `panel.group_by(["date","symbol"]).count().filter(pl.col("count") > 1)` must be empty | PENDING |
| No unexpected trading-calendar gaps (> 5 consecutive business days missing) | Compare each symbol's date series against NYSE calendar; flag symbols with multi-day gaps | PENDING |
| All symbols have sufficient history for the longest window (252 days) | `panel.group_by("symbol").agg(pl.count()).filter(pl.col("count") < 252 + horizon)` must be empty or those symbols dropped | PENDING |

---

### 3. Fingerprint Features Are Right-Anchored (No Look-Ahead)

| Item | How to verify | Status |
|------|--------------|--------|
| Rolling windows in `build_fingerprint_features` use only `t` and earlier | Unit tests in `tests/signal_research/fingerprint_vwap/test_fingerprint.py` already enforce this: first W-1 rows null | PASS (covered by unit tests) |
| VWAP proxy in `daily_vwap_proxy` uses only same-day or prior-day columns | Confirmed by code inspection: `vwap_proxy = (close * volume).rolling_sum / volume.rolling_sum`, fully backward-looking | PASS (code inspection) |
| No `.shift(-n)` with n>0 appears in fingerprint or vwap modules | `grep -n "shift(-" src/quant_research_stack/signal_research/fingerprint_vwap/` returns empty | PENDING (re-verify on each module change) |

---

### 4. Labels Timestamped After Entry

| Item | How to verify | Status |
|------|--------------|--------|
| Triple-barrier label references `close[t + horizon]` not `close[t]` | Confirmed in `meta_label_walk_forward._feature_frame`: `pl.col("close").shift(-vertical_barrier_days)` | PASS (code inspection) |
| Forward-return column `future_return_horizon` is excluded from feature matrix | `_feature_frame` explicitly lists `_FEATURE_COLUMNS` and `extra_feature_columns`; `future_return_horizon` is not in either | PASS (code inspection) |
| Walk-forward folds never allow test-period labels to appear in train set | `purge_days` gap enforced between `train_end_idx` and `test_start_idx` in `train_meta_label_walk_forward` | PASS (code inspection) |

---

### 5. Corporate-Action Adjustment Consistency

| Item | How to verify | Status |
|------|--------------|--------|
| Prices are split- and dividend-adjusted throughout | `fetch_one_ticker` uses `auto_adjust=True` via `LongHistoryConfig`; verify this flag is set in `long_history.py` | PENDING |
| Adjustment is consistent between cached and freshly fetched bars | If a cached parquet was written before a split, its prices are stale; cache invalidation policy needed | PENDING |
| Volume is adjusted for splits (shares-outstanding scale) | yfinance `auto_adjust=True` also adjusts volume; confirm or set `back_adjust=True` explicitly | PENDING |

---

### 6. Cross-Symbol Consistency

| Item | How to verify | Status |
|------|--------------|--------|
| All symbols share the same trading calendar (no crypto vs equity mixing) | All tickers sourced from SP500 list — equity-only by construction; confirm no ETF or trust tickers that trade after-hours differently | PENDING |
| No NaN-filled forward or backward carries inserted by the normalizer | `_normalize_one` calls `drop_nulls()` — no implicit fill; confirm PENDING if any upstream source changed | PENDING |

---

## Smoke Run Status

**EXECUTED** (2026-06-03)

```bash
PYTHONPATH=src uv run python scripts/run_fingerprint_vwap_meta_backtest.py \
    --top-n 5 --start 2018-01-01 --end 2024-12-31 \
    --out reports/signal_research/fingerprint_vwap_meta_v1/smoke
```

Results recorded:

| Item | Value |
|------|-------|
| Panel shape | 8800 rows × 5 symbols × 1760 dates |
| SP500 constituent list | Today's Wikipedia snapshot (NOT point-in-time) — **survivorship bias present** |
| Selected tickers | AAPL, AMZN, AMD, GOOGL, GOOG |
| Pipeline status | `evaluated` (primary eligible) |
| Primary net Sharpe | 1.117 (4869 events) |
| Meta net Sharpe | 0.929 |
| Baseline net Sharpe | 1.117 |
| Lift | −0.189 (below 0.20 threshold) |
| Deflated Sharpe probability | 0.343 (below 0.95 threshold) |
| Verdict | **DO_NOT_ADVANCE** — failed gates: `deflated_sharpe`, `lift` |
| Console warnings | None |

**CONCERNS from this smoke run:**

1. **Survivorship bias** — the SP500 constituent list is today's snapshot.
   Checks 1a–1c in the checklist above are FAIL until a PIT source is used.
   The Sharpe figures above are NOT trustworthy for a strategy claim.

2. **Meta-labeler reduces Sharpe vs. take-every-entry** (−0.189 lift).
   The meta-filter is not adding value on this tiny 5-ticker universe;
   a full `--top-n 30` run on a PIT universe is required before any conclusion.

3. **Deflated Sharpe probability 0.343 << 0.95** with 50 trials.
   Even at face value the out-of-sample Sharpe does not survive multiple-testing
   correction.

The committed deliverable (script + audit doc) is complete. The real gated run
on a PIT universe with `--top-n 30` is a follow-up required before the
fingerprint-VWAP signal can advance.

---

## References

- `src/quant_research_stack/signal_research/fingerprint_vwap/pipeline.py` — pipeline entry point
- `src/quant_research_stack/signal_research/training/meta_label_walk_forward.py` — walk-forward trainer; `net_return` column definition at line 191
- `src/quant_research_stack/signal_research/data/long_history.py` — bar fetcher; verify `auto_adjust` flag
- `tests/signal_research/fingerprint_vwap/` — unit tests for look-ahead checks
- CLAUDE.md §1 rule 2: "Do not use future data in features."
- CLAUDE.md §1 rule 3: "Do not fit scalers, imputers, encoders, or normalizers on validation or test data."
