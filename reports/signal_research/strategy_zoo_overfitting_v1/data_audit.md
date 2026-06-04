# Data Audit — Strategy Zoo Overfitting v1

**Status: PENDING** (update each check to PASS / FAIL after executing against the fetched panel)

> Rule: If ANY check below is marked FAIL, STOP. Do not trust downstream results.
> Write a negative note explaining what failed and halt the analysis.

---

## Scope

- **Universes:** All 11 `UNIVERSES` from `strategy_benchmark.data`
  (ES_F, NQ_F, SPY, QQQ, EW_BASKET, IWM, DIA, XLK, XLF, XLE, EW_SECTORS)
- **Underlying tickers (9):** DIA, ES=F, IWM, NQ=F, QQQ, SPY, XLE, XLF, XLK
- **Date range:** configured by `--start` / `--end` CLI flags
- **Cache location:** `data/processed/strategy_zoo_overfitting_v1/`

---

## Checklist

### 1. Row completeness — no missing (date, symbol) pairs

**Check:** For each ticker, count trading days in the date range using a
market-calendar reference (or compare to SPY row count as proxy).
No ticker should be missing more than 5 consecutive business days
except around known market closures.

**Method:**
```python
import polars as pl
from pathlib import Path
cache = Path("data/processed/strategy_zoo_overfitting_v1")
spy = pl.read_parquet(cache / "SPY.parquet")
spy_dates = set(spy["date"].to_list())
for p in sorted(cache.glob("*.parquet")):
    df = pl.read_parquet(p)
    missing = spy_dates - set(df["date"].to_list())
    print(p.stem, "missing", len(missing), "dates vs SPY")
```

**Status: PENDING**

---

### 2. No duplicate (date, symbol) rows

**Check:** Each (date, symbol) pair must appear exactly once per ticker file.

**Method:**
```python
for p in sorted(cache.glob("*.parquet")):
    df = pl.read_parquet(p)
    dups = df.filter(df.select(pl.struct("date","symbol")).is_duplicated())
    assert dups.height == 0, f"{p.stem}: {dups.height} duplicates"
```

**Status: PENDING**

---

### 3. Adjusted-close consistency (no split/dividend discontinuities)

**Check:** Day-over-day log return magnitude should not exceed ±50% on any
single day (larger gaps almost always signal a data-error or unadjusted split,
not a true market move). Futures (ES=F, NQ=F) are exempt from the dividend
adjustment check but are included in the gap check.

**Method:**
```python
import numpy as np
for p in sorted(cache.glob("*.parquet")):
    df = pl.read_parquet(p).sort("date")
    c = df["close"].to_numpy()
    lr = np.log(c[1:] / c[:-1])
    bad = np.abs(lr) > 0.50
    if bad.any():
        dates = df["date"].to_list()
        print(p.stem, "extreme moves on:", [dates[i+1] for i in np.where(bad)[0]])
```

**Status: PENDING**

---

### 4. IS / OOS purge-and-embargo gap is respected

**Check:** With `oos_fraction=0.3` and `embargo_days=10`, the last IS bar
must be at least 10 trading days before the first OOS bar.

**Verification method:**
```python
T = len(all_dates)
split = int(T * 0.70)
last_is_date  = all_dates[split - 1]
first_oos_date = all_dates[min(split + 10, T - 1)]
gap = (first_oos_date - last_is_date).days
assert gap >= 10, f"embargo gap too small: {gap} days"
```

**Status: PENDING**

---

### 5. No future-data leakage in signal construction

**Check:** Every signal family in `SIGNAL_FAMILIES` uses only
`close.shift(1)` (or equivalent shifted versions) as input to its
threshold comparisons. Rolling windows are computed on the period
ending at `t-1` relative to the trade entry at bar `t`.

**Method:** Manual inspection of `src/quant_research_stack/strategy_benchmark/signals.py`
and `zoo/transforms.py`. Confirm that no signal generator reads
`close[t]` to decide the position held at bar `t`.

**Status: PENDING**

---

### 6. Survivorship-bias caveat (PIT note for ETF universes)

**WARNING — known limitation (not a blocker, but must be documented):**

The universe tickers (SPY, QQQ, IWM, DIA, XLK, XLF, XLE) are **today's
tickers**, fetched at today's date. This means:

- The ETF tickers themselves are fixed (they are indices/funds, not
  individual stocks), so there is no survivorship bias at the
  *instrument* level.
- However, the **internal composition** of each ETF (e.g., sector weights
  in XLK) reflects today's constituents, not the constituents at the time
  of each historical bar. The ETF's adjusted-close price series does embed
  the historical composition implicitly via dividends and price, so
  daily return attribution is correct.
- **Point-in-time caveat:** Any feature that depends on the ETF's live
  constituent list (e.g., sector exposure, factor loadings) would be
  look-ahead biased. This study uses only OHLCV-derived signals, so
  this limitation does not affect the current results.
- **Conclusion:** Acceptable for an overfitting-demonstration study on
  liquid single-instrument ETFs/futures. For live-signal production with
  basket construction, switch to a PIT constituent list.

**Status: DOCUMENTED — not a blocker for this study**

---

## Summary table (fill in after execution)

| Check | Status | Notes |
|-------|--------|-------|
| 1. No missing bars | PENDING | |
| 2. No duplicate rows | PENDING | |
| 3. Adj-close consistency | PENDING | |
| 4. IS/OOS embargo gap | PENDING | |
| 5. No look-ahead leakage | PENDING | |
| 6. Survivorship caveat | DOCUMENTED | ETF tickers stable; PIT caveat noted |

---

## How to mark checks PASS

After running the script and inspecting cache output, update each row above
to `PASS` with a brief note (e.g., "9 tickers, 0 duplicates, max gap 2 days").
If any check returns `FAIL`, stop all downstream analysis and open an issue.
