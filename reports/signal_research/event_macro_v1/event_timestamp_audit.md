# Event Timestamp Audit — event_conditioned_macro_v1

**Manifest:** `manifests/event_calendar/event_calendar_manifest.json`  built `2026-05-29T21:50:51.766513+00:00`
**Trading index:** `data/processed/vrp/bars/SPY.parquet` — 4,122 bars, 2010-01-04 → 2026-05-22
**Binding question:** are the event dates ex-ante and timestamp-clean against the trading calendar?

## FOMC (active)

- Total scheduled dates in manifest: **163**  (range 2006-01-31 → 2026-04-29)
- Strictly monotonic: **True**  |  unique: **True**
- Per-year counts 2006-2025 all == 8: **True**
- FOMC dates within data range [2010-01-04..2026-05-22]: **131**
- Of those, on a SPY trading day: **131**  |  misaligned (holiday/gap): **0**
- Provenance: 2006-2018: https://raw.githubusercontent.com/tobiasi/FOMCscrape/master/FOMC_dates.csv (Scheduled==1 End dates); 2019-2026: federalreserve.gov FOMC calendars/historical (scheduled decision dates); exclusions: 2020-03-03 and 2020-03-15 emergency cuts excluded (not ex-ante scheduled)

### Look-ahead controls
- FOMC dates are published ~1 year ahead; conditioning on `days_to_next_fomc` / window flags uses only the ex-ante schedule.
- Emergency 2020 cuts (03-03, 03-15) excluded — not ex-ante scheduled (they would be look-ahead).
- Decision is ~14:00 ET; any daily position is set at the **prior close**, never intrabar.
- `attach_event_features` derives every column from the date + schedule only — no price input, so no future-return leakage is structurally possible.

### Feature coverage on the SPY index

| feature | days flagged |
|---|---:|
| `fomc_t0` | 131 |
| `fomc_tm1` | 131 |
| `fomc_tp1` | 131 |
| `fomc_win2` | 655 |
| `fomc_win5` | 1,441 |
| `is_month_end` | 197 |
| `is_quarter_end` | 66 |
| `in_earnings_season` | 1,470 |

## CPI / NFP (DEFERRED)

- CPI: ALFRED (rid=10) and BLS archive return 403 to automated fetch in this environment; approximating mid-month release would violate the timestamp-clean gate.
- NFP: ALFRED (rid=50) and BLS empsit archive return 403; first-Friday rule has documented exceptions and is not timestamp-clean.
- **Not run in v1.** Approximating release dates (CPI mid-month / NFP first-Friday) would violate the
  timestamp-clean gate; the CPI/CPI-combined variants are deferred until a clean release-date source is secured.

## Earnings-season / Period-end (deterministic)

- earnings_season: broad regime: trading days within the canonical reporting windows (approx Jan 15-Feb 15, Apr 15-May 15, Jul 15-Aug 15, Oct 15-Nov 15)
- period_end: last trading day of each month (month_end) and of each quarter (quarter_end), derived from the SPY/QQQ trading-date index
- Fully deterministic from the trading-date index; no external source, no look-ahead.

## Verdict

- **FOMC: PASS** — 131 clean, trading-day-aligned, ex-ante scheduled events in range.
- **Earnings-season / period-end: PASS** (deterministic).
- **CPI / NFP: DEFERRED** (no timestamp-clean source in this environment).

v1 proceeds on **FOMC + earnings-season + period-end**. Data window 2010-01-04 → 2026-05-22 (~131 FOMC events) — event count is modest; bootstrap CIs will be wide (pre-registered failure mode #5).
