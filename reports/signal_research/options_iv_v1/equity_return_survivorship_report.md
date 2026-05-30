# Equity-Return Audit — §2 Survivorship

- Missing names whose IV data ends mid-window (delisted-in-window): **701**
- Missing names with full IV coverage but absent from HexQuant: **968** (mix of ETFs and renamed/merged tickers).

## Known delisted/merged/failed names (in IV, should exist pre-delist in a survivorship-safe source):

| symbol | in IV | in HexQuant window |
|---|:---:|:---:|
| AABA | True | False |
| TWTR | True | False |
| FB | True | False |
| RTN | True | False |
| CELG | True | False |
| XLNX | True | False |
| MXIM | True | False |
| NLSN | True | False |
| CERN | True | False |
| ZNGA | True | False |
| SIVB | True | False |
| FRC | True | False |
| DISCA | True | False |
| WORK | True | False |
| MGLN | True | False |

**Every probed name that delisted/merged/failed during 2019-2023 (TWTR, FB→META, RTN→RTX, CELG, XLNX,
MXIM, NLSN, CERN, ZNGA, SIVB, FRC, AABA, …) is ABSENT** from HexQuant's in-window slice. HexQuant
carries currently-listed names with backfilled history; names that left the market before ~2025 are
dropped. For a 2019-2023 cross-section this is **textbook survivorship bias** — it would silently
exclude bankruptcies (SIVB, FRC) and merger targets (TWTR, CELG, XLNX), inflating any cross-sectional
result.

**§2 label: `survivorship_biased`.**
