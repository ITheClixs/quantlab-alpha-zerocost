# Options-IV Data Audit — §2 Universe & Survivorship

**Unique symbols:** 3,893 (far broader than S&P 500 — a broad US optionable universe).

Per-year unique symbols / rows:

| year | symbols | rows |
|---|---:|---:|
| 2019 | 3,827 | 171,263 |
| 2020 | 3,836 | 879,765 |
| 2021 | 3,644 | 858,437 |
| 2022 | 3,471 | 816,685 |
| 2023 | 3,254 | 435,511 |

- Index ETFs present (directly tradable as the instrument): SPY=932 rows, QQQ=935 rows, DIA=929 rows, IWM=937 rows  → all present: True.
- Delisted-name probe: AABA has 29 rows, last 2019-11-25 (< dataset end 2023-07-28) → **delisted names are retained**
  and drop out at delisting → NOT current-constituent survivorship-biased: True.
- Caveat: completeness of *additions* over time is unverifiable; declining yearly symbol counts (3,827→3,254) are consistent with attrition, not a current-only snapshot.

**§2 verdict: not current-constituent survivorship-biased; PIT-plausible within 2019-2023. Promotion
language is blocked anyway by the chain-structure gaps (§3).**
