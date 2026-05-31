# Sharadar Data Audit — TEMPLATE (no data present)

**Built:** 2026-05-30T10:13:45.744987+00:00
No Sharadar tables found. After acquisition + `ingest_sharadar.py`, this audit will check:

1. **Delisted-name probe** for: TWTR, CELG, XLNX, CERN, ATVI, SIVB, FRC, AABA — found?, permaticker, isdelisted, last price date, delisting/merger action, final-return computable.
2. **CIK-mapping loss** vs the EDGAR 727-company universe (direct `cik` field if present, else a degraded name bridge) — kill criterion: ≥90% mapped.
3. **Window coverage**: 2010-2022 (EDGAR 10-K) and 2019-10-14..2023-07-28 (options-IV).
4. **Corporate actions**: splits, dividends, delistings present.
5. **Return panel**: builds a date×instrument panel and asserts NO delisted name is dropped and NO future-survival filter is applied.
6. **License**: `license_local_research_use` must be operator-confirmed true in the manifest.

Kill criterion (all must pass to justify the purchase): delisted names present; survivorship-safe returns/fields; ticker changes + actions; ≥90% CIK mapping; window coverage; local-research license.

_No purchase, no strategy code. Next external action: §6 feasibility check on a free sample._
