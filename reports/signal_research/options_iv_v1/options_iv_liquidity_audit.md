# Options-IV Data Audit — §4 Data Quality & Liquidity

- Duplicate (symbol, date): **0**
- ATM_IV nulls: 0 | non-positive: 0 | >300 (impossible): 0
- ATM_IV quantiles (1/50/99): [7.96, 38.22, 122.75] (vol points; broad universe incl. small/volatile names)
- Zero call volume rows: 179,937 (5.69%)
- Zero call OI rows: 99,897 (3.16%)
- Key-column nulls: {'ATM_IV': 0, 'DOTM_IV': 0, 'DITM_IV': 0, 'hv_20': 0, 'VIX': 0, 'calls_open_interest': 0, 'expirations_number': 0}

- No bid/ask in the dataset → cannot screen crossed/zero-bid markets directly; liquidity must be proxied
  by contracts-traded / open-interest. A liquidity filter (min OI / min volume, or restrict to ETFs +
  large caps) is **feasible and necessary** (~5-6% of rows have zero call volume).

**§4 verdict: clean (0 dups, 0 bad ATM_IV, no key-col nulls); liquidity filter feasible via volume/OI;
label `liquidity_insufficient` applies only to the long tail of illiquid names, not to ETFs/large caps.**
