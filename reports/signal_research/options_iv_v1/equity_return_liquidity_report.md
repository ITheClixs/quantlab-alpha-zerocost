# Equity-Return Audit — §4/§6 Liquidity & Data Quality (overlap/survivor set)

Computed on the 2224 overlap (survivor) names — reported for completeness even though the
cross-sectional track is rejected on survivorship grounds.
- Duplicate (symbol, date): 0 | close <= 0: 0 | adj_close nulls: 0
- Median daily dollar volume: $21,077,559
- Tradable on sample date 2021-06-15 (close>=$5, $vol>=$1M): 2019 / 2207 survivor names
- adj_close: adj_close < close for dividend payers (e.g. AAPL) -> dividend+split adjusted (total-return proxy) → total-return computation feasible on covered names.

Liquidity/quality on the survivor set is fine; the binding defect is survivorship (§2), not liquidity.
