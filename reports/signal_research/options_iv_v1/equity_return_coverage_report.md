# Equity-Return Audit — §1 Coverage

**Source:** HexQuant__Stocks-Daily-Price | **IV window:** 2019-10-14 → 2023-07-28
- HexQuant symbols (all / in-window): 6320 / 5087
- IV universe symbols: 3893
- **Overlap (mapped): 2224 (57.1%)** | **Missing: 1669 (42.9%)**
- Missing with special chars (ticker-format issue): **0** → mapping is NOT the cause.
- Index ETFs in HexQuant: SPY=False, QQQ=False, DIA=False, IWM=False → **HexQuant is stocks-only; ETF returns must come from the clean SPY/QQQ bars (secondary track).**

Coverage is only 57% of the IV universe; the gap is survivorship + ETFs, not ticker formatting.
