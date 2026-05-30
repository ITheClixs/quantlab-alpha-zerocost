# News/Sentiment Audit — Universe & Survivorship

## EDGAR 10-K
- 727 unique companies (CIK), 6282 filings, 2010-01-22 → 2022-12-21 (annual).
- README: 'all SP500 **historical** constituents'. Delisted/merged names present: TWITTER=True, CELGENE=True, XILINX=True, CERNER=True, ACTIVISION=True, MAXIM=True → **survivorship-aware: True** (FRC/SIVB failed in 2023, outside the 2010-2022 window).

## Earnings transcripts
- 496 companies (from the **current** Wikipedia S&P 500 list), 20681 transcripts.
- Delisted-name presence: TWITTER=False, CELGENE=False, XILINX=False, CERNER=False, ACTIVISION=False, MAXIM=False → all absent → **survivorship-biased: True** (current-constituent universe).

## benstaf
- 84 tickers (curated mega-cap/NASDAQ universe), 105588 rows, 2019-01-02 → 2023-12-28 → narrow + survivorship-prone.
