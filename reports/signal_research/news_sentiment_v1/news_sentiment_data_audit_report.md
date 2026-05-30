# News/Sentiment Data Audit — Summary & Decision

**Built:** 2026-05-30T08:47:07.653029+00:00
**Binding question:** is there a timestamp-clean, look-ahead-safe news/sentiment signal feed?

## Candidate comparison

| candidate | timestamp | survivorship | look-ahead | freq | labels |
|---|---|---|---|---|---|
| **EDGAR 10-K** | filing date (PIT) ✓ | aware (727 cos) ✓ | none ✓ | annual | fwd returns incl. |
| earnings transcripts | earnings_date ✓ | **biased** (496 cur. cos) ✗ | fwd-EPS fields | quarterly | — |
| benstaf LLM | daily | narrow (84) ✗ | **LLM hindsight** ✗✗ | daily | — |

Training-only (no timestamp/ticker; excluded as feeds): FinGPT-sentiment, financial_phrasebank, Neil0930 —
usable only to TRAIN a sentiment classifier, not as a PIT signal.

## VERDICT: **PASS (EDGAR 10-K) — open a research v1 intake on the cleanest candidate**

- **Cleanest candidate: `edgar_10k` (EDGAR 10-K).** It is the first dataset in the whole program that
  is BOTH timestamp-clean (SEC filing date = true public-availability) AND survivorship-aware (727
  historical constituents incl. delisted names), with full 10-K item text and forward-return labels.
- Earnings transcripts: clean timestamp but **survivorship-biased universe** + forward-EPS fields →
  `research_only`, cross-sectional invalid (single-name/time-series use less affected).
- benstaf LLM-sentiment: **rejected** as a leak-safe signal (LLM hindsight + undocumented availability).

### Recommended next step
Open a **research v1 intake on EDGAR 10-K**: 10-K text features (risk-factor / MD&A tone, year-over-year
10-K text change) keyed to the **filing date**, traded **t+1**, with forward-return labels used as labels
only. Cross-sectional across historical constituents is valid here (survivorship-aware). It is annual /
low-frequency and long-horizon — pre-register that as a constraint. This is the strongest data the
program has found; it still faces the standard validation gate and the cost/subsumption walls.

## Constraints honored
- No strategy code. No silent dropping. Forward returns / forward-EPS flagged labels-only. LLM
  look-ahead investigated and the LLM feed rejected. Training-only sets excluded as feeds.
