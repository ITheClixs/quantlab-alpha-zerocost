# Strategy Intake Protocol

Every new strategy proposal must complete this intake **before** it consumes
compute on the validation pipeline. The intake is the contract between
the proposer and the validation infrastructure. It enforces the discipline
that emerged from the six-iteration price/volume search.

## Why this exists

After six independent search iterations on OHLCV-derived signals — three
on US large-cap equity (top-50, top-100, top-200 SP500), one on
sector-conditioned US equity, one full LightGBM scaleup, and one
independent crypto top-30 run — **no model variant from the M3 family
passes the promotion gates at hedge-fund-grade costs**. The family
ordering reproduces across orthogonal microstructures. The 12-1 momentum
signal produces a +0.15-0.59 holdout Sharpe in every iteration; PSR_zero
on that number is 0.55, and PBO/DSR multi-test correction kills it
every time.

The conclusion: **the constraint is the information set, not the universe,
not the model class, not the costs in isolation.** Future research must
either consume a fundamentally new information channel or accept that it
is producing a noise-floor result.

## The intake form

Fill this template before any backtest run. Save to
`docs/research/intake/YYYY-MM-DD-<strategy-name>.md` and link it from the
proposed `ValidationSpec.intake_doc_ref`.

### 1. Strategy name and one-line description

A canonical kebab-case name used throughout the pipeline. One sentence
that another engineer can read in five seconds.

### 2. Hypothesis statement (one paragraph)

The economic or behavioral mechanism that creates predictable returns.
Not "X predicts Y" — explain **why** X should predict Y. A vague
hypothesis like "machine learning on price/volume" is a red flag; the
GKX scaleup result documents the cost of that vagueness.

Acceptable examples:
- "Implied volatility minus realized volatility prices a variance risk
  premium; selling that premium on liquid index options has historically
  earned a positive return because long-vol hedgers pay a structural
  premium during normal regimes (Bondarenko 2014)."
- "Negative news sentiment in earnings transcripts correlates with
  forward 60-day relative-strength reversals among large-cap names,
  because slow-moving institutional sentiment updates create lag-driven
  pricing dislocations (Garcia 2013, Tetlock 2007)."

Unacceptable:
- "LightGBM on standard price features should find non-linear patterns."
- "Mean reversion has historically worked."
- "Add a transformer to capture sequence dependencies."

### 3. Information source declaration (required, machine-readable)

Declare every information channel the strategy consumes, from the
`InformationSource` enum:

- `ohlcv` — price/volume bars
- `options_implied_vol` — IV surfaces, VIX-family, VXN, VVIX, SKEW
- `options_volume` — option-trade volume, put/call ratios
- `sentiment_news` — wire/transcript text
- `sentiment_social` — social media text
- `earnings_fundamentals` — financial statements, earnings revisions
- `macro_rates` — interest rates, yield curves
- `macro_fx` — foreign exchange
- `macro_commodity` — energy, metals, agriculture
- `microstructure_tick` — trade-level prints
- `microstructure_book` — limit-order book
- `cross_asset` — equity-bond, equity-FX cross signals
- `event_window` — FOMC, CPI, earnings, expiration dates
- `alternative` — satellite, web-scraped, etc.

**Rule (enforced in code):** strategies declaring **only** `ohlcv` cannot
reach `promotion_eligible` status regardless of metrics. This is the
"no promotion without new information source" rule. The 6-iteration
search demonstrated that OHLCV-only mechanical strategies do not pass
honest validation; we will not waste capital deployment review on them
unless an operator overrides explicitly.

### 4. Why is this not just an OHLCV restatement?

If the only declared source is `ohlcv`, the proposal must explain why
this strategy will succeed where six prior iterations failed. The bar
is **substantive, not procedural** — adding more trees to a LightGBM
or a wider parameter sweep is not a new information channel.

If a non-OHLCV source is declared, this section is replaced by:
- Provenance and timestamp integrity of the new source (no look-ahead?)
- License terms and cost
- Coverage (date range, universe)
- Known data-quality limitations

### 5. Expected gross Sharpe and capacity

Before running the backtest, state the expected gross Sharpe and the
expected capacity (in USD AUM, with the assumed implementation friction).
A proposal that "expects to find" alpha without an ex-ante magnitude is
a fishing expedition; the pre-registration of an expected number anchors
the post-hoc result.

### 6. Cost assumptions

State commission and spread in bps one-way. The validation pipeline will
apply your declared numbers, plus a 2× stress, plus a 1-bar delay stress.
**A strategy that requires lower-than-hedge-fund-grade costs is not a
hedge-fund strategy.** Document why your costs are what they are.

### 7. Universe and history

- Universe definition (and survivorship status)
- Start/end dates
- Dev/holdout split (the holdout window must extend at least 18 months
  after dev_end; the pipeline enforces dev-only-guard)
- Minimum-size constraints

### 8. What would make this fail?

State three specific failure modes the proposer expects. This is a
falsification check — a proposer who cannot articulate how the strategy
fails has not thought about it. Pre-register the failure modes; if all
three fail to materialize, the result is more credible.

Examples for the VRP intake:
- "Realized vol may spike above implied during crisis periods, causing
  short-vol losses concentrated in 2-3 events. Concentration diagnostic
  should catch this."
- "VRP may be priced in by 2015+ index option market makers; expect dev
  Sharpe to be lower in the 2018+ subsample."
- "Term-structure features are highly correlated; PBO should be modest
  but DSR after multi-test should still penalize the variant grid."

### 9. Promotion intent

State whether the strategy is intended for:
- `research_only` — produce knowledge, no capital deployment intent
- `paper_trade_after_pass` — paper trade if it survives, no live capital
- `live_capital_after_pass` — live capital deployment if it survives
  (this triggers the §11 promotion gate review in CLAUDE.md)

### 10. Sign-off

Proposer name, intake date, and a one-sentence acknowledgement that the
strategy will be subjected to the 8-criteria validation gate with no
post-hoc tuning permitted after the holdout pass.

## What happens after intake

1. The intake document is committed to `docs/research/intake/`.
2. A `ValidationSpec` is constructed referring to the intake doc by path.
3. The signal generation function is implemented in
   `src/quant_research_stack/signal_research/strategies/<strategy_name>.py`.
4. The CLI driver is added to `scripts/validate_<strategy_name>.py`.
5. `validate_strategy()` runs the full pipeline.
6. `render_pipeline_report()` produces the standard report.
7. The 8-criteria gate is applied; failure classes are recorded.
8. The status assignment (NONE / RESEARCH_PASS / PROMOTION_ELIGIBLE) is
   committed alongside the report.

## What does not happen after intake

- No re-tuning of parameters after holdout numbers are seen.
- No swapping of dev/holdout windows to "make it work".
- No expanding the variant grid in response to a near-miss.
- No selective reporting of the best variant; the full grid stays in
  the PBO pool.
- No promotion of OHLCV-only strategies regardless of metrics.

## Permitted iterations

If a strategy fails, the proposer may:
- Document the failure classes and update the intake doc.
- Propose a **structurally different** strategy with a new intake doc.
- Propose a strategy in a **new information channel** with a new intake.

Not permitted:
- "Re-run with different parameters and see if it works this time."
- "Try a deeper model on the same OHLCV features."
- "Use a different label horizon and re-test the holdout."

## References

- Bailey & López de Prado (2014). The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting, and Non-Normality.
- López de Prado (2018). Advances in Financial Machine Learning, ch. 3, 7, 11.
- Politis & Romano (1994). The Stationary Bootstrap.
- Gu, Kelly & Xiu (2020). Empirical Asset Pricing via Machine Learning.
- Avellaneda & Lee (2010). Statistical Arbitrage in the US Equities Market.
- Bondarenko (2014). Variance Trading and Market Price of Variance Risk.
