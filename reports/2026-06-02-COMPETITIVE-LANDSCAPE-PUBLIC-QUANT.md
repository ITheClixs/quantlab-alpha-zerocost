# Competitive Landscape — Public & Published Quantitative Systems vs. QuantLab Alpha

**Date:** 2026-06-02
**Status:** Research report / external corroboration of the program thesis.
**Author:** QuantLab research
**Scope:** Surveys the public quant ecosystem (GitHub frameworks, open platforms,
crowdsourced funds, and the academic record) and tests the operator's hypothesis —
*"successful public systems exist and are beating us."* It does not.
**Companion documents:** [`reports/2026-05-30-PROGRAM-REVIEW-DATA-CONSTRAINT.md`](2026-05-30-PROGRAM-REVIEW-DATA-CONSTRAINT.md),
repository `README.md` §6.

---

## 0. Executive summary

The operator asked a reasonable question: *are there public repositories or published
systems that achieve what QuantLab could not — deployable, promotable, hedge-fund-grade
alpha — and if so, why are they better?* After a multi-source survey, the answer is:

1. **No public artifact contains a deployable, costed, capacity-aware, gate-surviving
   alpha.** The "successful" public projects are **infrastructure** (backtest/execution
   engines, research pipelines), **education** (strategy cookbooks), or **unaudited
   self-reported metrics** (the 2025–2026 LLM-agent wave). None publishes what
   `CLAUDE.md` §13 demands.
2. **The genuine winners win on axes a free-data solo operator structurally cannot
   replicate:** private/entitled data, execution and transaction-cost infrastructure,
   capacity-constrained niches, portfolio construction across many *decayed* signals,
   and incentive design — **not a better learner.**
3. **QuantLab's "0 deployable" is the correct, honest output of a stricter standard,**
   not evidence of being behind. The external literature reproduces the program's two
   walls (cost, subsumption) and four-wall taxonomy almost verbatim.

This *strengthens* the program thesis: **the binding constraint is the information set,
not the method.**

---

## 1. What is actually public — a four-bucket taxonomy

A GitHub/web survey (stars as of 2026-06) sorts the entire visible ecosystem into four
buckets. Crucially, **stars measure usefulness-as-infrastructure, not profitability.**

| Bucket | Representative repos | What it actually is | Ships deployable alpha? |
|---|---|---|---|
| **A. Backtest / execution engines** | [NautilusTrader](https://nautilustrader.io/), backtrader (~22k★), Zipline / `zipline-reloaded` (~20k★), `ricequant/rqalpha`, `fasiondog/hikyuu`, `vnpy`/`51bitquant/howtrader`, Superalgos, Freqtrade, Hummingbot | Infrastructure to run *your* idea | **No** — engines, not edges |
| **B. Research platforms** | [microsoft/qlib](https://github.com/microsoft/qlib) (+ `RD-Agent`) | ML pipeline + reproducible benchmarks | **No** — pipeline; benchmarks report tiny ICs, not net PnL |
| **C. Education / cookbooks** | `paperswithbacktest/awesome-systematic-trading`, `wangzhe3224/awesome-systematic-trading`, `wilsonfreitas/awesome-quant`, `je-suis-tm/quant-trading`, `EliteQuant` | RSI/MACD/Bollinger/pairs textbook code | **No** — the "subsumed/noise" bucket already closed by QuantLab |
| **D. AI/LLM-agent wave (2025–26)** | `microsoft/RD-Agent`, `QuantaAlpha`, `AgentQuant`, `AutoHypothesis`, "DeepAlpha", "Orallexa", "Vibe-Trading" | Agentic factor mining; impressive READMEs | **Unverified** — in-sample / self-reported; no audited live track record |

**The decisive filter:** *not one* public project publishes a costed, turnover-aware,
purged-and-embargoed walk-forward holdout gated by Probability of Backtest Overfitting
(PBO) and a Deflated Sharpe Ratio (DSR) with a reproducible audit log. They look
successful precisely because they **omit the test that would reveal they are not.**

### 1.1 The most instructive case: Microsoft Qlib

Qlib (Yang et al., 2020; [github.com/microsoft/qlib](https://github.com/microsoft/qlib))
is the strongest public artifact — a genuine, AI-oriented research platform with reproducible,
20-seed-averaged benchmarks on China A-share CSI300 (Alpha158/Alpha360 feature sets).
But its headline numbers are **Information Coefficients of ≈ 0.03–0.05** — a *ranking
correlation*, **not net-of-cost tradable PnL**. Run through QuantLab's cost/delay/PBO
gate, Qlib's best published model would look exactly like a QuantLab closed channel:
a faint, real, sub-gate signal. Qlib hands you a rigorous *pipeline*; it does not hand
you deployable alpha. This is corroboration, not competition.

---

## 2. Why the genuine winners win (it is not the algorithm)

Four well-documented mechanisms explain real, durable edge. Each is something a
free-data solo operator **cannot** acquire — and each maps onto a QuantLab wall.

### 2.1 Alpha is private by construction; publication destroys it

McLean & Pontiff (2016), tracked across 97 cross-sectional predictors: returns fall
**~26% out-of-sample** (statistical bias / overfitting) and **~58% post-publication**
(an additional ~32% from investor learning), with the **liquid, easy-to-arbitrage**
anomalies — i.e. *anything tradable from free OHLCV* — decaying **most**
(McLean & Pontiff, 2016). The surviving edge concentrates in illiquid,
expensive-to-arbitrage corners (Shleifer & Vishny, 1997) — exactly where taker costs
kill a retail book. **→ QuantLab cost wall + subsumption wall.**

### 2.2 Public backtests lie in three structural ways

Survivorship bias, look-ahead bias, and overfitting all inflate returns in the same
direction — easily 2× ([Tessera Alpha](https://tesseraalpha.com/methodology/backtesting-survivorship-lookahead)).
Bailey & López de Prado: with only **~45 strategy variants on 5 years of daily data,
P(best backtest is overfit) > 50%** ([Quant Alpha](https://quantalpha.co/en/blog/backtest-overfitting-and-live-performance)).
QuantLab's PBO/DSR gate exists *specifically* to catch this and demonstrably did — it
caught the `mid_direction_up` future-label leak and a falsy-zero PBO classifier bug.
The flashy public repos run none of these controls.

### 2.3 Edge is 90% execution / data / risk — Quantopian proved it at scale

Quantopian gave hundreds of thousands of quants free survivorship-bias-free minute data
and a licensing payout. It **still failed to find scalable alpha** and shut its fund.
Post-mortem, verbatim: *"the idea is only 10% of a successful quant strategy; the other
90% is execution, risk management, portfolio construction, and transaction-cost
analysis"* ([BrokersDB](https://brokersdb.com/learn/quantopian-history-legacy-review)).
It also noted retail can exploit micro-cap inefficiencies a large fund cannot — but
those do not *scale* (**capacity wall**). Its open-source legacy (Zipline, Alphalens,
Pyfolio) survives; its alpha did not.

### 2.4 Pod shops do not even need un-decayed alpha

Goldman prime brokerage clocked systematic equity managers **−4.2%** in summer 2025,
yet Millennium / Citadel / Point72 finished the year double-digits positive
([Young & Calculated](https://youngandcalculated.substack.com/p/factor-decay-is-real-how-published)).
They knowingly trade *decayed* factors because a low-Sharpe sleeve still earns its place
inside a market-neutral, hundred-signal, leverage-and-risk-managed book. **The unit of
success was never a single strategy — it is the diversified, risk-managed portfolio.**

### 2.5 The crowdsourced model that worked fixed *incentives*, not algorithms

Numerai succeeded where Quantopian died by (a) distributing obfuscated,
point-in-time-clean data so everyone shares one leakage-safe panel, and (b) forcing
**skin-in-the-game staking** so overfit models lose the modeler's own capital
([BrokersDB](https://brokersdb.com/learn/quantopian-history-legacy-review)). Mechanism
design, not a better learner.

### 2.6 LLMs are accelerating decay

When many participants prompt similar models on similar public data, signals homogenize
and decay faster. The 2025 *AlphaAgent* paper warns LLM alpha-mining produces
"homogeneous factors that worsen crowding and accelerate decay"
([IBKR Quant](https://www.interactivebrokers.com/campus/ibkr-quant-news/llms-and-the-shortening-shelf-life-of-copyable-alpha/);
[arXiv 2605.23905](https://arxiv.org/html/2605.23905)). Copyable, public-data alpha now
has a *shorter* half-life than ever — directly devaluing the bucket-D LLM-agent wave.

---

## 3. Mapping the external record onto the QuantLab walls

| QuantLab finding (repo + program review) | External corroboration |
|---|---|
| **Cost wall** — microstructure signals real but die on taker cost | Surviving public alpha lives in illiquid, expensive-to-arbitrage names (McLean–Pontiff); cost-aware backtesting flips most gross-positive HFT to net-negative (Quant Alpha) |
| **Subsumption wall** — index risk-timing captured by vol-targeting | Liquid, easily-described signals decay fastest; durable edge is cross-sectional/diversifiable (McLean–Pontiff; pod-shop practice) |
| **Data-access / survivorship wall** | Quantopian + pod shops: edge = data + execution + portfolio construction, not idea. Free OHLCV is the most-arbitraged data there is |
| **Frequency / sample wall** | Minimum detectable IC scales as $1.96/\sqrt{N}$; low-frequency free panels cannot resolve the small IC real alpha carries (see README §6 figure) |
| **0 deployable after disciplined gating** | The correct output of applying PBO/DSR/cost/placebo gates that public repos skip — a sign the harness works |

---

## 4. Conclusion and recommendations

The operator's premise — *"public systems are beating us"* — does not survive contact
with the evidence. **QuantLab is more rigorous than essentially every public project it
would be compared against.** The public "successes" are either infrastructure to reuse
(Qlib, Nautilus), or metrics that would not survive QuantLab's own gates.

Consistent with the May-30 program review:

1. **Treat the validation stack as the durable, open-sourceable asset** (program-review
   Option C is not a failure). Qlib and Nautilus are public proof that *infrastructure*,
   not strategy, is the shareable win.
2. **If capital is spent, spend it on the one channel that beats both walls — futures-
   curve carry** (cross-sectional, diversifiable, documented premium), with a
   pre-committed kill criterion (program-review Option A, §7).
3. **Stop optimizing for a single promotable strategy.** The winning unit is a
   risk-managed *portfolio* of several modest, decayed-but-diversifying sleeves — a
   different S4 design goal.
4. **The only moat is the winners' moat:** differentiated/entitled data + execution
   realism + portfolio construction — never a cleverer model on the same free OHLCV
   everyone (and every LLM) is mining.

---

## 5. References

### 5.1 Primary (peer-reviewed)

- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*, 20(4), 39–69. https://doi.org/10.21314/JCF.2016.322
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*, 40(5), 94–107. https://doi.org/10.3905/jpm.2014.40.5.094
- Berk, J. B., & Green, R. C. (2004). Mutual Fund Flows and Performance in Rational Markets. *Journal of Political Economy*, 112(6), 1269–1295. https://doi.org/10.1086/424739
- Grossman, S. J., & Stiglitz, J. E. (1980). On the Impossibility of Informationally Efficient Markets. *American Economic Review*, 70(3), 393–408.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5–68. https://doi.org/10.1093/rfs/hhv059
- Lo, A. W. (2004). The Adaptive Markets Hypothesis. *Journal of Portfolio Management*, 30(5), 15–29.
- McLean, R. D., & Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? *Journal of Finance*, 71(1), 5–32. https://doi.org/10.1111/jofi.12365
- Shleifer, A., & Vishny, R. W. (1997). The Limits of Arbitrage. *Journal of Finance*, 52(1), 35–55. https://doi.org/10.1111/j.1540-6261.1997.tb03807.x
- Yang, X., Liu, W., Zhou, D., Bian, J., & Liu, T.-Y. (2020). Qlib: An AI-oriented Quantitative Investment Platform. *arXiv:2009.11189*. https://arxiv.org/abs/2009.11189

### 5.2 Secondary & industry (non-peer-reviewed; context only)

- AI-driven alpha decay — [arXiv:2605.23905](https://arxiv.org/html/2605.23905) (2026); [IBKR Quant](https://www.interactivebrokers.com/campus/ibkr-quant-news/llms-and-the-shortening-shelf-life-of-copyable-alpha/).
- Backtest-bias surveys — [The three ways backtests lie](https://tesseraalpha.com/methodology/backtesting-survivorship-lookahead); [Backtest overfitting & live performance](https://quantalpha.co/en/blog/backtest-overfitting-and-live-performance); [McLean–Pontiff explainer](https://quantdecoded.com/en/anomaly-decay-after-publication-mclean-pontiff).
- Crowdsourced alpha & crowding — [Rise and fall of Quantopian](https://brokersdb.com/learn/quantopian-history-legacy-review); [Factor decay & pod shops](https://youngandcalculated.substack.com/p/factor-decay-is-real-how-published); [Why most retail algo systems fail](https://algotradingdesk.com/why-retail-algo-trading-systems-fail/).
- Platforms / curated lists — [Microsoft Qlib](https://github.com/microsoft/qlib) · [NautilusTrader](https://nautilustrader.io/) · [awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading) · [best-of-algorithmic-trading](https://github.com/merovinh/best-of-algorithmic-trading) · [awesome-quant](https://github.com/wilsonfreitas/awesome-quant).

## 6. Methodology

Multi-source survey via GitHub repository search (stars, 2026-06), web search, and
neural (Exa) search. ~30 sources reviewed; 11 cited. Sub-questions: (i) top public quant
repos and what they are; (ii) whether any contains deployable alpha; (iii) why genuine
winners win; (iv) structural reasons retail/personal quant fails; (v) corroboration of
the QuantLab four-wall thesis. Confidence: **High** for buckets A–C and mechanisms
2.1–2.5; **Medium** for bucket-D claims (self-reported, unaudited).
