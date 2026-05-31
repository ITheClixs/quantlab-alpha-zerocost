# Zero-Cost Risk-Allocator v1 — Paper-Trading Decision

**Built:** 2026-05-30T15:13:52.779560+00:00

## Decision: **DO_NOT_ADVANCE**
- failure_class: **crypto_regime_concentration** | product label: **crisis_insurance_allocator** | promotion_eligible: False (paper only at most)

## Decision-rule scorecard
- beats vol-targeted BAH on ≥3/5 holdouts (Sharpe or Calmar): True (4/5)
- improves max drawdown consistently (≥4/5): True (5/5)
- survives 1- & 2-bar delay (≤0.5 Sharpe loss): True
- not crypto-dependent (<50% PnL) + crypto-out still beats on DD: False
- not one-instrument (≤35%): True
- ex-crisis acceptable for role (≥0): True
- clear product label: crisis_insurance_allocator

## Rationale
- **Do NOT advance to paper.** Classified `crypto_regime_concentration`: the candidate's apparent edge does not hold up under the stricter review. No paper, no live; kept research_only. This is consistent with the program's recurring finding that single-index risk-timing overlays are subsumed by / reduce to vol-targeting + crisis luck.

_No paper trading executed. No live. No gate weakening. No tuning._