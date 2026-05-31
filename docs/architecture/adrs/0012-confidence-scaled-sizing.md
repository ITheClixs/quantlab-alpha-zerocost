# ADR 0012: Confidence-scaled position sizing with hard caps

## Status
Accepted, 2026-05-20.

## Context
S4 must translate an S1 numeric prediction (`predicted_score`, `confidence`) and an
S2 `GovernorVerdict` (`decision`, `direction`, `confidence`) into an order quantity.
The S2 spec's Tier-3 verdict contributes a stance modifier whose magnitude is in
`configs/governor.yaml` as `stance.tier3_stance_modifier_pct` (default 0.20). The
goal is a sizing rule that (a) respects the predictor's strength signal, (b) is
hard-capped so a runaway confidence value cannot blow up exposure, and (c) shrinks
when Tier 3 disagrees and grows when Tier 3 agrees.

## Decision
We use a confidence-scaled rule with hard caps:

```text
stance_mod ∈ {-cfg_pct, 0, +cfg_pct} depending on tier3 vs primary direction
target_notional = equity * base_pct * primary.confidence * (1 + stance_mod)
target_notional = min(target_notional, equity * max_per_symbol_pct)
qty = primary.direction * target_notional / mid_price  ; rounded to lot
```

`Decision.veto` short-circuits the Sizer to `qty = 0`. `direction == 0` also yields
`qty = 0`. The `max_per_symbol_pct` and `base_notional_per_trade_pct` come from
`configs/risk.yaml`.

## Alternatives considered
- Kelly-lite fixed-fractional: simpler but discards the predictor's strength signal.
- Volatility-targeted: principled but adds a rolling-vol feature dependency we
  don't yet have wired through.

## Consequences
The sizer is bounded by construction; the worst-case single-trade notional is
`equity * max_per_symbol_pct`. Tier-3 disagreements shrink positions by up to
`cfg_pct` (default 20%). Veto closes the trade entirely.
