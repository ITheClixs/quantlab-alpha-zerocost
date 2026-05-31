# crypto_only_riskalloc_v2 — Paper-Trading Decision

**Built:** 2026-05-30T15:29:57.515484+00:00

## Decision: **DO_NOT_ADVANCE** | failure_class: **single_asset_concentration_eth** | promotion_eligible: False

## Binding-gate scorecard
- ✅ windows_beaten_ge_3of5
- ✅ dd_improved_ge_4of5
- ✅ sharpe_pos_2x
- ✅ sharpe_pos_3x
- ✅ delay_ok
- ✅ no_year_gt_50pct
- ❌ no_asset_gt_65pct
- ❌ bootstrap_lower_pos
- ✅ dsr_ok
- ✅ random_inverted_fail
- ✅ paper_feasible_spot_weekly

## Rationale
- **Do NOT advance.** Classified `single_asset_concentration_eth`. Kept research_only; no paper/live. The crypto sleeve does not clear the stricter crypto-only gate either — consistent with the program's subsumption pattern.

_No paper trading executed. No live. No equity sleeve. No tuning. No new paid data._