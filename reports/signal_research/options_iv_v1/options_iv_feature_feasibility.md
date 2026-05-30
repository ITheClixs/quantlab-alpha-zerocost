# Options-IV Data Audit — §3/§6 Chain Structure & Feature Feasibility

## §3 Chain structure (what the dataset is)
- This is a **daily per-underlying AGGREGATE**, not a raw option chain.
- strikes: **NO** | expiry dates: **NO** (only `expirations_number` count) | bid/ask: **NO** |
  per-contract prices: **NO** | greeks: **NO**.
- volume: YES (calls/puts contracts traded) | open interest: YES (calls/puts) | IV: SUPPLIED as 7
  moneyness buckets (DITM, ITM, sITM, ATM, sOTM, OTM, DOTM) | call/put: split for volume/OI, but IV
  buckets are by moneyness (not explicitly call vs put).

## §6 Feature feasibility (timestamp-safe, EOD → next-day)

| feature | feasible | how |
|---|:---:|---|
| ATM IV | YES | `ATM_IV` |
| 25-delta put/call skew | PROXY | no true delta; moneyness-IV spread `DOTM_IV-DITM_IV` / `OTM-ITM` (approximate) |
| IV term structure | **NO** | no per-expiry IV; only `expirations_number` count |
| IV rank / percentile | YES | trailing per-symbol history of ATM_IV |
| put/call IV spread | PARTIAL | IV not split call/put; put/call **volume & OI imbalance** YES |
| realized − implied (VRP proxy) | YES | `hv_20..200` vs `ATM_IV` |
| option volume / OI imbalance | YES | calls vs puts traded / OI |
| skew change | YES | Δ of the moneyness-IV-spread proxy |
| term-structure slope change | **NO** | no term structure |
| cross-sectional IV dispersion | YES | across the ~3,900-name universe per day |

**§3/§6 verdict: `options_features_only` — usable as features for trading SPY/QQQ/DIA/IWM (present) or
underlying equities; term-structure features are NOT available. Direct option trading is impossible
(no strikes/expiries/bid-ask).**
