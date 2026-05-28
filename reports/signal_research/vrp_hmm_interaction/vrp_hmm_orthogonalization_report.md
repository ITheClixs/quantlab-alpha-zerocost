# Orthogonalized VRP — Independence Test

## Method

On the dev window, regress the binary `vrp_long_only` signal on the
binary `hmm_risk_on` signal (with intercept). Use the fitted (α, β) to
compute the residual VRP component on the full sample. Trade sign(residual).

This isolates the VRP information that is NOT explained by the HMM regime.
If this strategy generates positive risk-adjusted returns, VRP carries
independent information beyond HMM risk-timing.

## Results

- orthogonalized_vrp dev Sharpe: +0.898
- bootstrap 95% CI: [+0.454, +1.419]
- holdout Sharpe: +1.203
- cost-stress 2× Sharpe: +0.880
- 1-bar delay Sharpe: +0.787
- correlation with HMM-only (dev): +0.563
- residual Sharpe vs HMM-only (dev): +0.000
- Sharpe excluding 2020: +0.890
- Sharpe excluding 2022: +1.038
- exposure time fraction: 86.1%

## Survival criteria (all required)

- bootstrap CI lower > 0: PASS (+0.454)
- dev Sharpe > 0.3: PASS (+0.898)
- cost-stress 2× Sharpe > 0: PASS (+0.880)
- residual Sharpe vs HMM > 0.3: FAIL (+0.000)

## Verdict: FAILS — VRP information is subsumed by HMM regime
