"""EDGAR 10-K text-features v1 — classical-feature cross-sectional validation.

Intake: docs/research/intake/2026-05-30-edgar-10k-text-features-v1.md (research_only).
Signal timestamp = SEC filing date; cross-section by filing year; forward returns
are LABELS ONLY. Classical features only (no embeddings/LLM/NN in v1). Conservative
models (Ridge, ElasticNet, LightGBM-secondary). The text model must beat the
OHLCV/factor baselines (size, event-return) on rank IC AND decile spread AND net
long-short PnL to remain alive.

Emits under reports/signal_research/edgar_10k_v1/:
  edgar_10k_feature_registry.parquet, edgar_10k_validation_report.md,
  edgar_10k_rank_ic_report.md, edgar_10k_decile_spread_report.md,
  edgar_10k_baseline_comparison_report.md, edgar_10k_cost_decomposition.parquet,
  edgar_10k_failure_classification.md

Usage:
    PYTHONPATH=src uv run python scripts/run_edgar_10k_v1.py
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet, Ridge

from quant_research_stack.crypto_research.perps.validation import (
    bootstrap_sharpe_payload,
    deflated_sharpe_payload,
    estimate_registry_pbo,
)
from quant_research_stack.signal_research.edgar.features import build_feature_frame, feature_columns

warnings.filterwarnings("ignore")
_OUT = Path("reports/signal_research/edgar_10k_v1")
_REGISTRY = _OUT / "edgar_10k_feature_registry.parquet"
_LABEL = "fwd_ret_63"
_TRAIN_MAX, _VAL_MAX = 2017, 2019  # holdout = 2020-2022
_DECILES = 5  # quintiles (annual cohorts ~480 names -> 5 robust buckets)
_ONE_WAY_BPS = 10.0
_BASELINES = {"baseline_size", "baseline_event_ret", "placebo_random_rank",
              "placebo_shuffled_text", "sanity_inverted_text"}


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order.astype(np.float64)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 5 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _decile_spread(sig: np.ndarray, y: np.ndarray, q: int = _DECILES) -> tuple[float, list[float]]:
    if sig.size < q * 2:
        return 0.0, []
    ranks = _rank(sig)
    edges = np.quantile(ranks, np.linspace(0, 1, q + 1))
    means = []
    for i in range(q):
        lo, hi = edges[i], edges[i + 1]
        mask = (ranks >= lo) & (ranks <= hi) if i == q - 1 else (ranks >= lo) & (ranks < hi)
        means.append(float(np.mean(y[mask])) if mask.any() else 0.0)
    return means[-1] - means[0], means


def _zscore_by_cohort(frame: pl.DataFrame, cols: list[str], cohort: str) -> np.ndarray:
    out = []
    for c in cols:
        s = frame.select([cohort, c]).with_columns(
            ((pl.col(c) - pl.col(c).mean().over(cohort)) / (pl.col(c).std().over(cohort) + 1e-9)).alias("z")
        )["z"].fill_null(0.0).fill_nan(0.0).to_numpy()
        out.append(s)
    return np.column_stack(out) if out else np.zeros((frame.height, 0))


def _fit_models(xtr: np.ndarray, ytr: np.ndarray, xall: np.ndarray) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    ridge = Ridge(alpha=10.0).fit(xtr, ytr)
    preds["model_ridge_text"] = ridge.predict(xall)
    enet = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000).fit(xtr, ytr)
    preds["model_elasticnet_text"] = enet.predict(xall)
    try:
        from lightgbm import LGBMRegressor
        lgbm = LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.6, min_child_samples=50,
                             random_state=17, verbosity=-1).fit(xtr, ytr)
        preds["model_lgbm_text"] = lgbm.predict(xall)
    except Exception:
        pass
    return preds


def _evaluate(sig: np.ndarray, y: np.ndarray, years: np.ndarray, split_years: set[int]) -> dict[str, Any]:
    ics, spreads = [], []
    for yr in sorted(set(years.tolist()) & split_years):
        m = years == yr
        if m.sum() < _DECILES * 2:
            continue
        ics.append(_spearman(sig[m], y[m]))
        sp, _ = _decile_spread(sig[m], y[m])
        spreads.append(sp)
    ics_a, sp_a = np.array(ics), np.array(spreads)
    n = max(ics_a.size, 1)
    ic_t = float(np.mean(ics_a) / (np.std(ics_a, ddof=1) + 1e-9) * np.sqrt(n)) if ics_a.size > 1 else 0.0
    return {
        "mean_ic": float(np.mean(ics_a)) if ics_a.size else 0.0,
        "ic_tstat": ic_t,
        "mean_spread": float(np.mean(sp_a)) if sp_a.size else 0.0,
        "spread_series": spreads,
        "cohorts": ics_a.size,
    }


def _net_pnl(spread_series: list[float], *, cost_mult: float = 1.0) -> float:
    cost = 2.0 * 2.0 * _ONE_WAY_BPS * 1e-4 * cost_mult  # two legs, round trip, per annual rebalance
    net = [s - cost for s in spread_series]
    return float(np.prod([1.0 + r for r in net]) - 1.0) if net else 0.0


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(_REGISTRY) if _REGISTRY.exists() else build_feature_frame()
    if not _REGISTRY.exists():
        frame.write_parquet(_REGISTRY)
    frame = frame.filter(pl.col(_LABEL).is_not_null())
    years = frame["filing_year"].to_numpy()
    y = frame[_LABEL].to_numpy().astype(np.float64)

    all_feats = feature_columns(frame)
    text_feats = [c for c in all_feats if c not in ("size_log_mktcap", "event_ret")]
    z_text = _zscore_by_cohort(frame, text_feats, "filing_year")
    train_mask = years <= _TRAIN_MAX

    signals: dict[str, np.ndarray] = _fit_models(z_text[train_mask], y[train_mask], z_text)
    # univariate baselines / placebos (sign fixed on TRAIN ic)
    rng = np.random.default_rng(20260530)
    raw = {
        "baseline_size": frame["size_log_mktcap"].to_numpy().astype(np.float64),
        "baseline_event_ret": frame["event_ret"].to_numpy().astype(np.float64),
        "feat_rf_negative_ratio": frame["rf_negative_ratio"].fill_null(0.0).to_numpy().astype(np.float64),
        "feat_rf_uncertainty_ratio": frame["rf_uncertainty_ratio"].fill_null(0.0).to_numpy().astype(np.float64),
        "feat_rf_yoy_change": frame["rf_yoy_change"].fill_null(0.0).to_numpy().astype(np.float64),
        "feat_mda_net_tone": frame["mda_net_tone"].fill_null(0.0).to_numpy().astype(np.float64),
    }
    for name, s in raw.items():
        tr_ic = _spearman(s[train_mask], y[train_mask])
        signals[name] = s * (-1.0 if tr_ic < 0 else 1.0)
    signals["placebo_random_rank"] = rng.normal(size=frame.height)
    base_pred = signals.get("model_ridge_text", np.zeros(frame.height)).copy()
    shuffled = base_pred.copy()
    for yr in set(years.tolist()):
        idx = np.where(years == yr)[0]
        shuffled[idx] = rng.permutation(shuffled[idx])
    signals["placebo_shuffled_text"] = shuffled
    signals["sanity_inverted_text"] = -base_pred

    splits = {"train": set(range(2010, _TRAIN_MAX + 1)),
              "val": set(range(_TRAIN_MAX + 1, _VAL_MAX + 1)),
              "holdout": set(range(_VAL_MAX + 1, 2023))}
    results: dict[str, dict[str, Any]] = {}
    for name, sig in signals.items():
        results[name] = {sp: _evaluate(sig, y, years, yrs) for sp, yrs in splits.items()}

    # pool-level multiple-testing control: per-cohort holdout spread matrix
    holdout_years = sorted(splits["holdout"])
    pool = list(signals.keys())
    spread_cols: dict[str, list[float]] = {}
    for name in pool:
        per = []
        sig = signals[name]
        for yr in holdout_years:
            m = years == yr
            sp, _ = _decile_spread(sig[m], y[m]) if m.sum() >= _DECILES * 2 else (0.0, [])
            per.append(sp)
        spread_cols[name] = per
    pbo_df = pl.DataFrame(spread_cols).with_row_index("event_index")
    pbo = estimate_registry_pbo(pbo_df, strategy_columns=pool) if len(holdout_years) >= 2 else {"pbo_probability": None}

    # best TEXT model on holdout
    text_models = [n for n in signals if n.startswith("model_")]
    best = max(text_models, key=lambda n: results[n]["holdout"]["mean_ic"]) if text_models else ""
    best_h = results.get(best, {}).get("holdout", {})
    best_spread_series = best_h.get("spread_series", [])
    boot = bootstrap_sharpe_payload(np.array(best_spread_series)) if best_spread_series else {}
    dsr = deflated_sharpe_payload(np.array(best_spread_series), trials=len(pool)) if best_spread_series else {}
    best_net = _net_pnl(best_spread_series)
    best_net_2x = _net_pnl(best_spread_series, cost_mult=2.0)

    size_h = results.get("baseline_size", {}).get("holdout", {})
    er_h = results.get("baseline_event_ret", {}).get("holdout", {})
    beats_ic = best_h.get("mean_ic", 0) > max(size_h.get("mean_ic", 0), er_h.get("mean_ic", 0))
    beats_spread = best_h.get("mean_spread", 0) > max(size_h.get("mean_spread", 0), er_h.get("mean_spread", 0))
    beats_pnl = best_net > max(_net_pnl(size_h.get("spread_series", [])), _net_pnl(er_h.get("spread_series", [])))
    placebo_max_ic = max(results["placebo_random_rank"]["holdout"]["mean_ic"],
                         results["placebo_shuffled_text"]["holdout"]["mean_ic"])

    classification = _classify(best, best_h, beats_ic, beats_spread, beats_pnl, best_net, best_net_2x,
                               placebo_max_ic, pbo, results, frame)
    _write_reports(frame, results, pool, best, best_h, best_net, best_net_2x, boot, dsr, pbo,
                   size_h, er_h, beats_ic, beats_spread, beats_pnl, placebo_max_ic, classification,
                   text_feats, splits)
    print(f"best text model: {best} | holdout IC={best_h.get('mean_ic'):.4f} "
          f"spread={best_h.get('mean_spread'):.4f} net_pnl={best_net:.4f}")
    print(f"beats baselines: IC={beats_ic} spread={beats_spread} pnl={beats_pnl} | PBO={pbo.get('pbo_probability')}")
    print(f"classification: {classification}")
    return 0


def _classify(best, best_h, beats_ic, beats_spread, beats_pnl, best_net, best_net_2x,
              placebo_max_ic, pbo, results, frame) -> dict[str, Any]:
    blockers: list[str] = []
    if not (beats_ic and beats_spread and beats_pnl):
        blockers.append("text_signal_subsumed_by_ohlcv_or_factors")
    if best_h.get("mean_ic", 0) <= placebo_max_ic + 0.01:
        blockers.append("placebo_indistinguishable")
    if abs(best_h.get("mean_ic", 0)) < 0.02:
        blockers.append("weak_rank_ic")
    if best_net <= 0 or best_net_2x <= 0:
        blockers.append("cost_failure")
    p = pbo.get("pbo_probability")
    if p is not None and p > 0.25:
        blockers.append("high_pbo")
    if best_h.get("cohorts", 0) < 5:  # annual filings -> too few independent holdout cross-sections
        blockers.append("low_frequency_insufficient_sample")
    blockers.append("research_only")
    hard = [b for b in blockers if b != "research_only"]
    if not hard:
        failure = "none"
    elif "low_frequency_insufficient_sample" in blockers:
        failure = "low_frequency_insufficient_sample"  # root cause when holdout is tiny
    elif "text_signal_subsumed_by_ohlcv_or_factors" in blockers:
        failure = "text_signal_subsumed_by_ohlcv_or_factors"
    elif "placebo_indistinguishable" in blockers or "weak_rank_ic" in blockers:
        failure = "no_text_edge"
    elif "cost_failure" in blockers:
        failure = "cost_failure"
    else:
        failure = "holdout_failure"
    return {
        "strategy_id": "edgar_10k_text_features_v1", "best_model": best,
        "status": "research_pass" if not hard else "none",
        "research_candidate": not hard, "promotion_eligible": False,
        "failure_class": failure, "blockers": blockers,
    }


def _write_reports(frame, results, pool, best, best_h, best_net, best_net_2x, boot, dsr, pbo,
                   size_h, er_h, beats_ic, beats_spread, beats_pnl, placebo_max_ic, classification,
                   text_feats, splits) -> None:
    built = datetime.now(UTC).isoformat()

    def ic_row(name: str) -> str:
        r = results[name]
        return (f"| `{name}` | {r['train']['mean_ic']:.4f} | {r['val']['mean_ic']:.4f} | "
                f"{r['holdout']['mean_ic']:.4f} | {r['holdout']['ic_tstat']:.2f} | {r['holdout']['mean_spread']:.4f} |")

    ordered = sorted(pool, key=lambda n: -results[n]["holdout"]["mean_ic"])
    (_OUT / "edgar_10k_rank_ic_report.md").write_text("\n".join([
        "# EDGAR 10-K v1 — Rank IC Report",
        f"\n**Built:** {built} | label `{_LABEL}` (~63d fwd) | cross-section = filing year | quintile buckets.",
        "Splits: train ≤2017, val 2018-2019, holdout 2020-2022. Univariate sign fixed on train.\n",
        "| signal | train IC | val IC | holdout IC | holdout IC t | holdout spread |",
        "|---|---:|---:|---:|---:|---:|",
        *[ic_row(n) for n in ordered],
    ]) + "\n")

    (_OUT / "edgar_10k_decile_spread_report.md").write_text("\n".join([
        "# EDGAR 10-K v1 — Decile (Quintile) Spread Report",
        f"\n**Built:** {built} | top-minus-bottom quintile mean ~63d forward return, per filing-year cohort.\n",
        f"- Best text model `{best}`: holdout mean spread **{best_h.get('mean_spread', 0):.4f}**, "
        f"net LS PnL (compounded, cost {_ONE_WAY_BPS}bps/leg) **{best_net*100:.2f}%**, @2x cost **{best_net_2x*100:.2f}%**.",
        f"- Size baseline holdout spread: {size_h.get('mean_spread', 0):.4f} | event-ret: {er_h.get('mean_spread', 0):.4f}.",
        f"- Holdout spread series (best): {[round(x, 4) for x in best_h.get('spread_series', [])]}",
    ]) + "\n")

    (_OUT / "edgar_10k_baseline_comparison_report.md").write_text("\n".join([
        "# EDGAR 10-K v1 — Baseline Comparison",
        f"\n**Built:** {built} | decision = text model must beat OHLCV/factor baselines on IC AND spread AND net PnL.\n",
        "| metric | best text model | size baseline | event-ret baseline | placebo max |",
        "|---|---:|---:|---:|---:|",
        f"| holdout mean IC | {best_h.get('mean_ic', 0):.4f} | {size_h.get('mean_ic', 0):.4f} | "
        f"{er_h.get('mean_ic', 0):.4f} | {placebo_max_ic:.4f} |",
        f"| holdout mean spread | {best_h.get('mean_spread', 0):.4f} | {size_h.get('mean_spread', 0):.4f} | "
        f"{er_h.get('mean_spread', 0):.4f} | — |",
        f"| net LS PnL | {best_net*100:.2f}% | {_net_pnl(size_h.get('spread_series', []))*100:.2f}% | "
        f"{_net_pnl(er_h.get('spread_series', []))*100:.2f}% | — |",
        "",
        f"- Beats baselines — IC: **{beats_ic}**, spread: **{beats_spread}**, net PnL: **{beats_pnl}**.",
        "- **Data caveat:** full momentum/vol/low-vol factor baselines are NOT computable — they need a",
        "  survivorship-safe price panel the program lacks (equity-return audit `3f9a658`). Size (mkt_cap) is",
        "  the binding factor baseline here; full factor subsumption is a v2 item gated on a clean price panel.",
    ]) + "\n")

    # cost decomposition parquet
    pl.DataFrame({
        "scenario": ["gross", "net_1x", "net_2x"],
        "ls_pnl": [_net_pnl(best_h.get("spread_series", []), cost_mult=0.0), best_net, best_net_2x],
        "one_way_bps": [0.0, _ONE_WAY_BPS, 2 * _ONE_WAY_BPS],
        "rebalance": ["annual", "annual", "annual"],
    }).write_parquet(_OUT / "edgar_10k_cost_decomposition.parquet")

    # concentration by sector (sic2) on holdout for the best signal
    val = [
        "# EDGAR 10-K v1 — Validation Report",
        f"\n**Built:** {built} | git label `{_LABEL}` | research_only.",
        "**Intake:** `docs/research/intake/2026-05-30-edgar-10k-text-features-v1.md`.",
        f"\n## Data\n- {frame.height:,} filings (label-valid), {frame['cik'].n_unique()} companies, "
        f"filing years {frame['filing_year'].min()}-{frame['filing_year'].max()}.",
        f"- {len(text_feats)} classical text features (no embeddings/LLM/NN). LM lexicon is a compact subset",
        "  (v1 approximation — pre-registered limitation).",
        "\n## Leakage audit",
        "- Signal = SEC filing date; cross-section by filing year; forward returns are LABELS ONLY (guarded in",
        "  `features._assert_no_label_leak`). YoY features use only the same CIK's strictly-earlier filing.",
        "- Univariate signal signs fixed on TRAIN only; models fit on TRAIN cohorts (≤2017); holdout 2020-2022.",
        "\n## Pool-level multiple-testing control",
        f"- PBO (CSCV over {len(pool)} signals, holdout cohorts): **{pbo.get('pbo_probability')}**",
        f"- Best text model `{best}`: holdout IC **{best_h.get('mean_ic', 0):.4f}** (t={best_h.get('ic_tstat', 0):.2f}), "
        f"DSR **{dsr.get('probability')}**, bootstrap spread CI lower **{boot.get('ci_lower_95')}**.",
        "\n## Classification",
        f"- status **{classification['status']}** | failure_class **{classification['failure_class']}** | "
        f"research_candidate {classification['research_candidate']} | promotion_eligible False",
        f"- blockers: `{', '.join(classification['blockers'])}`",
    ]
    (_OUT / "edgar_10k_validation_report.md").write_text("\n".join(val) + "\n")

    if classification["failure_class"] != "none":
        (_OUT / "edgar_10k_failure_classification.md").write_text("\n".join([
            "# EDGAR 10-K v1 — Failure Classification",
            f"\n**Primary failure class:** `{classification['failure_class']}`",
            f"**Best text model:** `{best}` (holdout IC {best_h.get('mean_ic', 0):.4f}).",
            "\n## Evidence",
            f"- Beats size/event-ret baselines: IC={beats_ic}, spread={beats_spread}, net PnL={beats_pnl}.",
            f"- Holdout IC {best_h.get('mean_ic', 0):.4f} vs placebo max {placebo_max_ic:.4f}.",
            f"- Net LS PnL {best_net*100:.2f}% (1x), {best_net_2x*100:.2f}% (2x cost).",
            f"- PBO {pbo.get('pbo_probability')}, DSR {dsr.get('probability')}.",
            "\n## Decision",
            "Per the intake §16: classical 10-K text features did not clear the bar (beat OHLCV/factor baselines",
            "on rank IC AND decile spread AND net long-short PnL). v1 closed at the stated failure class.",
            "Embeddings/LLM features are NOT added in this run (would require a separate intake). The compact",
            "LM-lexicon limitation is noted; a full-dictionary v2 could be reconsidered only with a fresh intake.",
        ]) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
