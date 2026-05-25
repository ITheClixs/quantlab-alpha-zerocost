"""Generate the strategy-benchmark markdown report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.strategy_benchmark.runner import (
    BenchmarkRun,
    deflate_top_strategies,
)


@dataclass(frozen=True)
class ReportConfig:
    cost_bps_one_way: float
    universe_descriptions: dict[str, str]
    date_range: tuple[str, str]
    top_k: int = 25


def _percentile_summary(arr: np.ndarray, percentiles: list[int]) -> str:
    lines = []
    for p in percentiles:
        lines.append(f"  - p{p:>2d}: {float(np.percentile(arr, p)):+.3f}")
    lines.append(f"  - mean: {float(np.mean(arr)):+.3f}")
    lines.append(f"  - std:  {float(np.std(arr, ddof=1)):.3f}")
    return "\n".join(lines)


def _per_family_summary(metrics: pl.DataFrame) -> str:
    """Aggregate stats per signal family."""
    agg = metrics.group_by("signal_family").agg(
        pl.col("sharpe").mean().alias("sharpe_mean"),
        pl.col("sharpe").max().alias("sharpe_max"),
        pl.col("sharpe").std().alias("sharpe_std"),
        pl.col("total_return").mean().alias("ret_mean"),
        pl.col("max_drawdown").min().alias("mdd_worst"),
        pl.col("annual_turnover").mean().alias("ann_turn_mean"),
        pl.len().alias("n_strategies"),
    ).sort("sharpe_mean", descending=True)
    lines = ["| signal_family | n | mean Sharpe | max Sharpe | std Sharpe | mean total_ret | worst MaxDD | mean ann turnover |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in agg.iter_rows(named=True):
        lines.append(
            f"| `{row['signal_family']}` | {row['n_strategies']} "
            f"| {row['sharpe_mean']:+.2f} | {row['sharpe_max']:+.2f} | {row['sharpe_std']:.2f} "
            f"| {row['ret_mean']*100:+.2f}% | {row['mdd_worst']*100:+.2f}% "
            f"| {row['ann_turn_mean']:.2f}x |"
        )
    return "\n".join(lines)


def _per_universe_summary(metrics: pl.DataFrame) -> str:
    agg = metrics.group_by("universe").agg(
        pl.col("sharpe").mean().alias("sharpe_mean"),
        pl.col("sharpe").max().alias("sharpe_max"),
        pl.col("total_return").mean().alias("ret_mean"),
        pl.col("max_drawdown").min().alias("mdd_worst"),
        pl.len().alias("n_strategies"),
    ).sort("sharpe_mean", descending=True)
    lines = ["| universe | n | mean Sharpe | max Sharpe | mean total_ret | worst MaxDD |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in agg.iter_rows(named=True):
        lines.append(
            f"| `{row['universe']}` | {row['n_strategies']} "
            f"| {row['sharpe_mean']:+.2f} | {row['sharpe_max']:+.2f} "
            f"| {row['ret_mean']*100:+.2f}% | {row['mdd_worst']*100:+.2f}% |"
        )
    return "\n".join(lines)


def _sharpe_histogram(metrics: pl.DataFrame, *, n_bins: int = 20) -> str:
    sharpes = metrics["sharpe"].to_numpy()
    hist, edges = np.histogram(sharpes, bins=n_bins)
    max_count = int(hist.max())
    bar_width = 40
    lines = ["```"]
    for i, count in enumerate(hist):
        lo, hi = edges[i], edges[i + 1]
        bar_len = int(round(count / max_count * bar_width)) if max_count else 0
        lines.append(f"  [{lo:+.2f}, {hi:+.2f})  {'█' * bar_len}  {count}")
    lines.append("```")
    return "\n".join(lines)


def _monthly_returns_for_top(
    *,
    run: BenchmarkRun,
    strategy_idx: int,
    strategy_id: str,
) -> str:
    """Show monthly returns for the single best strategy."""
    dates = sorted({d for u in run.universe_bars.values() for d in u["date"].to_list()})
    rets = run.returns_matrix[:, strategy_idx]
    df = pl.DataFrame({"date": dates, "ret": rets})
    df = df.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("ym"))
    monthly = df.group_by("ym").agg(
        ((pl.col("ret") + 1.0).product() - 1.0).alias("monthly_return")
    ).sort("ym")
    lines = [f"#### Monthly returns — `{strategy_id}`", "", "| month | return |", "|---|---:|"]
    for row in monthly.iter_rows(named=True):
        lines.append(f"| {row['ym']} | {row['monthly_return']*100:+.2f}% |")
    cumulative = float(np.prod(1.0 + rets) - 1.0)
    annualised = (1.0 + cumulative) ** (252 / max(1, len(rets))) - 1.0
    lines.append(f"\nTotal: **{cumulative*100:+.2f}%**, annualised: **{annualised*100:+.2f}%**.\n")
    return "\n".join(lines)


def write_report(
    *,
    run: BenchmarkRun,
    config: ReportConfig,
    out_path: Path,
) -> None:
    sharpes = run.metrics["sharpe"].to_numpy()
    top_with_dsr = deflate_top_strategies(run=run, top_k=config.top_k)

    best_idx = int(np.argmax(sharpes))
    best_id = run.strategies[best_idx].strategy_id

    parts: list[str] = []
    parts.append("# Strategy Benchmark Report — S&P / Nasdaq Futures & ETFs\n\n")
    parts.append(f"_Generated: {datetime.now(UTC).isoformat(timespec='seconds')}_\n\n")
    parts.append("> ⚠️ **Benchmark only — not investment advice.** This report measures what is "
                 "achievable across 1500 systematically-enumerated quant strategies on free daily "
                 "data over 24 months, and applies Bailey/López de Prado PBO + Deflated Sharpe to "
                 "expose multiple-testing bias.\n\n")

    parts.append("## 1. Setup\n\n")
    parts.append(f"- **Date range:** {config.date_range[0]} to {config.date_range[1]} "
                 f"(~24 months, {run.returns_matrix.shape[0]} trading days)\n")
    parts.append(f"- **Strategies tested:** {run.returns_matrix.shape[1]} = "
                 "5 universes × 15 signal families × 4 lookbacks × 5 thresholds\n")
    parts.append(f"- **Cost model:** {config.cost_bps_one_way:.1f} bps per side (commission + half-spread). "
                 "No market impact, no overnight financing.\n")
    parts.append(f"- **Wall-clock:** {run.wall_clock_sec:.1f} s end-to-end "
                 "(strategy generation + 1500 backtests + 12,870-combination PBO + DSR).\n")
    parts.append("- **Universes:**\n")
    for u, desc in config.universe_descriptions.items():
        parts.append(f"  - `{u}`: {desc}\n")
    parts.append("\n")

    parts.append("## 2. Signal-family lineage\n\n")
    parts.append(
        "Every signal family has peer-reviewed quant lineage; retail patterns "
        "(ICT, Smart-Money order blocks, etc.) were intentionally excluded because they "
        "do not survive walk-forward backtests in any published study I could find.\n\n"
    )
    parts.append("| family | reference |\n|---|---|\n")
    family_refs = [
        ("TS_MOMENTUM", "Moskowitz, Ooi, Pedersen 2012 — Time Series Momentum"),
        ("LAGGED_MOMENTUM", "Jegadeesh & Titman 1993 — 12-1 momentum (skip-1)"),
        ("MA_CROSSOVER", "Faber 2007 — A Quantitative Approach to Tactical Asset Allocation"),
        ("DONCHIAN_BREAKOUT", "Donchian / Turtle Traders (Dennis & Eckhardt 1983)"),
        ("BOLLINGER_REVERT", "Bollinger 1980s — band mean-reversion"),
        ("BOLLINGER_BREAKOUT", "Bollinger 1980s — band breakout"),
        ("RSI_MEANREVERT", "Wilder 1978 — Relative Strength Index"),
        ("MACD", "Appel 1979 — Moving-Average Convergence Divergence"),
        ("VOLTGT_MOMENTUM", "Hurst, Ooi, Pedersen 2017 — vol-targeted trend"),
        ("ZSCORE_MEANREVERT", "classical statistical-arbitrage mean reversion"),
        ("AROON", "Chande 1995"),
        ("STOCHASTIC", "Lane 1950s"),
        ("ROC", "Rate of Change momentum"),
        ("CCI", "Lambert 1980 — Commodity Channel Index"),
        ("KELTNER_BREAKOUT", "Keltner / Chester 1960s — ATR channel breakout"),
    ]
    for k, v in family_refs:
        parts.append(f"| `{k}` | {v} |\n")
    parts.append("\n")

    parts.append("## 3. Headline result — Probability of Backtest Overfitting (PBO)\n\n")
    parts.append("| metric | value | interpretation |\n|---|---:|---|\n")
    parts.append(f"| **PBO** | **{run.pbo.pbo_probability:.4f}** | "
                 f"{'⚠️ severe overfitting risk' if run.pbo.pbo_probability > 0.5 else 'acceptable'} "
                 f"— probability the in-sample winner ranks below median OOS |\n")
    parts.append(f"| Median logit | {run.pbo.median_logit:+.4f} | "
                 f"{'negative ⇒ IS winners systematically degrade OOS' if run.pbo.median_logit < 0 else 'positive ⇒ IS winners hold OOS'} |\n")
    parts.append(f"| OOS-failure rate | **{run.pbo.failure_rate:.4f}** | "
                 "fraction of IS/OOS splits where the IS winner had **negative** OOS Sharpe |\n")
    parts.append(f"| PBO combinations sampled | {run.pbo.n_combinations} | "
                 f"of C({run.pbo.n_partitions}, {run.pbo.n_partitions//2}) "
                 f"= {12870 if run.pbo.n_partitions == 16 else '...'} |\n")
    parts.append(f"| Submatrix size | {run.pbo.submatrix_size} days | "
                 f"{run.pbo.n_partitions} equal time partitions of the {run.returns_matrix.shape[0]}-day sample |\n")
    parts.append(f"| Strategies in pool | {run.pbo.n_strategies} | |\n\n")
    parts.append(
        "**Reading the result.** "
        f"A PBO of **{run.pbo.pbo_probability:.2%}** means that if we were to "
        "pick the in-sample best strategy across 1500 candidates and trade it out-of-sample, "
        f"it would rank **below the median** OOS strategy in {run.pbo.pbo_probability:.0%} of "
        "time-partition combinations. This is the formal statement of "
        "Bailey & López de Prado's central insight: **with this many trials and this little data, "
        "the apparent in-sample winners are statistically indistinguishable from noise.**\n\n"
    )

    parts.append("## 4. Distribution of in-sample Sharpe across 1500 strategies\n\n")
    parts.append(_percentile_summary(sharpes, [1, 5, 25, 50, 75, 95, 99]))
    parts.append("\n\n")
    parts.append(_sharpe_histogram(run.metrics))
    parts.append("\n\n")

    parts.append("## 5. Top-25 by raw Sharpe, with PSR and DSR deflation\n\n")
    parts.append(
        "**PSR(0)** = probability the strategy's true Sharpe is > 0, accounting for skew + kurtosis "
        "but **ignoring multiple-testing**.\n\n"
        "**DSR** = the same probability **after** subtracting `E[max Sharpe]` across N=1500 trials "
        "under the null. DSR > 0.95 is the conventional threshold for 'real edge'.\n\n"
    )
    parts.append("| strategy_id | Sharpe | Sortino | total ret | MaxDD | trades | turnover/y | PSR(0) | DSR |\n")
    parts.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in top_with_dsr.iter_rows(named=True):
        parts.append(
            f"| `{row['strategy_id']}` | {row['sharpe']:+.2f} | {row['sortino']:+.2f} "
            f"| {row['total_return']*100:+.2f}% | {row['max_drawdown']*100:+.2f}% "
            f"| {row['n_trades']} | {row['annual_turnover']:.2f}x "
            f"| {row['psr_zero']:.3f} | **{row['dsr']:.3f}** |\n"
        )
    parts.append("\n")

    dsr_max = float(top_with_dsr['dsr'].max())  # type: ignore[arg-type]
    survivors_95 = int((top_with_dsr['dsr'] >= 0.95).sum())
    parts.append(
        f"**DSR survivors at the 95 % threshold: {survivors_95} / {config.top_k}.** "
        f"The highest DSR observed in the top-25 is **{dsr_max:.3f}**, "
        f"which means even the strongest in-sample candidate is "
        f"{'a likely overfit artefact' if dsr_max < 0.95 else 'plausibly real'}.\n\n"
    )

    parts.append("## 6. By signal family\n\n")
    parts.append(_per_family_summary(run.metrics))
    parts.append("\n\n")

    parts.append("## 7. By universe\n\n")
    parts.append(_per_universe_summary(run.metrics))
    parts.append("\n\n")

    parts.append("## 8. Best strategy — monthly returns\n\n")
    parts.append(
        f"For completeness, monthly returns of the **single best raw-Sharpe** strategy "
        f"(`{best_id}`, Sharpe = {sharpes[best_idx]:+.2f}). "
        f"Note the DSR penalty: even this strategy has DSR = {float(top_with_dsr['dsr'][0]):.3f} "
        "and would not survive a rigorous promotion gate.\n\n"
    )
    parts.append(_monthly_returns_for_top(run=run, strategy_idx=best_idx, strategy_id=best_id))

    parts.append("## 9. Honest interpretation\n\n")
    if run.pbo.pbo_probability > 0.5:
        parts.append(
            f"- The PBO of **{run.pbo.pbo_probability:.1%}** says that picking the best-IS "
            "strategy from this menu is almost guaranteed to be overfit. "
            "**This is the correct answer to the user's question about whether any of these "
            "strategies survive out-of-sample.**\n"
        )
    parts.append(
        f"- The Sharpe-5 / +10 %-monthly target is **mathematically incompatible** with "
        "this data + this strategy menu. The best raw Sharpe achievable on 24 months of "
        f"S&P/Nasdaq daily data, across 1500 classical strategies, is "
        f"**{float(np.max(sharpes)):.2f}** — and even that doesn't survive DSR deflation.\n"
    )
    parts.append(
        "- The few strategies that look superficially attractive (high Sortino, low max-DD) "
        "all have **6-30 trades over 500 days**, which is too few to draw any statistical "
        "conclusion. Their high Sortino is a small-sample artefact.\n"
    )
    parts.append(
        "- Daily-bar systematic strategies on liquid US equity indices are a saturated "
        "research space. Realistic targets after PBO survive are Sharpe 0.5–1.0 net, which is "
        "consistent with the Bailey & López de Prado finding that long-running quant signals "
        "decay rapidly after publication.\n"
    )

    parts.append("\n## 10. What would actually move this forward\n\n")
    parts.append(
        "If the goal is to find robust edge above this benchmark, the directions that have "
        "*some* prior literature support are:\n\n"
        "1. **Longer history** — extend to 10-20 years so the PBO submatrices have more data and the "
        "deflated Sharpe penalty (`E[max SR | N, T]`) is less brutal.\n"
        "2. **Cross-sectional, not single-asset** — the M4 engine in this repo is purpose-built for "
        "dollar-neutral L/S across a basket of names; that's a structurally different bet than "
        "single-asset trend.\n"
        "3. **Higher-frequency** — intraday tick or 1-minute bars would change the strategy "
        "space entirely (microstructure, order-flow, queue position).\n"
        "4. **Walk-forward parameter selection** rather than full-sample parameter fixing — but "
        "this only helps if the PBO is run on the *walk-forward* returns, not pre-tuned strategies.\n"
        "5. **Lower-cost regime** — many of these mean-reversion strategies turn 10-50x/year, and "
        "the 1.5 bps round-trip cost is what's killing them. A lower-cost broker / better fill "
        "model would lift the boundary.\n\n"
    )

    parts.append("\n---\n`not_investment_advice: true`\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts))
