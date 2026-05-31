"""Run strict Nasdaq/S&P backtests for the signal_research meta-labeler.

This runner trains the current triple-barrier meta-label model on each market
profile with chronological walk-forward folds, then runs diagnostics that are
strict enough to keep the result research-only unless full promotion gates are
later satisfied.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import polars as pl

from quant_research_stack.signal_research.papers.triple_barrier import TripleBarrierConfig
from quant_research_stack.signal_research.training.backtest_diagnostics import (
    StrictBacktestDiagnosticsConfig,
    StrictBacktestDiagnosticsResult,
    run_strict_backtest_diagnostics,
    write_strict_backtest_artifacts,
)
from quant_research_stack.signal_research.training.meta_label_walk_forward import (
    MetaLabelWalkForwardConfig,
    train_meta_label_walk_forward,
    write_meta_label_walk_forward_artifacts,
)

MARKET_PROFILES: dict[str, tuple[str, ...]] = {
    "nasdaq": ("NQ=F", "QQQ"),
    "sp500": ("ES=F", "SPY", "EW_BASKET"),
}


def _load_panel(data_root: Path, symbols: tuple[str, ...]) -> pl.DataFrame:
    files = sorted(data_root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {data_root}")
    panel = pl.concat([pl.read_parquet(path) for path in files], how="diagonal_relaxed").sort(["symbol", "date"])
    filtered = panel.filter(pl.col("symbol").is_in(list(symbols)))
    if filtered.is_empty():
        raise ValueError(f"no rows found for symbols {symbols} under {data_root}")
    return filtered


def _inventory(panel: pl.DataFrame, *, data_root: Path, profile: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for symbol, group in panel.group_by("symbol", maintain_order=True):
        rows.append(
            {
                "symbol": symbol[0] if isinstance(symbol, tuple) else symbol,
                "rows": group.height,
                "date_start": str(group["date"].min()),
                "date_end": str(group["date"].max()),
                "columns": group.columns,
            }
        )
    return {
        "profile": profile,
        "data_root": str(data_root),
        "row_count": panel.height,
        "date_start": str(panel["date"].min()),
        "date_end": str(panel["date"].max()),
        "symbols": rows,
        "timestamp_semantics": "daily bar date; no intraday timestamp, bid/ask, order book, queue, borrow, or funding fields",
        "quality_label": "research_proxy_daily_ohlcv",
    }


def _train_config(args: argparse.Namespace) -> MetaLabelWalkForwardConfig:
    return MetaLabelWalkForwardConfig(
        lookback_days=args.lookback_days,
        train_window_days=args.train_window_days,
        test_window_days=args.test_window_days,
        step_days=args.step_days,
        purge_days=args.purge_days,
        min_train_events=args.min_train_events,
        random_forest_estimators=args.random_forest_estimators,
        probability_threshold=args.probability_threshold,
        cost_bps_one_way=args.cost_bps_one_way,
        seed=args.seed,
        triple_barrier=TripleBarrierConfig(
            vertical_barrier_days=args.vertical_barrier_days,
            profit_take_multiplier=args.profit_take_multiplier,
            stop_loss_multiplier=args.stop_loss_multiplier,
            vol_estimator_window=args.vol_estimator_window,
            seed=args.seed,
        ),
    )


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.3%}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _summary_row(result: StrictBacktestDiagnosticsResult) -> dict[str, Any]:
    base = result.summary.get("base_metrics", {})
    pbo = result.summary.get("pbo", {})
    dsr = result.summary.get("deflated_sharpe", {})
    bootstrap = result.summary.get("bootstrap_sharpe_ci", {})
    return {
        "market": result.market_name,
        "status": result.summary["status"],
        "promotion_eligible": result.summary["promotion_eligible"],
        "net_total_return": base.get("net_total_return", 0.0),
        "avg_monthly_net_return": base.get("avg_monthly_net_return", 0.0),
        "net_daily_sharpe": base.get("net_daily_sharpe", 0.0),
        "max_drawdown": base.get("max_drawdown", 0.0),
        "trade_count": base.get("trade_count", 0),
        "net_hit_rate": base.get("net_hit_rate", 0.0),
        "bootstrap_ci_lower_95": bootstrap.get("ci_lower_95", 0.0),
        "bootstrap_ci_upper_95": bootstrap.get("ci_upper_95", 0.0),
        "deflated_sharpe_probability": dsr.get("probability", 0.0),
        "pbo_status": pbo.get("status", ""),
        "pbo_probability": pbo.get("pbo_probability", 0.0),
    }


def _write_comprehensive_report(
    *,
    output_root: Path,
    diagnostics: list[StrictBacktestDiagnosticsResult],
    inventories: list[dict[str, Any]],
    training_summaries: list[dict[str, Any]],
    command: str,
) -> Path:
    lines = [
        "# Nasdaq And S&P Strict Meta-Label Backtest Report",
        "",
        "## Executive Summary",
        "This report benchmarks the current supervised triple-barrier meta-labeler on Nasdaq and S&P proxy profiles. The run is deliberately strict about status: these are research-validation artifacts, not promotion-eligible trading systems.",
        "",
        "Important return semantics: the reported returns are equal-weight averages of event-level forward-horizon returns grouped by signal date. The triple-barrier labels can overlap, and execution is a daily close-to-close proxy rather than a venue-grade fill model. Treat daily Sharpe and monthly net as research diagnostics, not production portfolio statistics.",
        "",
        "| market | status | trades | net return | avg monthly net | daily Sharpe | max DD | net hit | bootstrap Sharpe 95% CI | DSR prob | PBO status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in diagnostics:
        row = _summary_row(result)
        lines.append(
            "| "
            f"{row['market']} | "
            f"{row['status']} | "
            f"{int(row['trade_count'])} | "
            f"{_fmt_pct(row['net_total_return'])} | "
            f"{_fmt_pct(row['avg_monthly_net_return'])} | "
            f"{_fmt_float(row['net_daily_sharpe'])} | "
            f"{_fmt_pct(row['max_drawdown'])} | "
            f"{_fmt_pct(row['net_hit_rate'])} | "
            f"{_fmt_float(row['bootstrap_ci_lower_95'])} to {_fmt_float(row['bootstrap_ci_upper_95'])} | "
            f"{_fmt_float(row['deflated_sharpe_probability'])} | "
            f"{row['pbo_status']} |"
        )
    lines.extend(
        [
            "",
            "## Data Scope",
            "The available local data is daily OHLCV-style benchmark data. It does not contain true bid/ask, order-book depth, queue position, futures roll, borrow, funding, or intraday execution timestamps. The cost model is therefore a conservative proxy, not a venue-grade execution simulator.",
            "",
        ]
    )
    for inventory in inventories:
        lines.extend(
            [
                f"### {inventory['profile']}",
                f"- data root: `{inventory['data_root']}`",
                f"- rows: `{inventory['row_count']}`",
                f"- date range: `{inventory['date_start']}` to `{inventory['date_end']}`",
                f"- quality label: `{inventory['quality_label']}`",
                f"- timestamp semantics: {inventory['timestamp_semantics']}",
                "",
                "| symbol | rows | date start | date end |",
                "|---|---:|---|---|",
            ]
        )
        for symbol_row in inventory["symbols"]:
            lines.append(
                f"| {symbol_row['symbol']} | {symbol_row['rows']} | {symbol_row['date_start']} | {symbol_row['date_end']} |"
            )
        lines.append("")
    lines.extend(["## Training Configuration", ""])
    for summary in training_summaries:
        cfg = summary.get("config", {})
        tb = cfg.get("triple_barrier", {})
        lines.extend(
            [
                f"### {summary.get('market', '')}",
                "- model: `RandomForestClassifier` secondary meta-label head on triple-barrier labels",
                "- primary signal: `log_return_lookback` momentum sign gated by meta-label probability",
                f"- folds: `{summary.get('fold_count', 0)}`",
                f"- prediction rows: `{summary.get('prediction_rows', 0)}`",
                f"- trades: `{summary.get('trade_count', 0)}`",
                f"- validation range: `{summary.get('date_start', '')}` to `{summary.get('date_end', '')}`",
                f"- lookback days: `{cfg.get('lookback_days', '')}`",
                f"- train/test/step/purge days: `{cfg.get('train_window_days', '')}` / `{cfg.get('test_window_days', '')}` / `{cfg.get('step_days', '')}` / `{cfg.get('purge_days', '')}`",
                f"- probability threshold: `{cfg.get('probability_threshold', '')}`",
                f"- cost bps one-way: `{cfg.get('cost_bps_one_way', '')}`",
                f"- triple-barrier vertical/profit/stop: `{tb.get('vertical_barrier_days', '')}` / `{tb.get('profit_take_multiplier', '')}` / `{tb.get('stop_loss_multiplier', '')}`",
                "",
            ]
        )
    lines.extend(["## Market-Level Diagnostics", ""])
    for result in diagnostics:
        report_path = output_root / result.market_name / "strict_backtest_report.md"
        lines.extend(
            [
                f"### {result.market_name}",
                f"- full report: `{report_path}`",
                f"- trade audit: `{output_root / result.market_name / 'strict_trade_audit.parquet'}`",
                f"- daily returns: `{output_root / result.market_name / 'strict_daily_returns.parquet'}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Promotion Decision",
            "No profile is promotion eligible from this run. The blockers are structural, not just numerical: no isolated permanent holdout for final selection, diagnostic-only PBO, approximate DSR, and proxy daily execution data rather than point-in-time tradable venue data.",
            "",
            "## Reproduction",
            f"```bash\n{command}\n```",
            "",
            "## Disclaimer",
            "The project may be production-intended, but this artifact is research output only and is not automatically investment advice. External advisory or capital-management use requires legal, regulatory, licensing, and compliance review before deployment.",
            "",
        ]
    )
    path = output_root / "comprehensive_report.md"
    path.write_text("\n".join(lines))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Nasdaq/S&P meta-label backtest benchmark.")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/strategy_benchmark"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--profiles", default="nasdaq,sp500", help="Comma-separated profiles: nasdaq,sp500")
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--train-window-days", type=int, default=252)
    parser.add_argument("--test-window-days", type=int, default=63)
    parser.add_argument("--step-days", type=int, default=63)
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--min-train-events", type=int, default=100)
    parser.add_argument("--random-forest-estimators", type=int, default=100)
    parser.add_argument("--probability-threshold", type=float, default=0.55)
    parser.add_argument("--cost-bps-one-way", type=float, default=1.0)
    parser.add_argument("--vertical-barrier-days", type=int, default=20)
    parser.add_argument("--profit-take-multiplier", type=float, default=1.5)
    parser.add_argument("--stop-loss-multiplier", type=float, default=1.5)
    parser.add_argument("--vol-estimator-window", type=int, default=20)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--multiple-testing-trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    unknown = sorted(set(requested_profiles) - set(MARKET_PROFILES))
    if unknown:
        raise ValueError(f"unknown profiles: {unknown}; available={sorted(MARKET_PROFILES)}")

    output_root = args.out_root
    output_root.mkdir(parents=True, exist_ok=True)
    train_cfg = _train_config(args)
    diagnostics: list[StrictBacktestDiagnosticsResult] = []
    inventories: list[dict[str, Any]] = []
    training_summaries: list[dict[str, Any]] = []

    all_metrics: list[pl.DataFrame] = []
    all_daily: list[pl.DataFrame] = []
    all_audits: list[pl.DataFrame] = []

    for profile in requested_profiles:
        symbols = MARKET_PROFILES[profile]
        profile_dir = output_root / profile
        panel = _load_panel(args.data_root, symbols)
        inventories.append(_inventory(panel, data_root=args.data_root, profile=profile))
        result = train_meta_label_walk_forward(panel=panel, config=train_cfg)
        summary = dict(result.summary)
        summary["market"] = profile
        training_summaries.append(summary)
        write_meta_label_walk_forward_artifacts(result, output_dir=profile_dir)

        diagnostics_result = run_strict_backtest_diagnostics(
            result.predictions,
            config=StrictBacktestDiagnosticsConfig(
                market_name=profile,
                cost_bps_one_way=args.cost_bps_one_way,
                holding_horizon_days=args.vertical_barrier_days,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.seed,
                multiple_testing_trials=args.multiple_testing_trials,
                random_seed=args.seed,
            ),
        )
        write_strict_backtest_artifacts(diagnostics_result, output_dir=profile_dir)
        diagnostics.append(diagnostics_result)
        all_metrics.append(diagnostics_result.variant_metrics)
        all_daily.append(diagnostics_result.daily_returns)
        all_audits.append(diagnostics_result.trade_audit.with_columns(pl.lit(profile).alias("market")))

    if all_metrics:
        pl.concat(all_metrics, how="vertical").write_parquet(output_root / "all_strict_variant_metrics.parquet")
    if all_daily:
        pl.concat(all_daily, how="vertical").write_parquet(output_root / "all_strict_daily_returns.parquet")
    if all_audits:
        pl.concat(all_audits, how="vertical").write_parquet(output_root / "all_strict_trade_audit.parquet")

    (output_root / "data_inventory.json").write_text(json.dumps(inventories, indent=2, sort_keys=True, default=str) + "\n")
    (output_root / "training_summaries.json").write_text(
        json.dumps(training_summaries, indent=2, sort_keys=True, default=str) + "\n"
    )
    summary_rows = [_summary_row(result) for result in diagnostics]
    pl.DataFrame(summary_rows).write_parquet(output_root / "market_summary.parquet")
    (output_root / "market_summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True, default=str) + "\n")
    command = " ".join(shlex.quote(arg) for arg in [sys.executable, *sys.argv])
    report = _write_comprehensive_report(
        output_root=output_root,
        diagnostics=diagnostics,
        inventories=inventories,
        training_summaries=training_summaries,
        command=command,
    )
    print(f"ok wrote strict Nasdaq/S&P report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
