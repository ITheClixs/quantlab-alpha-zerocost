from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_research_stack.crypto_research.backtest import (
    BacktestConfig,
    BacktestResult,
    run_variant_backtest,
    summarize_backtest_frames,
)
from quant_research_stack.crypto_research.data import (
    Period,
    build_chronological_periods,
    chronological_blocks,
    dataset_manifest_from_frame,
    load_btcusdt_1m_panel,
    write_dataset_manifest,
)
from quant_research_stack.crypto_research.pbo import approximate_multiple_testing_payload, estimate_pbo
from quant_research_stack.crypto_research.reports import write_research_outputs
from quant_research_stack.crypto_research.strategies import StrategyVariant, generate_strategy_variants

DEFAULT_BTCUSDT_1M = Path(
    "data/raw/huggingface/vaquum__binance_btcusdt_1m_klines/btcusdt_1m_kline_20200101_to_20260511.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict crypto strategy research loop with PBO diagnostics.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_BTCUSDT_1M)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/crypto_strategy_loop"))
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--target-count", type=int, default=1500)
    parser.add_argument("--pbo-blocks", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=5)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--half-spread-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--execution-delay-bars", type=int, default=1)
    parser.add_argument("--notional-usd", type=float, default=100_000.0)
    return parser.parse_args()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _to_float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _slice_timestamp(frame: pl.DataFrame, period: Period) -> pl.DataFrame:
    return frame.filter((pl.col("timestamp") >= period.start) & (pl.col("timestamp") <= period.end))


def _summarize_period(
    result: BacktestResult,
    variant: StrategyVariant,
    *,
    config: BacktestConfig,
    period: Period,
) -> dict[str, Any]:
    pnl = _slice_timestamp(result.pnl, period)
    trades = _slice_timestamp(result.trades, period) if not result.trades.is_empty() else result.trades
    return summarize_backtest_frames(
        strategy_id=variant.strategy_id,
        family=variant.family,
        pnl=pnl,
        trades=trades,
        config=config,
        period_name=period.name,
    )


def _block_periods(frame: pl.DataFrame, *, block_count: int) -> list[Period]:
    periods: list[Period] = []
    for index, block in enumerate(chronological_blocks(frame, timestamp_column="timestamp", block_count=block_count)):
        timestamps = block.get_column("timestamp")
        periods.append(Period(name=f"block_{index}", start=timestamps[0], end=timestamps[-1]))
    return periods


def _monthly_net_mean(pnl: pl.DataFrame) -> float:
    if pnl.is_empty():
        return 0.0
    monthly = (
        pnl.with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(((pl.col("net_return") + 1.0).product() - 1.0).alias("net_return"))
    )
    if monthly.is_empty():
        return 0.0
    return _to_float(monthly.get_column("net_return").mean())


def _best_day_concentration(pnl: pl.DataFrame) -> float:
    if pnl.is_empty():
        return 1.0
    daily = (
        pnl.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("net_return").sum().alias("net_return"))
    )
    positive_total = _to_float(daily.filter(pl.col("net_return") > 0.0).get_column("net_return").sum())
    if positive_total <= 0.0:
        return 1.0
    return _to_float(daily.get_column("net_return").max()) / positive_total


def _select_validation_candidates(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    validation = [row for row in rows if row.get("period") == "validation" and int(row.get("trade_count", 0)) > 0]
    return sorted(
        validation,
        key=lambda row: (
            float(row.get("net_daily_sharpe", 0.0)),
            float(row.get("net_total_return", 0.0)),
            -abs(float(row.get("max_drawdown", 0.0))),
        ),
        reverse=True,
    )[:limit]


def _base_config(
    args: argparse.Namespace,
    *,
    cost_multiplier: float = 1.0,
    delay: int | None = None,
    fee_bps: float | None = None,
    half_spread_bps: float | None = None,
    slippage_bps: float | None = None,
    invert_signal: bool = False,
) -> BacktestConfig:
    return BacktestConfig(
        fee_bps=args.fee_bps if fee_bps is None else fee_bps,
        half_spread_bps=args.half_spread_bps if half_spread_bps is None else half_spread_bps,
        slippage_bps=args.slippage_bps if slippage_bps is None else slippage_bps,
        execution_delay_bars=args.execution_delay_bars if delay is None else delay,
        notional_usd=args.notional_usd,
        cost_multiplier=cost_multiplier,
        invert_signal=invert_signal,
    )


def _null_baseline_variants() -> list[StrategyVariant]:
    return [
        StrategyVariant(
            strategy_id="baseline_always_long",
            family="baseline",
            feature_set="always_long",
            parameters={"threshold": 0.0},
            horizon=1,
            entry_rule="always long",
            exit_rule="rebalance each bar",
            execution_assumption="one-bar delayed taker execution",
            cost_assumption="same configured cost model as candidate strategies",
            source="null_baseline",
        ),
        StrategyVariant(
            strategy_id="baseline_always_short",
            family="baseline",
            feature_set="always_short",
            parameters={"threshold": 0.0},
            horizon=1,
            entry_rule="always short",
            exit_rule="rebalance each bar",
            execution_assumption="one-bar delayed taker execution",
            cost_assumption="same configured cost model as candidate strategies",
            source="null_baseline",
        ),
        StrategyVariant(
            strategy_id="baseline_deterministic_random",
            family="baseline",
            feature_set="deterministic_random",
            parameters={"threshold": 0.0},
            horizon=1,
            entry_rule="deterministic pseudo-random sign from row index",
            exit_rule="rebalance each bar",
            execution_assumption="one-bar delayed taker execution",
            cost_assumption="same configured cost model as candidate strategies",
            source="null_baseline",
        ),
    ]


def _gate_row(
    row: dict[str, Any],
    *,
    pbo: float,
    monthly_net_mean: float,
    cost_2x_positive: bool,
    delay_positive: bool,
    best_day_concentration: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    net_sharpe = float(row.get("net_daily_sharpe", 0.0))
    net_return = float(row.get("net_total_return", 0.0))
    max_drawdown = abs(float(row.get("max_drawdown", 0.0)))
    calmar = net_return / max(max_drawdown, 1e-12)
    if net_sharpe < 1.0:
        reasons.append("net daily Sharpe below 1.0")
    if net_return <= 0.0:
        reasons.append("net total return not positive")
    if calmar <= 1.0:
        reasons.append("Calmar not above 1.0")
    if pbo >= 0.25:
        reasons.append("PBO not below 0.25")
    if monthly_net_mean < 0.05:
        reasons.append("average monthly net return below 5%")
    if not cost_2x_positive:
        reasons.append("not positive under 2x costs")
    if not delay_positive:
        reasons.append("not positive under one extra bar delay")
    if best_day_concentration > 0.50:
        reasons.append("more than half of positive PnL comes from one day")
    return not reasons, reasons


def _registry_frame(variants: list[StrategyVariant], periods_payload: dict[str, dict[str, str]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        row = variant.to_row()
        row["train_period"] = json.dumps(periods_payload["development"], sort_keys=True)
        row["validation_period"] = json.dumps(periods_payload["validation"], sort_keys=True)
        row["holdout_period"] = json.dumps(periods_payload["holdout"], sort_keys=True)
        row["pass_fail_status"] = "not_evaluated"
        rows.append(row)
    return pl.DataFrame(rows)


def main() -> int:
    args = parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    command = "PYTHONPATH=src uv run python " + " ".join(sys.argv)
    print(f"loading BTCUSDT 1m panel from {args.data_path}")
    frame = load_btcusdt_1m_panel(args.data_path, months=args.months, max_rows=args.max_rows)
    periods = build_chronological_periods(frame, timestamp_column="timestamp")
    periods_payload = periods.to_dict()
    manifest = dataset_manifest_from_frame(
        frame,
        dataset_id="vaquum/binance_btcusdt_1m_klines",
        source_path=args.data_path,
        timestamp_column="timestamp",
        known_limitations=[
            "BTCUSDT 1-minute OHLCV only; no true tick data, order book depth, news, funding, or broker fill feed.",
            "Bid/ask in per-trade audits is approximated from configured half-spread around close mid.",
            "Rule scores are not calibrated expected returns unless the family score is return-scaled.",
        ],
    )
    write_dataset_manifest(output_dir / "dataset_manifest.json", manifest)

    variants = generate_strategy_variants(target_count=args.target_count)
    registry = _registry_frame(variants, periods_payload)
    config = _base_config(args)
    blocks = _block_periods(frame, block_count=args.pbo_blocks)
    all_rows: list[dict[str, Any]] = []
    pbo_rows: list[dict[str, Any]] = []
    variants_by_id = {variant.strategy_id: variant for variant in variants}

    print(f"testing {len(variants)} variants over {frame.height:,} rows")
    for index, variant in enumerate(variants, start=1):
        result = run_variant_backtest(frame, variant, config=config)
        for period in [periods.development, periods.validation]:
            row = _summarize_period(result, variant, config=config, period=period)
            row["horizon"] = variant.horizon
            row["feature_set"] = variant.feature_set
            row["pass_gate"] = False
            all_rows.append(row)
        for block_index, block_period in enumerate(blocks):
            block_row = _summarize_period(result, variant, config=config, period=block_period)
            pbo_rows.append(
                {
                    "strategy_id": variant.strategy_id,
                    "block": block_index,
                    "net_sharpe": float(block_row.get("net_daily_sharpe", 0.0)),
                }
            )
        if index == 1 or index % 100 == 0 or index == len(variants):
            print(f"  tested {index}/{len(variants)}")
        del result

    pbo_scores = pl.DataFrame(pbo_rows)
    pbo_report = estimate_pbo(pbo_scores, score_column="net_sharpe", min_blocks=min(args.pbo_blocks, 6))
    validation_candidates = _select_validation_candidates(all_rows, limit=args.finalists)
    best_validation_sharpe = max((float(row.get("net_daily_sharpe", 0.0)) for row in validation_candidates), default=0.0)
    pbo_payload = pbo_report.to_dict()
    pbo_payload["multiple_testing"] = approximate_multiple_testing_payload(
        best_validation_sharpe,
        trial_count=len(variants),
        observations=max(frame.height // (24 * 60), 1),
    )
    pbo_payload["tested_strategy_variants"] = len(variants)
    pbo_scores.write_parquet(output_dir / "pbo_scores.parquet")
    _write_json(output_dir / "pbo_scores.json", {"rows": pbo_rows[:1000], "truncated": len(pbo_rows) > 1000})

    holdout_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    finalist_gate_reasons: dict[str, list[str]] = {}
    promoted = False
    print(f"evaluating {len(validation_candidates)} finalists on permanent holdout")
    for candidate in validation_candidates:
        strategy_id = str(candidate["strategy_id"])
        variant = variants_by_id[strategy_id]
        base_result = run_variant_backtest(frame, variant, config=config)
        holdout_pnl = _slice_timestamp(base_result.pnl, periods.holdout)
        holdout_trades = _slice_timestamp(base_result.trades, periods.holdout) if not base_result.trades.is_empty() else base_result.trades
        audit_path = output_dir / f"per_trade_audit_{strategy_id}.parquet"
        holdout_trades.write_parquet(audit_path)
        row = _summarize_period(base_result, variant, config=config, period=periods.holdout)
        row["horizon"] = variant.horizon
        row["feature_set"] = variant.feature_set
        row["monthly_net_mean"] = _monthly_net_mean(holdout_pnl)
        row["best_day_concentration"] = _best_day_concentration(holdout_pnl)

        stress_results: dict[str, dict[str, Any]] = {}
        for label, stress_config in [
            ("base", config),
            ("no_cost", _base_config(args, fee_bps=0.0, half_spread_bps=0.0, slippage_bps=0.0)),
            ("spread_only", _base_config(args, fee_bps=0.0, slippage_bps=0.0)),
            ("fee_only", _base_config(args, half_spread_bps=0.0, slippage_bps=0.0)),
            ("cost_2x", _base_config(args, cost_multiplier=2.0)),
            ("cost_3x", _base_config(args, cost_multiplier=3.0)),
            ("delay_plus_one", _base_config(args, delay=args.execution_delay_bars + 1)),
            ("inverted_signal", _base_config(args, invert_signal=True)),
        ]:
            stress_result = base_result if label == "base" else run_variant_backtest(frame, variant, config=stress_config)
            stress_row = _summarize_period(stress_result, variant, config=stress_config, period=periods.holdout)
            stress_row["stress"] = label
            stress_row["horizon"] = variant.horizon
            cost_rows.append(stress_row)
            stress_results[label] = stress_row

        pass_gate, reasons = _gate_row(
            row,
            pbo=float(pbo_report.pbo),
            monthly_net_mean=float(row["monthly_net_mean"]),
            cost_2x_positive=float(stress_results["cost_2x"].get("net_total_return", 0.0)) > 0.0,
            delay_positive=float(stress_results["delay_plus_one"].get("net_total_return", 0.0)) > 0.0,
            best_day_concentration=float(row["best_day_concentration"]),
        )
        row["pass_gate"] = pass_gate
        row["audit_path"] = str(audit_path)
        row["gate_reasons"] = "; ".join(reasons)
        finalist_gate_reasons[strategy_id] = reasons
        promoted = promoted or pass_gate
        holdout_rows.append(row)
        all_rows.append(row)

    for baseline in _null_baseline_variants():
        baseline_result = run_variant_backtest(frame, baseline, config=config)
        baseline_row = _summarize_period(baseline_result, baseline, config=config, period=periods.holdout)
        baseline_row["stress"] = f"null_{baseline.feature_set}"
        baseline_row["horizon"] = baseline.horizon
        cost_rows.append(baseline_row)

    best_candidates = validation_candidates
    failure_reasons: list[str] = []
    if not promoted:
        if not validation_candidates:
            failure_reasons.append("No validation candidate produced any trades.")
        else:
            best_holdout = max(holdout_rows, key=lambda row: float(row.get("net_daily_sharpe", 0.0)), default={})
            failure_reasons.append(
                "No finalist passed the predefined promotion gate "
                f"(best holdout strategy={best_holdout.get('strategy_id', 'n/a')}, "
                f"net_return={float(best_holdout.get('net_total_return', 0.0)):.6g}, "
                f"monthly_net_mean={float(best_holdout.get('monthly_net_mean', 0.0)):.6g}, "
                f"Sharpe={float(best_holdout.get('net_daily_sharpe', 0.0)):.6g}, "
                f"PBO={pbo_report.pbo:.6g})."
            )
            for strategy_id, reasons in finalist_gate_reasons.items():
                failure_reasons.append(f"{strategy_id}: {', '.join(reasons) if reasons else 'passed'}")

    _write_json(
        output_dir / "run_config.json",
        {
            "args": vars(args),
            "git_sha": _git_sha(),
            "run_id": run_id,
            "periods": periods_payload,
            "command": command,
            "rows": frame.height,
        },
    )
    write_research_outputs(
        output_dir=output_dir,
        registry=registry,
        all_backtests=pl.DataFrame(all_rows, infer_schema_length=max(len(all_rows), 1)),
        pbo_payload=pbo_payload,
        best_candidates=best_candidates,
        holdout_rows=holdout_rows,
        cost_sensitivity_rows=cost_rows,
        failure_reasons=failure_reasons,
        commands=[
            command,
            f"open {output_dir}",
            f"python - <<'PY'\nimport polars as pl\nprint(pl.read_parquet('{output_dir / 'all_backtests.parquet'}').head())\nPY",
        ],
    )
    print(f"wrote research artifacts to {output_dir}")
    if promoted:
        print("promotion gate passed for at least one finalist")
        return 0
    print("no strategy passed the promotion gate")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
