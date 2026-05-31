"""Train the signal_research triple-barrier meta-labeler walk-forward.

Example:
    PYTHONPATH=src uv run python scripts/train_signal_research_meta_label.py \\
        --data-root data/processed/strategy_benchmark \\
        --out experiments/signal_research/meta_label/latest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.signal_research.papers.triple_barrier import TripleBarrierConfig
from quant_research_stack.signal_research.training.meta_label_walk_forward import (
    MetaLabelWalkForwardConfig,
    train_meta_label_walk_forward,
    write_meta_label_walk_forward_artifacts,
)

console = Console()


def _load_panel(data_root: Path, symbols: list[str] | None) -> pl.DataFrame:
    files = sorted(Path(data_root).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {data_root}")
    frames = [pl.read_parquet(path) for path in files]
    panel = pl.concat(frames, how="diagonal_relaxed").sort(["symbol", "date"])
    if symbols:
        panel = panel.filter(pl.col("symbol").is_in(symbols))
    if panel.is_empty():
        raise ValueError("loaded panel is empty after symbol filtering")
    return panel


def _parse_symbols(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train triple-barrier meta-labeler with chronological walk-forward folds.")
    p.add_argument("--data-root", type=Path, default=Path("data/processed/strategy_benchmark"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--symbols", default=None, help="Optional comma-separated symbol filter, matching the parquet symbol column.")
    p.add_argument("--lookback-days", type=int, default=20)
    p.add_argument("--train-window-days", type=int, default=252)
    p.add_argument("--test-window-days", type=int, default=63)
    p.add_argument("--step-days", type=int, default=63)
    p.add_argument("--purge-days", type=int, default=20)
    p.add_argument("--min-train-events", type=int, default=200)
    p.add_argument("--random-forest-estimators", type=int, default=200)
    p.add_argument("--probability-threshold", type=float, default=0.55)
    p.add_argument("--cost-bps-one-way", type=float, default=1.0)
    p.add_argument("--vertical-barrier-days", type=int, default=20)
    p.add_argument("--profit-take-multiplier", type=float, default=1.5)
    p.add_argument("--stop-loss-multiplier", type=float, default=1.5)
    p.add_argument("--vol-estimator-window", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    panel = _load_panel(args.data_root, _parse_symbols(args.symbols))
    cfg = MetaLabelWalkForwardConfig(
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
    result = train_meta_label_walk_forward(panel=panel, config=cfg)
    written = write_meta_label_walk_forward_artifacts(result, output_dir=args.out)
    console.print(
        "[green]ok[/green] trained research-only meta-label walk-forward "
        f"folds={result.summary['fold_count']} trades={result.summary['trade_count']} "
        f"net={result.summary['net_total_return']:.6g} report={written['report']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
