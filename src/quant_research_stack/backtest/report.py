from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quant_research_stack.backtest.runner import BacktestResult


@dataclass(frozen=True)
class BacktestReport:
    root: Path

    def __post_init__(self) -> None:
        Path(self.root).mkdir(parents=True, exist_ok=True)

    def write(self, result: BacktestResult, *, run_id: str, strategy_name: str) -> None:
        root = Path(self.root)
        metrics = dict(result.metrics)
        metrics["run_id"] = run_id
        metrics["strategy_name"] = strategy_name
        (root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str))
        if result.fills:
            fills_df = pl.DataFrame([f.model_dump(mode="json") for f in result.fills])
        else:
            fills_df = pl.DataFrame({
                "client_order_id": [], "fill_id": [], "symbol": [],
                "side": [], "price": [], "quantity": [], "timestamp_utc": [], "commission": [],
            })
        fills_df.write_parquet(root / "fills.parquet", compression="zstd")
        result.equity_curve.write_parquet(root / "equity_curve.parquet", compression="zstd")
        self._plot_equity(result, root / "equity_curve.png")
        self._plot_drawdown(result, root / "drawdown.png")
        (root / "report.md").write_text(self._markdown(result, run_id, strategy_name))

    def _plot_equity(self, result: BacktestResult, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ts = result.equity_curve["timestamp_utc"].to_list()
        eq = result.equity_curve["equity"].to_list()
        ax.plot(ts, eq)
        ax.set_title("Equity Curve")
        ax.set_xlabel("time UTC")
        ax.set_ylabel("equity")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _plot_drawdown(self, result: BacktestResult, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 3))
        eq = result.equity_curve["equity"].to_list()
        peak = float("-inf")
        dd = []
        for v in eq:
            if v > peak:
                peak = v
            dd.append(((v - peak) / peak) if peak > 0 else 0.0)
        ax.fill_between(range(len(dd)), dd, 0.0, alpha=0.4)
        ax.set_title("Drawdown")
        ax.set_xlabel("step")
        ax.set_ylabel("fractional drawdown")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _markdown(self, result: BacktestResult, run_id: str, strategy_name: str) -> str:
        lines = [
            f"# Backtest report `{run_id}`",
            "",
            f"Strategy: `{strategy_name}`",
            "",
            "## Metrics",
            "",
            "| metric | value |",
            "|---|---|",
        ]
        for key, value in sorted(result.metrics.items()):
            lines.append(f"| `{key}` | `{value}` |")
        lines += [
            "",
            "## Equity curve",
            "",
            "![equity_curve](equity_curve.png)",
            "",
            "## Drawdown",
            "",
            "![drawdown](drawdown.png)",
            "",
            "## Notes",
            "",
            "- This backtest uses fixed-bps slippage and commission. L2 order-book impact",
            "  is not modeled in S3; defer large-size studies to S3.3.",
            "- `not_investment_advice: true` — every artifact is research output only.",
        ]
        return "\n".join(lines) + "\n"
