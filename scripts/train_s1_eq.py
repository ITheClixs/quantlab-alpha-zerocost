"""Unified S1-EQ trainer CLI.

Usage:
    PYTHONPATH=src uv run python scripts/train_s1_eq.py \
        --config configs/alpha_eq.yaml --mode fast_v1 \
        --equity-root data/processed/equities \
        --experiments-root experiments/alpha_eq
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from rich.console import Console

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode
from quant_research_stack.alpha_eq.data.holdout import (
    HoldoutGate,
    assert_min_holdout_length,
    compute_holdout_dates,
)
from quant_research_stack.alpha_eq.data.loaders import EquityRootLoader
from quant_research_stack.alpha_eq.features.builder import (
    FeatureBuildConfig,
    build_features,
)
from quant_research_stack.alpha_eq.features.labels import build_labels
from quant_research_stack.alpha_eq.training.persist import persist_fast_v1_run

console = Console()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/alpha_eq.yaml")
    p.add_argument("--mode", default="fast_v1", choices=[m.value for m in TrainingMode])
    p.add_argument("--equity-root", default="data/processed/equities")
    p.add_argument("--experiments-root", default="experiments/alpha_eq")
    return p.parse_args()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _build_panel(loader: EquityRootLoader) -> pl.DataFrame:
    tradable = loader.load_tradable_prices()
    total_return = loader.load_total_return_prices()
    if "close_tr" in total_return.columns:
        panel = tradable.join(
            total_return.select(["date", "symbol", "close_tr"]),
            on=["date", "symbol"],
            how="left",
        )
    else:
        panel = tradable.with_columns(pl.col("close").alias("close_tr"))
    panel = panel.with_columns(pl.lit(True).alias("in_universe"))
    return panel


def _feature_build_config(config: AlphaEqConfig, panel: pl.DataFrame) -> FeatureBuildConfig:
    unique_dates = int(panel["date"].n_unique()) if "date" in panel.columns else 0
    max_lookback = max(unique_dates - 3, 1)
    horizons = tuple(h for h in config.features.momentum_horizons if h <= max_lookback) or (1,)
    windows = tuple(w for w in config.features.rolling_windows if w <= max_lookback) or (min(5, max_lookback),)
    micro_window = min(20, max_lookback)
    rank_columns = [
        "dollar_volume",
        "overnight_gap",
        "intraday_return",
    ]
    for horizon in (1, 5, 20):
        if horizon in horizons:
            rank_columns.append(f"log_return_{horizon}")
    if 20 in windows:
        rank_columns.append("realized_vol_20")
    rank_columns.extend([f"amihud_illiq_{micro_window}", f"close_location_{micro_window}"])
    return FeatureBuildConfig(
        momentum_horizons=horizons,
        vol_windows=windows,
        micro_window=micro_window,
        liquidity_window=micro_window,
        rank_columns=tuple(rank_columns),
        noise_seed=config.features.noise_seed,
        enable_meta_features=config.features.enable_meta_features,
    )


def _select_training_feature_cols(
    features: pl.DataFrame,
    candidate_cols: list[str],
    *,
    target: str,
) -> list[str]:
    labeled = features.drop_nulls(subset=[target])
    if labeled.is_empty():
        return candidate_cols
    usable = [column for column in candidate_cols if labeled.get_column(column).null_count() < labeled.height]
    if not labeled.drop_nulls(subset=[*usable, target]).is_empty():
        return usable
    strict = [column for column in usable if labeled.get_column(column).null_count() == 0]
    if strict:
        return strict
    return usable


def main() -> int:
    args = _parse_args()
    cfg_dict = yaml.safe_load(Path(args.config).read_text())
    cfg_dict["mode"] = args.mode
    config = AlphaEqConfig.model_validate(cfg_dict)

    np.random.seed(config.reproducibility.numpy_seed)

    loader = EquityRootLoader(root=Path(args.equity_root))
    panel = _build_panel(loader)

    sorted_dates = sorted(panel["date"].unique().to_list())
    dev_dates, hold_dates = compute_holdout_dates(
        sorted_dates, fraction=config.data.permanent_holdout_fraction
    )
    if len(hold_dates) >= config.data.min_holdout_trading_days:
        assert_min_holdout_length(
            hold_dates, min_trading_days=config.data.min_holdout_trading_days
        )

    gate = HoldoutGate(holdout_dates=hold_dates)
    dev_panel = gate.filter_for_caller(
        panel.filter(pl.col("date").is_in(dev_dates)),
        caller="training",
    )

    features = build_features(panel=dev_panel, config=_feature_build_config(config, dev_panel))
    vol_cols = sorted(c for c in features.columns if c.startswith("realized_vol_"))
    vol_col = "realized_vol_20" if "realized_vol_20" in features.columns else vol_cols[0]
    features = build_labels(
        features, close_tr="close_tr", vol_col=vol_col, universe_col="in_universe"
    )

    candidate_feature_cols = [
        c for c in features.columns
        if c.startswith(
            (
                "log_return_", "realized_vol_", "amihud_illiq_", "roll_spread_",
                "kyle_proxy_signed_volume_", "overnight_gap", "intraday_return",
                "close_location_", "dollar_volume", "log_dollar_volume_",
                "volume_zscore_", "rank_", "spy_log_return_", "spy_realized_vol_",
                "vix_close", "cross_sectional_", "gaussian_noise_",
            )
        )
        and c != "vix_is_proxy"  # boolean flag column, not a feature
    ]
    feature_cols = _select_training_feature_cols(features, candidate_feature_cols, target="y_xs")

    run_dir = Path(args.experiments_root) / _run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    persist_fast_v1_run(
        run_dir=run_dir,
        config=config,
        feature_cols=feature_cols,
        dev_panel=features,
        target="y_xs",
    )

    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["git_sha"] = _git_sha()
    eq_manifest_path = Path(args.equity_root) / "_manifest.json"
    if eq_manifest_path.exists():
        meta["data_manifest_sha256"] = hashlib.sha256(eq_manifest_path.read_bytes()).hexdigest()
    meta["build_command_line"] = " ".join(sys.argv)
    meta["python_version"] = platform.python_version()
    meta["holdout_dates_count"] = len(hold_dates)
    meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2))

    (run_dir / "holdout_dates.json").write_text(
        json.dumps([str(d) for d in hold_dates], sort_keys=True)
    )

    console.print(f"[bold green]Run persisted:[/bold green] {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
