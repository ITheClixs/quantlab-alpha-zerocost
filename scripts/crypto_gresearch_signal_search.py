from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant_research_stack.crypto_research.data import dataset_manifest_from_frame, write_dataset_manifest
from quant_research_stack.crypto_research.gresearch import (
    FEATURE_COLUMNS,
    PortfolioBacktestResult,
    build_gresearch_features,
    chronological_split,
    portfolio_backtest,
)
from quant_research_stack.crypto_research.pbo import approximate_multiple_testing_payload, estimate_pbo
from quant_research_stack.crypto_research.reports import write_research_outputs

DEFAULT_DATA_DIR = Path("data/raw/kaggle/datasets/bariscan07__g-research-crypto-forecasting-dataset")
DEFAULT_TRAIN = DEFAULT_DATA_DIR / "train.csv"
DEFAULT_SUPPLEMENTAL = DEFAULT_DATA_DIR / "supplemental_train.csv"
DEFAULT_ASSET_DETAILS = DEFAULT_DATA_DIR / "asset_details.csv"

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "core": ("ret1", "ret5", "ret15", "vwap_dev", "log_volume", "log_count", "vol15"),
    "trend": ("ret1", "ret5", "ret15", "ret60", "vol15", "vol60"),
    "liquidity_proxy": ("ret1", "vwap_dev", "log_volume", "log_count", "vol15", "vol60"),
    "full": FEATURE_COLUMNS,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    factory: Callable[[], Any]


@dataclass(frozen=True)
class GResearchVariant:
    strategy_id: str
    family: str
    model_name: str
    feature_set: str
    label_name: str
    horizon_minutes: int
    threshold_quantile: float
    threshold: float
    side_policy: str
    cost_multiplier: float

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_columns"] = json.dumps(FEATURE_SETS[self.feature_set])
        payload["entry_rule"] = (
            "trade sign(model_prediction) only when abs(prediction) exceeds a development-set "
            "quantile threshold"
        )
        payload["exit_rule"] = f"close after {self.horizon_minutes} minutes on non-overlapping rebalance bars"
        payload["execution_assumption"] = "taker execution proxy using fee + slippage; no maker fill assumption"
        payload["cost_assumption"] = f"round-trip fee/slippage cost multiplier={self.cost_multiplier:g}"
        payload["parameters_json"] = json.dumps(
            {
                "threshold_quantile": self.threshold_quantile,
                "threshold": self.threshold,
                "side_policy": self.side_policy,
                "cost_multiplier": self.cost_multiplier,
            },
            sort_keys=True,
        )
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a chronological G-Research crypto signal search with PBO and holdout gates."
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--supplemental-path", type=Path, default=DEFAULT_SUPPLEMENTAL)
    parser.add_argument("--asset-details-path", type=Path, default=DEFAULT_ASSET_DETAILS)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/crypto_gresearch_signal_search"))
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=800_000)
    parser.add_argument("--horizons", type=str, default="15")
    parser.add_argument("--sample-interval-minutes", type=int, default=15)
    parser.add_argument("--target-count", type=int, default=1500)
    parser.add_argument("--pbo-blocks", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=10)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--threshold-quantiles", type=str, default="0.80,0.85,0.90,0.95,0.975,0.99")
    parser.add_argument("--feature-sets", type=str, default="core,trend,liquidity_proxy,full")
    parser.add_argument("--model-set", choices=["linear", "linear_plus_hgb"], default="linear")
    parser.add_argument("--min-validation-trades", type=int, default=100)
    parser.add_argument("--promotion-min-trades", type=int, default=100)
    parser.add_argument("--promotion-sharpe", type=float, default=5.0)
    parser.add_argument("--promotion-monthly-net", type=float, default=0.10)
    return parser.parse_args()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _parse_csv_floats(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def _parse_feature_sets(raw: str) -> list[str]:
    names = [value.strip() for value in raw.split(",") if value.strip()]
    missing = [name for name in names if name not in FEATURE_SETS]
    if missing:
        raise ValueError(f"unknown feature sets: {missing}")
    return names


def _scalar_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _scalar_iso(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _ridge_factory(alpha: float) -> Callable[[], Any]:
    def factory() -> Ridge:
        return Ridge(alpha=alpha)

    return factory


def _elastic_factory(alpha: float, l1: float) -> Callable[[], Any]:
    def factory() -> ElasticNet:
        return ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=3000)

    return factory


def _sgd_factory(alpha: float) -> Callable[[], Any]:
    def factory() -> SGDRegressor:
        return SGDRegressor(
            alpha=alpha,
            loss="huber",
            penalty="l2",
            max_iter=1000,
            tol=1e-4,
            random_state=17,
        )

    return factory


def _huber_factory() -> HuberRegressor:
    return HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500)


def _hgb_factory(learning_rate: float, l2: float) -> Callable[[], Any]:
    def factory() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            l2_regularization=l2,
            max_leaf_nodes=31,
            max_iter=120,
            random_state=17,
        )

    return factory


def _model_specs(model_set: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec(f"ridge_a{alpha:g}", "linear_ridge", _ridge_factory(alpha))
        for alpha in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    ]
    specs.extend(
        [
            ModelSpec(
                f"elastic_a{alpha:g}_l1{l1:g}",
                "linear_elasticnet",
                _elastic_factory(alpha, l1),
            )
            for alpha in (0.0001, 0.001, 0.01)
            for l1 in (0.05, 0.20)
        ]
    )
    specs.extend(
        [
            ModelSpec(
                f"sgd_l2_a{alpha:g}",
                "linear_sgd",
                _sgd_factory(alpha),
            )
            for alpha in (0.000001, 0.00001, 0.0001, 0.001)
        ]
    )
    specs.append(
        ModelSpec(
            "huber_eps1.35",
            "linear_huber",
            _huber_factory,
        )
    )
    if model_set == "linear_plus_hgb":
        specs.extend(
            [
                ModelSpec(
                    f"hgb_lr{learning_rate:g}_l2{l2:g}",
                    "hist_gradient_boosting",
                    _hgb_factory(learning_rate, l2),
                )
                for learning_rate in (0.03, 0.06)
                for l2 in (0.0, 0.1)
            ]
        )
    return specs


def _load_asset_names(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame({"Asset_ID": [], "symbol": []}, schema={"Asset_ID": pl.Int64, "symbol": pl.String})
    return pl.read_csv(path).select(
        [
            pl.col("Asset_ID").cast(pl.Int64),
            pl.col("Asset_Name").cast(pl.String).alias("symbol"),
        ]
    )


def _load_gresearch_frame(
    *,
    train_path: Path,
    supplemental_path: Path,
    asset_details_path: Path,
    months: int,
    max_rows: int | None,
) -> pl.DataFrame:
    paths = [path for path in [train_path, supplemental_path] if path.exists()]
    if not paths:
        raise FileNotFoundError("no G-Research train/supplemental CSV files found")
    required = ["timestamp", "Asset_ID", "Count", "Open", "High", "Low", "Close", "Volume", "VWAP", "Target"]
    scans = [pl.scan_csv(path).select(required) for path in paths]
    lazy = pl.concat(scans).filter(pl.col("Target").is_not_null())
    max_timestamp = int(_scalar_float(lazy.select(pl.col("timestamp").max()).collect().item()))
    start_timestamp = max_timestamp - int(months * 30.4375 * 24 * 60 * 60)
    frame = lazy.filter(pl.col("timestamp") >= start_timestamp).collect().sort(["timestamp", "Asset_ID"])
    if max_rows is not None and max_rows > 0 and frame.height > max_rows:
        frame = frame.tail(max_rows)
    asset_names = _load_asset_names(asset_details_path)
    if not asset_names.is_empty():
        frame = frame.join(asset_names, on="Asset_ID", how="left")
    return frame.with_columns(
        [
            pl.from_epoch("timestamp", time_unit="s").alias("timestamp_dt"),
            pl.col("Asset_ID").cast(pl.Int64),
        ]
    )


def _thin_by_timestamp(frame: pl.DataFrame, *, interval_minutes: int) -> pl.DataFrame:
    if interval_minutes <= 1:
        return frame
    first_timestamp = int(_scalar_float(frame.get_column("timestamp").min()))
    seconds = interval_minutes * 60
    return frame.filter(((pl.col("timestamp") - first_timestamp) % seconds) == 0)


def _train_frame(frame: pl.DataFrame, *, max_train_rows: int) -> pl.DataFrame:
    if max_train_rows > 0 and frame.height > max_train_rows:
        return frame.tail(max_train_rows)
    return frame


def _fit_predict_frames(
    *,
    spec: ModelSpec,
    feature_columns: tuple[str, ...],
    label_name: str,
    development: pl.DataFrame,
    validation: pl.DataFrame,
    holdout: pl.DataFrame,
    max_train_rows: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train = _train_frame(development.drop_nulls([*feature_columns, label_name]), max_train_rows=max_train_rows)
    if train.height < 100:
        raise ValueError(f"not enough training rows for {spec.name}/{label_name}")
    estimator = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), spec.factory())
    x_train = train.select(feature_columns).to_numpy()
    y_train = train.get_column(label_name).to_numpy().astype(np.float64)
    estimator.fit(x_train, y_train)

    def predict(frame: pl.DataFrame) -> pl.DataFrame:
        clean = frame.drop_nulls([*feature_columns])
        if clean.is_empty():
            return clean.with_columns(pl.lit(None).cast(pl.Float64).alias("prediction"))
        predictions = estimator.predict(clean.select(feature_columns).to_numpy())
        return clean.with_columns(pl.Series("prediction", predictions.astype(np.float64)))

    return predict(development), predict(validation), predict(holdout)


def _threshold_from_development(predictions: pl.DataFrame, quantile: float) -> float:
    if predictions.is_empty():
        return math.inf
    value = predictions.select(pl.col("prediction").abs().quantile(quantile)).item()
    if value is None or not math.isfinite(float(value)):
        return math.inf
    return float(value)


def _generate_variants(
    *,
    horizons: list[int],
    feature_sets: list[str],
    model_specs: list[ModelSpec],
    threshold_quantiles: list[float],
    target_count: int,
) -> list[GResearchVariant]:
    variants: list[GResearchVariant] = []
    base_groups = [
        (horizon, feature_set, spec)
        for horizon in horizons
        for feature_set in feature_sets
        for spec in model_specs
    ]
    for quantile in threshold_quantiles:
        for side_policy in ["both", "long_only", "short_only"]:
            for horizon, feature_set, spec in base_groups:
                strategy_id = (
                    f"gresearch_{spec.name}_{feature_set}_h{horizon}_"
                    f"q{int(quantile * 1000):03d}_{side_policy}"
                )
                variants.append(
                    GResearchVariant(
                        strategy_id=strategy_id,
                        family=spec.family,
                        model_name=spec.name,
                        feature_set=feature_set,
                        label_name="Target",
                        horizon_minutes=horizon,
                        threshold_quantile=quantile,
                        threshold=math.nan,
                        side_policy=side_policy,
                        cost_multiplier=1.0,
                    )
                )
                if len(variants) >= target_count:
                    return variants
    return variants


def _registry_frame(
    variants: list[GResearchVariant],
    periods_payload: dict[str, dict[str, str]],
) -> pl.DataFrame:
    rows = []
    for variant in variants:
        row = variant.to_row()
        row["train_period"] = json.dumps(periods_payload["development"], sort_keys=True)
        row["validation_period"] = json.dumps(periods_payload["validation"], sort_keys=True)
        row["holdout_period"] = json.dumps(periods_payload["holdout"], sort_keys=True)
        row["pass_fail_status"] = "not_evaluated"
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=max(len(rows), 1)) if rows else pl.DataFrame()


def _period_payload(split: Any) -> dict[str, dict[str, str]]:
    def period(frame: pl.DataFrame) -> dict[str, str]:
        timestamps = frame.get_column("timestamp_dt")
        return {"start": _scalar_iso(timestamps.min()), "end": _scalar_iso(timestamps.max())}

    return {
        "development": period(split.development),
        "validation": period(split.validation),
        "holdout": period(split.holdout),
    }


def _best_day_concentration(bars: pl.DataFrame) -> float:
    if bars.is_empty():
        return 1.0
    daily = (
        bars.with_columns(pl.from_epoch("timestamp", time_unit="s").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("net_return").sum().alias("net_return"))
    )
    positive_total = float(daily.filter(pl.col("net_return") > 0.0).get_column("net_return").sum() or 0.0)
    if positive_total <= 0.0:
        return 1.0
    return _scalar_float(daily.get_column("net_return").max()) / positive_total


def _result_row(
    *,
    variant: GResearchVariant,
    period: str,
    result: PortfolioBacktestResult,
) -> dict[str, Any]:
    metrics = dict(result.metrics)
    row: dict[str, Any] = {
        "strategy_id": variant.strategy_id,
        "family": variant.family,
        "model_name": variant.model_name,
        "feature_set": variant.feature_set,
        "label_name": variant.label_name,
        "horizon": variant.horizon_minutes,
        "threshold_quantile": variant.threshold_quantile,
        "threshold": variant.threshold,
        "side_policy": variant.side_policy,
        "cost_multiplier": variant.cost_multiplier,
        "period": period,
        "net_total_return": float(metrics.get("net_total_return", 0.0)),
        "gross_total_return": float(metrics.get("gross_total_return", 0.0)),
        "average_monthly_net_return": float(metrics.get("average_monthly_net_return", 0.0)),
        "net_daily_sharpe": float(metrics.get("net_sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "trade_count": int(metrics.get("trade_count", 0)),
        "bar_count": int(metrics.get("bar_count", 0)),
        "hit_rate": float(metrics.get("hit_rate", 0.0)),
        "gross_hit_rate": float(metrics.get("gross_hit_rate", 0.0)),
        "avg_net_return": float(metrics.get("avg_net_return", 0.0)),
        "pass_gate": False,
    }
    return row


def _gate_row(
    row: dict[str, Any],
    *,
    pbo: float,
    cost_2x_positive: bool,
    delay_positive: bool,
    best_day_concentration: float,
    promotion_sharpe: float,
    promotion_monthly_net: float,
    promotion_min_trades: int,
    null_baseline_dominates: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    net_sharpe = float(row.get("net_daily_sharpe", 0.0))
    net_return = float(row.get("net_total_return", 0.0))
    monthly = float(row.get("average_monthly_net_return", 0.0))
    max_drawdown = abs(float(row.get("max_drawdown", 0.0)))
    trade_count = int(row.get("trade_count", 0))
    calmar = net_return / max(max_drawdown, 1e-12)
    if net_sharpe < promotion_sharpe:
        reasons.append(f"net daily Sharpe below {promotion_sharpe:g}")
    if net_return <= 0.0:
        reasons.append("net total return not positive")
    if monthly < promotion_monthly_net:
        reasons.append(f"average monthly net return below {promotion_monthly_net:.1%}")
    if calmar <= 1.0:
        reasons.append("Calmar not above 1.0")
    if trade_count < promotion_min_trades:
        reasons.append(f"holdout trade count below {promotion_min_trades}")
    if pbo >= 0.25:
        reasons.append("PBO not below 0.25")
    if not cost_2x_positive:
        reasons.append("not positive under 2x costs")
    if not delay_positive:
        reasons.append("not positive under one-horizon execution delay")
    if best_day_concentration > 0.50:
        reasons.append("more than half of positive PnL comes from one day")
    if null_baseline_dominates:
        reasons.append("null baseline matched or beat candidate")
    return not reasons, reasons


def _select_validation_candidates(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    min_trades: int,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("period") == "validation" and int(row.get("trade_count", 0)) >= min_trades
        and float(row.get("delay_net_total_return", 0.0)) > 0.0
    ]
    return sorted(
        candidates,
        key=lambda row: (
            float(row.get("net_daily_sharpe", 0.0)),
            float(row.get("average_monthly_net_return", 0.0)),
            float(row.get("net_total_return", 0.0)),
        ),
        reverse=True,
    )[:limit]


def _subset_by_timestamps(frame: pl.DataFrame, timestamps: set[int]) -> pl.DataFrame:
    if not timestamps:
        return frame.head(0)
    return frame.filter(pl.col("timestamp").is_in(list(timestamps)))


def _chronological_block_frames(frame: pl.DataFrame, *, block_count: int) -> list[pl.DataFrame]:
    timestamps = sorted(int(value) for value in frame.get_column("timestamp").unique().to_list())
    if len(timestamps) < block_count:
        raise ValueError("not enough timestamps for PBO blocks")
    block_size = len(timestamps) // block_count
    blocks: list[pl.DataFrame] = []
    for index in range(block_count):
        start = index * block_size
        end = len(timestamps) if index == block_count - 1 else (index + 1) * block_size
        blocks.append(_subset_by_timestamps(frame, set(timestamps[start:end])))
    return blocks


def _evaluate_variant(
    frame: pl.DataFrame,
    *,
    variant: GResearchVariant,
    fee_bps: float,
    slippage_bps: float,
    prediction_column: str = "prediction",
    threshold: float | None = None,
    cost_multiplier: float | None = None,
) -> PortfolioBacktestResult:
    return portfolio_backtest(
        frame,
        threshold=variant.threshold if threshold is None else threshold,
        horizon_minutes=variant.horizon_minutes,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        prediction_column=prediction_column,
        side_policy=variant.side_policy,
        cost_multiplier=variant.cost_multiplier if cost_multiplier is None else cost_multiplier,
    )


def _delayed_execution_frame(frame: pl.DataFrame, *, horizon_minutes: int) -> pl.DataFrame:
    target_column = f"future_return_{horizon_minutes}"
    if target_column not in frame.columns:
        raise ValueError(f"missing delayed-execution target column: {target_column}")
    return frame.sort(["Asset_ID", "timestamp"]).with_columns(
        pl.col(target_column).shift(-horizon_minutes).over("Asset_ID").alias(target_column)
    )


def _null_baseline_results(
    frame: pl.DataFrame,
    *,
    variant: GResearchVariant,
    fee_bps: float,
    slippage_bps: float,
) -> list[dict[str, Any]]:
    baseline_frames = {
        "always_long": frame.with_columns(pl.lit(1.0).alias("prediction")),
        "always_short": frame.with_columns(pl.lit(-1.0).alias("prediction")),
        "deterministic_random": frame.with_columns(
            pl.when(((pl.col("timestamp") + (pl.col("Asset_ID") * 9973)) % 2) == 0)
            .then(1.0)
            .otherwise(-1.0)
            .alias("prediction")
        ),
    }
    rows: list[dict[str, Any]] = []
    for label, baseline_frame in baseline_frames.items():
        baseline_variant = GResearchVariant(
            strategy_id=f"null_{label}_h{variant.horizon_minutes}",
            family="null_baseline",
            model_name=label,
            feature_set=variant.feature_set,
            label_name=variant.label_name,
            horizon_minutes=variant.horizon_minutes,
            threshold_quantile=0.0,
            threshold=0.0,
            side_policy="both",
            cost_multiplier=variant.cost_multiplier,
        )
        result = portfolio_backtest(
            baseline_frame,
            threshold=0.0,
            horizon_minutes=variant.horizon_minutes,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            side_policy="both",
            cost_multiplier=variant.cost_multiplier,
        )
        row = _result_row(variant=baseline_variant, period="holdout", result=result)
        row["stress"] = f"null_{label}"
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    command = "PYTHONPATH=src uv run python " + " ".join(sys.argv)

    horizons = _parse_csv_ints(args.horizons)
    threshold_quantiles = _parse_csv_floats(args.threshold_quantiles)
    feature_sets = _parse_feature_sets(args.feature_sets)
    model_specs = _model_specs(args.model_set)

    print("loading G-Research crypto CSV data")
    raw = _load_gresearch_frame(
        train_path=args.train_path,
        supplemental_path=args.supplemental_path,
        asset_details_path=args.asset_details_path,
        months=args.months,
        max_rows=args.max_rows,
    )
    write_dataset_manifest(
        output_dir / "dataset_manifest.json",
        dataset_manifest_from_frame(
            raw,
            dataset_id="kaggle/g-research-crypto-forecasting",
            source_path=args.train_path,
            timestamp_column="timestamp_dt",
            timestamp_semantics=(
                "one-minute crypto bars; features use current and past bar fields only; Target and "
                "future_return columns are labels unavailable at signal time"
            ),
            known_limitations=[
                "Dataset is historical Kaggle competition data ending in January 2022, not live 2026 market data.",
                "Asset universe is the fixed competition universe of 14 crypto assets.",
                "No order book, funding, news, or true exchange fill feed is included.",
                "Backtest uses close-to-future-close returns with taker cost proxies.",
            ],
        ),
    )

    base_variants = _generate_variants(
        horizons=horizons,
        feature_sets=feature_sets,
        model_specs=model_specs,
        threshold_quantiles=threshold_quantiles,
        target_count=args.target_count,
    )
    all_rows: list[dict[str, Any]] = []
    pbo_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    predictions_by_strategy: dict[str, tuple[GResearchVariant, pl.DataFrame, pl.DataFrame, pl.DataFrame]] = {}
    periods_payload: dict[str, dict[str, str]] | None = None

    print(f"testing up to {len(base_variants)} G-Research variants")
    variant_groups: dict[tuple[int, str, str], list[GResearchVariant]] = {}
    specs_by_name = {spec.name: spec for spec in model_specs}
    for variant in base_variants:
        variant_groups.setdefault((variant.horizon_minutes, variant.feature_set, variant.model_name), []).append(variant)

    featured_by_horizon: dict[int, pl.DataFrame] = {}
    split_cache: dict[tuple[int, str], Any] = {}
    completed_groups = 0
    for (horizon, feature_set, model_name), variants in variant_groups.items():
        print(f"  fitting {model_name}/{feature_set}/h{horizon} for {len(variants)} variants")
        if horizon not in featured_by_horizon:
            featured_by_horizon[horizon] = _thin_by_timestamp(
                build_gresearch_features(raw, horizon_minutes=horizon),
                interval_minutes=max(args.sample_interval_minutes, horizon),
            )
        split_key = (horizon, feature_set)
        if split_key not in split_cache:
            featured = featured_by_horizon[horizon].drop_nulls(
                [*FEATURE_SETS[feature_set], "Target", f"future_return_{horizon}"]
            )
            split_cache[split_key] = chronological_split(featured)
        split = split_cache[split_key]
        if periods_payload is None:
            periods_payload = _period_payload(split)
        spec = specs_by_name[model_name]
        dev_pred, val_pred, hold_pred = _fit_predict_frames(
            spec=spec,
            feature_columns=FEATURE_SETS[feature_set],
            label_name="Target",
            development=split.development,
            validation=split.validation,
            holdout=split.holdout,
            max_train_rows=args.max_train_rows,
        )
        prediction_pbo_blocks = _chronological_block_frames(
            pl.concat([dev_pred, val_pred], how="vertical"),
            block_count=args.pbo_blocks,
        )
        for variant in variants:
            threshold = _threshold_from_development(dev_pred, variant.threshold_quantile)
            variant = GResearchVariant(**{**asdict(variant), "threshold": threshold})
            dev_result = _evaluate_variant(dev_pred, variant=variant, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
            val_result = _evaluate_variant(val_pred, variant=variant, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
            val_delay_result = _evaluate_variant(
                _delayed_execution_frame(val_pred, horizon_minutes=variant.horizon_minutes),
                variant=variant,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
            )
            all_rows.append(_result_row(variant=variant, period="development", result=dev_result))
            val_row = _result_row(variant=variant, period="validation", result=val_result)
            val_row["delay_net_total_return"] = float(val_delay_result.metrics.get("net_total_return", 0.0))
            val_row["delay_net_daily_sharpe"] = float(val_delay_result.metrics.get("net_sharpe", 0.0))
            all_rows.append(val_row)
            for block_index, block in enumerate(prediction_pbo_blocks):
                block_result = _evaluate_variant(
                    block,
                    variant=variant,
                    fee_bps=args.fee_bps,
                    slippage_bps=args.slippage_bps,
                )
                pbo_rows.append(
                    {
                        "strategy_id": variant.strategy_id,
                        "block": block_index,
                        "net_sharpe": float(block_result.metrics.get("net_sharpe", 0.0)),
                    }
                )
            predictions_by_strategy[variant.strategy_id] = (variant, dev_pred, val_pred, hold_pred)
        completed_groups += 1
        if completed_groups == 1 or completed_groups % 10 == 0 or completed_groups == len(variant_groups):
            print(f"  completed {completed_groups}/{len(variant_groups)} prediction groups")

    if periods_payload is None:
        raise RuntimeError("no variants were evaluated")

    pbo_scores = pl.DataFrame(pbo_rows, infer_schema_length=max(len(pbo_rows), 1))
    pbo_report = estimate_pbo(pbo_scores, score_column="net_sharpe", min_blocks=min(args.pbo_blocks, 6))
    validation_candidates = _select_validation_candidates(
        all_rows,
        limit=args.finalists,
        min_trades=args.min_validation_trades,
    )
    best_validation_sharpe = max((float(row.get("net_daily_sharpe", 0.0)) for row in validation_candidates), default=0.0)
    pbo_payload = pbo_report.to_dict()
    pbo_payload["multiple_testing"] = approximate_multiple_testing_payload(
        best_validation_sharpe,
        trial_count=len(base_variants),
        observations=max(raw.height // (24 * 60 * 14), 1),
    )
    pbo_payload["tested_strategy_variants"] = len(base_variants)
    pbo_payload["model_set"] = args.model_set
    pbo_payload["promotion_sharpe"] = args.promotion_sharpe
    pbo_payload["promotion_monthly_net"] = args.promotion_monthly_net
    pbo_payload["min_validation_trades"] = args.min_validation_trades
    pbo_payload["promotion_min_trades"] = args.promotion_min_trades
    pbo_scores.write_parquet(output_dir / "pbo_scores.parquet")

    promoted = False
    finalist_reasons: dict[str, list[str]] = {}
    print(f"evaluating {len(validation_candidates)} validation-selected finalists on permanent holdout")
    for candidate in validation_candidates:
        strategy_id = str(candidate["strategy_id"])
        variant, _dev_pred, _val_pred, hold_pred = predictions_by_strategy[strategy_id]
        base_result = _evaluate_variant(hold_pred, variant=variant, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
        audit_path = output_dir / f"per_trade_audit_{strategy_id}.parquet"
        base_result.trades.write_parquet(audit_path)
        row = _result_row(variant=variant, period="holdout", result=base_result)
        row["audit_path"] = str(audit_path)
        row["best_day_concentration"] = _best_day_concentration(base_result.bars)
        baseline_rows = _null_baseline_results(
            hold_pred,
            variant=variant,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )
        cost_rows.extend(baseline_rows)
        null_baseline_dominates = any(
            float(baseline.get("net_total_return", 0.0)) >= float(row.get("net_total_return", 0.0))
            or float(baseline.get("net_daily_sharpe", 0.0)) >= float(row.get("net_daily_sharpe", 0.0))
            for baseline in baseline_rows
        )

        stress_results: dict[str, dict[str, Any]] = {}
        for stress, fee_bps, slippage_bps, prediction_column, cost_multiplier in [
            ("base", args.fee_bps, args.slippage_bps, "prediction", variant.cost_multiplier),
            ("no_cost", 0.0, 0.0, "prediction", 0.0),
            ("spread_fee_only", args.fee_bps, 0.0, "prediction", variant.cost_multiplier),
            ("slippage_only", 0.0, args.slippage_bps, "prediction", variant.cost_multiplier),
            ("cost_2x", args.fee_bps, args.slippage_bps, "prediction", 2.0),
            ("cost_3x", args.fee_bps, args.slippage_bps, "prediction", 3.0),
            ("delay_one_horizon", args.fee_bps, args.slippage_bps, "prediction", variant.cost_multiplier),
            ("inverted_signal", args.fee_bps, args.slippage_bps, "inverted_prediction", variant.cost_multiplier),
        ]:
            stress_frame = hold_pred
            if stress == "delay_one_horizon":
                stress_frame = _delayed_execution_frame(hold_pred, horizon_minutes=variant.horizon_minutes)
            if prediction_column == "inverted_prediction":
                stress_frame = hold_pred.with_columns((-pl.col("prediction")).alias("inverted_prediction"))
            stress_result = portfolio_backtest(
                stress_frame,
                threshold=variant.threshold,
                horizon_minutes=variant.horizon_minutes,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                prediction_column=prediction_column,
                side_policy=variant.side_policy,
                cost_multiplier=cost_multiplier,
            )
            stress_row = _result_row(variant=variant, period="holdout", result=stress_result)
            stress_row["stress"] = stress
            cost_rows.append(stress_row)
            stress_results[stress] = stress_row

        pass_gate, reasons = _gate_row(
            row,
            pbo=float(pbo_report.pbo),
            cost_2x_positive=float(stress_results["cost_2x"].get("net_total_return", 0.0)) > 0.0,
            delay_positive=float(stress_results["delay_one_horizon"].get("net_total_return", 0.0)) > 0.0,
            best_day_concentration=float(row["best_day_concentration"]),
            promotion_sharpe=args.promotion_sharpe,
            promotion_monthly_net=args.promotion_monthly_net,
            promotion_min_trades=args.promotion_min_trades,
            null_baseline_dominates=null_baseline_dominates,
        )
        row["pass_gate"] = pass_gate
        row["gate_reasons"] = "; ".join(reasons)
        finalist_reasons[strategy_id] = reasons
        promoted = promoted or pass_gate
        holdout_rows.append(row)
        all_rows.append(row)

    failure_reasons: list[str] = []
    if not promoted:
        if not validation_candidates:
            min_trade_validation = [
                row
                for row in all_rows
                if row.get("period") == "validation" and int(row.get("trade_count", 0)) >= args.min_validation_trades
            ]
            delay_survivors = [
                row for row in min_trade_validation if float(row.get("delay_net_total_return", 0.0)) > 0.0
            ]
            if not min_trade_validation:
                failure_reasons.append(
                    f"No validation candidate met the minimum trade gate of {args.min_validation_trades} trades."
                )
            elif not delay_survivors:
                best_delay = max(
                    (float(row.get("delay_net_total_return", 0.0)) for row in min_trade_validation),
                    default=0.0,
                )
                failure_reasons.append(
                    f"{len(min_trade_validation)} validation candidates met the trade-count gate, "
                    "but none remained positive after one-horizon delayed execution "
                    f"(best delayed validation net return={best_delay:.6g})."
                )
            else:
                failure_reasons.append("No validation candidate survived the configured finalist ranking filters.")
        else:
            best_holdout = max(holdout_rows, key=lambda row: float(row.get("net_daily_sharpe", 0.0)), default={})
            failure_reasons.append(
                "No finalist passed the predefined promotion gate "
                f"(best holdout strategy={best_holdout.get('strategy_id', 'n/a')}, "
                f"net_return={float(best_holdout.get('net_total_return', 0.0)):.6g}, "
                f"monthly_net={float(best_holdout.get('average_monthly_net_return', 0.0)):.6g}, "
                f"Sharpe={float(best_holdout.get('net_daily_sharpe', 0.0)):.6g}, "
                f"PBO={pbo_report.pbo:.6g})."
            )
            for strategy_id, reasons in finalist_reasons.items():
                failure_reasons.append(f"{strategy_id}: {', '.join(reasons) if reasons else 'passed'}")

    registry = _registry_frame(list(predictions_by_strategy[key][0] for key in predictions_by_strategy), periods_payload)
    _write_json(
        output_dir / "run_config.json",
        {
            "args": vars(args),
            "command": command,
            "git_sha": _git_sha(),
            "run_id": run_id,
            "raw_rows": raw.height,
            "periods": periods_payload,
            "strategy_count": len(predictions_by_strategy),
        },
    )
    write_research_outputs(
        output_dir=output_dir,
        registry=registry,
        all_backtests=pl.DataFrame(all_rows, infer_schema_length=max(len(all_rows), 1)),
        pbo_payload=pbo_payload,
        best_candidates=validation_candidates,
        holdout_rows=holdout_rows,
        cost_sensitivity_rows=cost_rows,
        failure_reasons=failure_reasons,
        commands=[
            command,
            f"python - <<'PY'\nimport polars as pl\nprint(pl.read_parquet('{output_dir / 'all_backtests.parquet'}).head())\nPY",
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
