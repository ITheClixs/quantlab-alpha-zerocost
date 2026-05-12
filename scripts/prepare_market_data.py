from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.artifacts import read_yaml, safe_repo_id, write_json


console = Console()


TIME_COLUMNS = ("timestamp", "time", "date", "datetime", "open_time", "close_time")
OPEN_COLUMNS = ("open", "o", "open_price")
HIGH_COLUMNS = ("high", "h", "high_price")
LOW_COLUMNS = ("low", "l", "low_price")
CLOSE_COLUMNS = ("close", "c", "close_price", "price", "last_price")
VOLUME_COLUMNS = ("volume", "vol", "base_volume", "quote_volume")
SYMBOL_COLUMNS = ("symbol", "ticker", "asset", "pair", "instrument")


def lower_map(columns: list[str]) -> dict[str, str]:
    return {col.lower().strip(): col for col in columns}


def first_present(mapping: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def scan_file(path: Path) -> pl.LazyFrame | None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix == ".csv":
        return pl.scan_csv(path, ignore_errors=True)
    return None


def canonical_market_frame(path: Path, dataset_id: str, horizons: list[int], vol_windows: list[int]) -> pl.LazyFrame | None:
    frame = scan_file(path)
    if frame is None:
        return None
    try:
        columns = frame.collect_schema().names()
    except Exception as exc:
        console.print(f"[yellow]Skipping unreadable tabular file[/yellow] {path}: {exc}")
        return None
    mapping = lower_map(columns)
    time_col = first_present(mapping, TIME_COLUMNS)
    open_col = first_present(mapping, OPEN_COLUMNS)
    high_col = first_present(mapping, HIGH_COLUMNS)
    low_col = first_present(mapping, LOW_COLUMNS)
    close_col = first_present(mapping, CLOSE_COLUMNS)
    volume_col = first_present(mapping, VOLUME_COLUMNS)
    symbol_col = first_present(mapping, SYMBOL_COLUMNS)

    if close_col is None:
        return None

    exprs = [
        pl.lit(dataset_id).alias("dataset_id"),
        pl.lit(str(path)).alias("source_file"),
        pl.col(close_col).cast(pl.Float64, strict=False).alias("close"),
    ]
    if time_col:
        exprs.append(pl.col(time_col).alias("timestamp"))
    else:
        exprs.append(pl.int_range(0, pl.len()).alias("timestamp"))
    if symbol_col:
        exprs.append(pl.col(symbol_col).cast(pl.Utf8, strict=False).alias("symbol"))
    else:
        exprs.append(pl.lit(dataset_id).alias("symbol"))
    if open_col:
        exprs.append(pl.col(open_col).cast(pl.Float64, strict=False).alias("open"))
    if high_col:
        exprs.append(pl.col(high_col).cast(pl.Float64, strict=False).alias("high"))
    if low_col:
        exprs.append(pl.col(low_col).cast(pl.Float64, strict=False).alias("low"))
    if volume_col:
        exprs.append(pl.col(volume_col).cast(pl.Float64, strict=False).alias("volume"))

    out = frame.select(exprs).drop_nulls(["close"])
    sort_cols = ["symbol", "timestamp"]
    out = out.sort(sort_cols)
    out = out.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias("return_1"),
            pl.col("close").log().diff().over("symbol").alias("log_return_1"),
        ]
    )
    for window in vol_windows:
        out = out.with_columns(
            pl.col("log_return_1")
            .rolling_std(window_size=window)
            .over("symbol")
            .alias(f"realized_vol_{window}")
        )
    for horizon in horizons:
        out = out.with_columns(
            [
                (pl.col("close").shift(-horizon).over("symbol") / pl.col("close") - 1.0).alias(f"future_return_{horizon}"),
                (pl.col("close").shift(-horizon).over("symbol") > pl.col("close")).cast(pl.Int8).alias(f"direction_up_{horizon}"),
            ]
        )
    if {"high", "low"}.issubset(set(out.collect_schema().names())):
        out = out.with_columns(((pl.col("high") - pl.col("low")) / pl.col("close")).alias("high_low_range"))
    return out


def process_dataset(
    dataset_dir: Path,
    output_root: Path,
    config: dict,
    max_files: int | None = None,
    max_source_file_gb: float | None = None,
    skip_existing: bool = False,
) -> dict:
    dataset_id = dataset_dir.name
    output_dir = output_root / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(x) for x in config["market_preparation"]["horizons"]]
    vol_windows = [int(x) for x in config["market_preparation"]["volatility_windows"]]
    files = sorted([*dataset_dir.rglob("*.parquet"), *dataset_dir.rglob("*.csv")])
    if max_source_file_gb is not None:
        max_bytes = int(max_source_file_gb * 1024**3)
        files = [path for path in files if path.stat().st_size <= max_bytes]
    if max_files is not None:
        files = files[:max_files]
    produced = []
    skipped = []
    for file_path in files:
        out_path = output_dir / f"{safe_repo_id(file_path.relative_to(dataset_dir).as_posix())}.features.parquet"
        if skip_existing and out_path.exists():
            produced.append(str(out_path))
            continue
        frame = canonical_market_frame(file_path, dataset_id, horizons, vol_windows)
        if frame is None:
            skipped.append(str(file_path))
            continue
        try:
            try:
                frame.sink_parquet(out_path)
            except Exception:
                frame.collect(engine="streaming").write_parquet(out_path)
        except Exception as exc:
            console.print(f"[yellow]Skipping unprocessable market file[/yellow] {file_path}: {exc}")
            skipped.append(str(file_path))
            continue
        produced.append(str(out_path))
    return {
        "dataset_dir": str(dataset_dir),
        "produced_files": produced,
        "skipped_files": skipped[:100],
        "skipped_count": len(skipped),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train-ready market feature parquet files.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--report", default="reports/market_preparation.json")
    parser.add_argument("--dataset", action="append", default=[], help="Process only these raw dataset directory names.")
    parser.add_argument("--max-files-per-dataset", type=int, default=None)
    parser.add_argument("--max-source-file-gb", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    input_root = Path(args.input_root or config["paths"]["raw_hf_root"])
    output_root = Path(args.output_root or config["paths"]["processed_market_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    if not input_root.exists():
        console.print(f"No input root exists: {input_root}")
        return 0
    requested = set(args.dataset)
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if requested and dataset_dir.name not in requested:
            continue
        result = process_dataset(
            dataset_dir,
            output_root,
            config,
            max_files=args.max_files_per_dataset,
            max_source_file_gb=args.max_source_file_gb,
            skip_existing=args.skip_existing,
        )
        results.append(result)
        console.print(f"{dataset_dir.name}: produced {len(result['produced_files'])} feature files")
    write_json(args.report, {"datasets": results})
    console.print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
