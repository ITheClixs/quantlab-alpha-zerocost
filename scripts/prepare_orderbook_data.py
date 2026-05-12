from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console

from quant_research_stack.artifacts import read_yaml, safe_repo_id, write_json


console = Console()


def parse_levels(raw: Any) -> list[tuple[float, float]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    levels = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            price = float(item[0])
            qty = float(item[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and math.isfinite(qty):
            levels.append((price, qty))
    return levels


def symbol_from_path(path: Path) -> str:
    parent = path.parent.name
    if parent and parent.upper().endswith("USDT"):
        return parent
    for part in path.stem.split("_"):
        if part.upper().endswith("USDT"):
            return part.upper()
    return parent or path.stem


def orderbook_features(path: Path, dataset_id: str, horizons: list[int], depth_levels: list[int]) -> pl.DataFrame | None:
    try:
        frame = pl.read_parquet(path)
    except Exception as exc:
        console.print(f"[yellow]Skipping unreadable order-book file[/yellow] {path}: {exc}")
        return None
    if not {"bids", "asks"}.issubset(set(frame.columns)):
        return None

    bids_raw = frame["bids"].to_list()
    asks_raw = frame["asks"].to_list()
    bid_levels = [parse_levels(raw) for raw in bids_raw]
    ask_levels = [parse_levels(raw) for raw in asks_raw]

    rows: list[dict[str, Any]] = []
    symbol = symbol_from_path(path)
    for index, (bids, asks) in enumerate(zip(bid_levels, ask_levels, strict=False)):
        if not bids or not asks:
            continue
        best_bid, best_bid_qty = bids[0]
        best_ask, best_ask_qty = asks[0]
        mid = (best_bid + best_ask) / 2.0
        denom = best_bid_qty + best_ask_qty
        microprice = ((best_ask * best_bid_qty) + (best_bid * best_ask_qty)) / denom if denom else None
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "source_file": str(path),
            "symbol": symbol,
            "row_index": index,
            "event_time": frame["E"][index] if "E" in frame.columns else None,
            "transaction_time": frame["T"][index] if "T" in frame.columns else None,
            "update_id": frame["u"][index] if "u" in frame.columns else frame["lastUpdateId"][index] if "lastUpdateId" in frame.columns else None,
            "best_bid": best_bid,
            "best_bid_qty": best_bid_qty,
            "best_ask": best_ask,
            "best_ask_qty": best_ask_qty,
            "mid_price": mid,
            "spread": best_ask - best_bid,
            "relative_spread": (best_ask - best_bid) / mid if mid else None,
            "microprice_l1": microprice,
            "imbalance_l1": (best_bid_qty - best_ask_qty) / denom if denom else None,
        }
        for depth in depth_levels:
            bid_depth = sum(qty for _, qty in bids[:depth])
            ask_depth = sum(qty for _, qty in asks[:depth])
            depth_denom = bid_depth + ask_depth
            row[f"bid_depth_{depth}"] = bid_depth
            row[f"ask_depth_{depth}"] = ask_depth
            row[f"imbalance_depth_{depth}"] = (bid_depth - ask_depth) / depth_denom if depth_denom else None
        rows.append(row)

    if not rows:
        return None
    out = pl.DataFrame(rows).sort(["symbol", "row_index"])
    for horizon in horizons:
        out = out.with_columns(
            [
                (pl.col("mid_price").shift(-horizon).over("symbol") / pl.col("mid_price") - 1.0).alias(f"future_mid_return_{horizon}"),
                (pl.col("mid_price").shift(-horizon).over("symbol") > pl.col("mid_price")).cast(pl.Int8).alias(f"mid_direction_up_{horizon}"),
            ]
        )
    return out


def process_dataset(
    dataset_dir: Path,
    output_root: Path,
    config: dict,
    max_files: int | None = None,
    max_source_file_gb: float | None = None,
) -> dict:
    dataset_id = dataset_dir.name
    output_dir = output_root / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(x) for x in config["market_preparation"]["horizons"]]
    depth_levels = [1, 5, 10, 20]
    files = sorted(dataset_dir.rglob("*.parquet"))
    if max_source_file_gb is not None:
        max_bytes = int(max_source_file_gb * 1024**3)
        files = [path for path in files if path.stat().st_size <= max_bytes]
    if max_files is not None:
        files = files[:max_files]

    produced = []
    skipped = []
    for file_path in files:
        frame = orderbook_features(file_path, dataset_id, horizons, depth_levels)
        if frame is None:
            skipped.append(str(file_path))
            continue
        out_path = output_dir / f"{safe_repo_id(file_path.relative_to(dataset_dir).as_posix())}.orderbook_features.parquet"
        frame.write_parquet(out_path)
        produced.append(str(out_path))
    return {
        "dataset_dir": str(dataset_dir),
        "produced_files": produced,
        "skipped_files": skipped[:100],
        "skipped_count": len(skipped),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train-ready order-book feature parquet files.")
    parser.add_argument("--config", default="configs/stack.yaml")
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--report", default="reports/orderbook_preparation.json")
    parser.add_argument("--dataset", action="append", default=[], help="Process only these raw dataset directory names.")
    parser.add_argument("--max-files-per-dataset", type=int, default=None)
    parser.add_argument("--max-source-file-gb", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    input_root = Path(args.input_root or config["paths"]["raw_hf_root"])
    output_root = Path(args.output_root or config["paths"]["processed_orderbook_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    requested = set(args.dataset)
    results = []
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if requested and dataset_dir.name not in requested:
            continue
        if "orderbook" not in dataset_dir.name.lower() and "order_book" not in dataset_dir.name.lower():
            continue
        result = process_dataset(
            dataset_dir,
            output_root,
            config,
            max_files=args.max_files_per_dataset,
            max_source_file_gb=args.max_source_file_gb,
        )
        results.append(result)
        console.print(f"{dataset_dir.name}: produced {len(result['produced_files'])} order-book feature files")
    write_json(args.report, {"datasets": results})
    console.print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
