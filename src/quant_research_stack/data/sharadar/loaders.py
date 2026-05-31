"""Local-file Sharadar table loaders (parquet/csv) + per-table metadata.

No API/purchase logic — ingestion is from local files only. Each load validates
the schema and computes metadata for the manifest (rows, date range, symbol /
permaticker counts, file sha256, schema fingerprint).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from quant_research_stack.data.sharadar.schema import (
    SCHEMAS,
    present_optional,
    schema_fingerprint,
    validate_schema,
)

# expected file stems per table (case-insensitive); first match in a data dir wins
_FILE_STEMS = {"SEP": ("sep", "sharadar_sep"), "TICKERS": ("tickers", "sharadar_tickers"),
               "ACTIONS": ("actions", "sharadar_actions"), "SF1": ("sf1", "sharadar_sf1")}


@dataclass
class LoadedTable:
    name: str
    path: str
    df: pl.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


def _read(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix in (".csv", ".gz", ".zip"):
        return pl.read_csv(path)
    raise ValueError(f"unsupported file type: {path.suffix}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_table_path(data_dir: Path | str, table: str) -> Path | None:
    d = Path(data_dir)
    if not d.exists():
        return None
    for stem in _FILE_STEMS[table]:
        for ext in (".parquet", ".csv", ".csv.gz", ".zip"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
        hits = sorted(d.glob(f"{stem}*"))
        if hits:
            return hits[0]
    return None


def table_metadata(df: pl.DataFrame, table: str, path: Path) -> dict[str, Any]:
    sch = SCHEMAS[table]
    meta: dict[str, Any] = {
        "table": table, "path": str(path), "sha256": _sha256(path),
        "rows": df.height, "schema_fingerprint": schema_fingerprint(df),
        "columns": df.columns, "optional_present": present_optional(df, table),
        "symbol_count": int(df[sch.symbol_col].n_unique()) if sch.symbol_col in df.columns else None,
        "permaticker_count": int(df[sch.permaticker_col].n_unique())
        if sch.permaticker_col and sch.permaticker_col in df.columns else None,
        "date_min": None, "date_max": None, "warnings": [],
    }
    if sch.date_col and sch.date_col in df.columns:
        meta["date_min"] = str(df[sch.date_col].min())
        meta["date_max"] = str(df[sch.date_col].max())
    if table == "TICKERS" and "cik" not in df.columns:
        meta["warnings"].append("no direct `cik` column — EDGAR CIK mapping needs a CUSIP/ticker bridge [VERIFY]")
    if table == "SEP" and "closeadj" not in df.columns:
        meta["warnings"].append("no `closeadj` — total returns must be built from close + dividends")
    return meta


def load_table(data_dir: Path | str, table: str) -> LoadedTable | None:
    path = find_table_path(data_dir, table)
    if path is None:
        return None
    df = _read(path)
    validate_schema(df, table)  # raises SchemaError on missing required columns
    return LoadedTable(name=table, path=str(path), df=df, metadata=table_metadata(df, table, path))


def load_all(data_dir: Path | str, *, tables: tuple[str, ...] = ("SEP", "TICKERS", "ACTIONS", "SF1")) -> dict[str, LoadedTable]:
    out: dict[str, LoadedTable] = {}
    for t in tables:
        loaded = load_table(data_dir, t)
        if loaded is not None:
            out[t] = loaded
    return out
