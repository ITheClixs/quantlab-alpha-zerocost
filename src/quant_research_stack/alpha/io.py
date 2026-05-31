from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class LoadConfig:
    target_column: str
    weight_column: str
    group_column: str
    holdout_fraction: float = 0.20


def load_jane_street(path: str | Path, config: LoadConfig) -> pl.DataFrame:
    """Load JS Parquet (single file or directory of parquet shards).

    Returns a Polars DataFrame sorted by group_column ascending.
    Raises FileNotFoundError if path does not exist.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    if target.is_file():
        df = pl.read_parquet(target)
    else:
        train_dir = target / "train.parquet"
        if train_dir.exists():
            df = pl.read_parquet(train_dir)
        else:
            parquet_files = sorted(p for p in target.rglob("*.parquet") if p.is_file())
            if not parquet_files:
                raise FileNotFoundError(f"no parquet files found under {target}")
            df = pl.read_parquet(parquet_files)
    required = {config.target_column, config.weight_column, config.group_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df.sort(config.group_column)


def permanent_holdout_split(df: pl.DataFrame, config: LoadConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Chronological holdout by group_column. Last `holdout_fraction` of unique groups go to holdout."""
    if not 0.0 < config.holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1); got {config.holdout_fraction}")
    unique_groups = df[config.group_column].unique().sort()
    n = unique_groups.len()
    cut = int(round(n * (1 - config.holdout_fraction)))
    if cut == 0 or cut == n:
        raise ValueError(f"holdout split degenerate: cut={cut}, n_groups={n}")
    train_groups = unique_groups.head(cut).to_list()
    holdout_groups = unique_groups.tail(n - cut).to_list()
    train = df.filter(pl.col(config.group_column).is_in(train_groups))
    holdout = df.filter(pl.col(config.group_column).is_in(holdout_groups))
    return train, holdout


def scan_jane_street(path: str | Path, config: LoadConfig) -> pl.LazyFrame:
    """Lazy scan of JS Parquet. Memory-friendly counterpart to load_jane_street.

    Returns a Polars LazyFrame whose schema has been validated for required columns.
    Use this for large datasets where materializing the whole frame would OOM.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    if target.is_file():
        lf = pl.scan_parquet(target)
    else:
        train_dir = target / "train.parquet"
        if train_dir.exists():
            lf = pl.scan_parquet(train_dir)
        else:
            parquet_files = sorted(p for p in target.rglob("*.parquet") if p.is_file())
            if not parquet_files:
                raise FileNotFoundError(f"no parquet files found under {target}")
            lf = pl.scan_parquet(parquet_files)
    schema = lf.collect_schema()
    required = {config.target_column, config.weight_column, config.group_column}
    missing = required - set(schema.names())
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return lf


def select_tail_by_row_budget(
    lf: pl.LazyFrame, group_column: str, max_rows: int
) -> pl.DataFrame:
    """Materialize the most-recent contiguous groups whose cumulative row count fits `max_rows`.

    Preserves chronology and never splits a group across the boundary. Always includes at least
    one group (the most recent), even if that single group exceeds `max_rows`.
    Returns a Polars DataFrame sorted by group_column ascending.
    """
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive; got {max_rows}")
    counts = (
        lf.group_by(group_column)
        .agg(pl.len().alias("__n"))
        .sort(group_column, descending=True)
        .collect()
    )
    if counts.height == 0:
        raise ValueError("no rows available to sample from")
    cum = counts.with_columns(pl.col("__n").cum_sum().alias("__cum"))
    under = cum.filter(pl.col("__cum") <= max_rows)
    if under.height == 0:
        selected = cum.head(1)
    elif under.height < cum.height:
        selected = cum.head(under.height + 1)
    else:
        selected = under
    selected_groups = selected[group_column].sort().to_list()
    return lf.filter(pl.col(group_column).is_in(selected_groups)).collect().sort(group_column)
