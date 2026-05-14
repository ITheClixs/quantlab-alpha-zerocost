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
        df = pl.read_parquet(list(target.rglob("*.parquet")))
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
    train_groups = unique_groups.head(cut)
    holdout_groups = unique_groups.tail(n - cut)
    train = df.filter(pl.col(config.group_column).is_in(train_groups))
    holdout = df.filter(pl.col(config.group_column).is_in(holdout_groups))
    return train, holdout
