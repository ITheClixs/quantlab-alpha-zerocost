"""PIT S&P 500 membership table + ticker-mapping logic.

Spec §2.2. The audit script `pit_quality_audit.py` decides which source
to write; this module only loads + applies the table.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl


class MembershipSource(enum.StrEnum):
    HF_PRIMARY = "hf_primary"
    HF_SECONDARY = "hf_secondary"
    KAGGLE = "kaggle"
    WIKIPEDIA_FALLBACK = "wikipedia_fallback"
    ABSENT_PROTOTYPE_ONLY = "absent_prototype_only"


_REQUIRED_COLS: tuple[str, ...] = (
    "date",
    "symbol",
    "in_index",
    "addition_date",
    "removal_date",
    "removal_reason",
)


@dataclass(frozen=True)
class PITMembership:
    source: MembershipSource
    table: pl.DataFrame

    def is_in_index(self, *, symbol: str, on: date) -> bool:
        f = self.table.filter(
            (pl.col("symbol") == symbol) & (pl.col("date") == on)
        )
        if f.is_empty():
            return False
        return bool(f["in_index"][0])


@dataclass(frozen=True)
class TickerMapping:
    """A list of (old_symbol, new_symbol, effective_date) transforms.

    The mapping is applied row-wise: any row whose date >= effective_date and
    whose current symbol is the old_symbol gets renamed to the new_symbol.
    """

    rows: list[tuple[str, str, date]] = field(default_factory=list)


def load_pit_membership(path: Path, *, source: MembershipSource) -> PITMembership:
    if not Path(path).exists():
        raise FileNotFoundError(f"PIT membership table missing: {path}")
    df = pl.read_parquet(path)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"PIT membership table missing column(s): {missing}")
    return PITMembership(source=source, table=df)


def apply_ticker_mapping(df: pl.DataFrame, mapping: TickerMapping) -> pl.DataFrame:
    out = df
    for old, new, effective in mapping.rows:
        out = out.with_columns(
            pl.when((pl.col("symbol") == old) & (pl.col("date") >= effective))
            .then(pl.lit(new))
            .otherwise(pl.col("symbol"))
            .alias("symbol")
        )
    return out
