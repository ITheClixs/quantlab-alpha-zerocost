"""Expected Sharadar table schemas + validation.

Required columns are the minimum needed for the audit + return panel; optional
columns are recorded if present. Schemas are tolerant of extra vendor columns.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class TableSchema:
    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    date_col: str | None
    symbol_col: str
    permaticker_col: str | None


SCHEMAS: dict[str, TableSchema] = {
    "SEP": TableSchema(
        name="SEP",
        required=("ticker", "date", "open", "high", "low", "close", "volume"),
        optional=("closeadj", "closeunadj", "dividends", "lastupdated"),
        date_col="date", symbol_col="ticker", permaticker_col=None,
    ),
    "TICKERS": TableSchema(
        name="TICKERS",
        required=("permaticker", "ticker", "name", "isdelisted"),
        optional=("exchange", "category", "cusips", "siccode", "sector", "industry",
                  "firstpricedate", "lastpricedate", "firstadded", "cik", "secfilings"),
        date_col=None, symbol_col="ticker", permaticker_col="permaticker",
    ),
    "ACTIONS": TableSchema(
        name="ACTIONS",
        required=("date", "action", "ticker"),
        optional=("name", "value", "contraticker", "contraname"),
        date_col="date", symbol_col="ticker", permaticker_col=None,
    ),
    "SF1": TableSchema(
        name="SF1",
        required=("ticker", "dimension", "datekey", "reportperiod"),
        optional=("calendardate", "cik", "revenue", "netinc", "assets", "equity"),
        date_col="datekey", symbol_col="ticker", permaticker_col=None,
    ),
}


class SchemaError(ValueError):
    pass


def validate_schema(df: pl.DataFrame, table: str) -> None:
    if table not in SCHEMAS:
        raise SchemaError(f"unknown Sharadar table {table!r}; known: {sorted(SCHEMAS)}")
    missing = [c for c in SCHEMAS[table].required if c not in df.columns]
    if missing:
        raise SchemaError(f"{table}: missing required columns {missing}")


def schema_fingerprint(df: pl.DataFrame) -> str:
    payload = ",".join(f"{c}:{df.schema[c]}" for c in sorted(df.columns))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def present_optional(df: pl.DataFrame, table: str) -> list[str]:
    return [c for c in SCHEMAS[table].optional if c in df.columns]
