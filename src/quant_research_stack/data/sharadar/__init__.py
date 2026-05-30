"""Sharadar (or equivalent survivorship-safe equity) ingestion + audit scaffold.

INFRASTRUCTURE ONLY — no strategy code, no backtest. Purpose: remove implementation
latency so that if a survivorship-safe equity dataset (Sharadar SEP/TICKERS/ACTIONS
/SF1) is acquired, the repo can immediately ingest, manifest, audit, and build a
leak-safe return panel BEFORE any alpha work. Feasibility/kill-criterion gate:
docs/research/2026-05-DATA-PURCHASE-FEASIBILITY-SHARADAR.md.
"""

from quant_research_stack.data.sharadar.loaders import LoadedTable, load_table
from quant_research_stack.data.sharadar.schema import (
    SCHEMAS,
    schema_fingerprint,
    validate_schema,
)

__all__ = ["SCHEMAS", "LoadedTable", "load_table", "schema_fingerprint", "validate_schema"]
