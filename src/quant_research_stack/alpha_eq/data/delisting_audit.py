"""Delisting-return audit (spec §2.9).

Classifies every symbol-exit into a known category so that missing
terminal losses do not silently inflate equity backtest performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

_EXIT_REASONS_DELISTED: frozenset[str] = frozenset(
    {"bankruptcy", "regulatory_delisting", "going_to_zero", "delisted"}
)
_EXIT_REASONS_MERGER: frozenset[str] = frozenset(
    {"acquired", "merger", "going_private", "buyout"}
)
_EXIT_REASONS_TICKER: frozenset[str] = frozenset({"ticker_change"})


def classify_exit(*, reason: str, terminal_return_known: bool) -> str:
    r = reason.lower()
    if r in _EXIT_REASONS_TICKER:
        return "ticker_changed"
    if r in _EXIT_REASONS_DELISTED:
        return "delisted_captured" if terminal_return_known else "delisted_missing"
    if r in _EXIT_REASONS_MERGER:
        return "merger_captured" if terminal_return_known else "merger_missing"
    return "unknown_exit"


@dataclass(frozen=True)
class DelistingAuditResult:
    counters: dict[str, int]
    audit_table: pl.DataFrame


def audit_delistings(*, panel: pl.DataFrame, exits: pl.DataFrame) -> DelistingAuditResult:
    """Inspect a panel + an exits feed.  Any symbol whose last observation
    in `panel` is before the panel's global max date AND has no matching
    `exits` row is recorded as `unknown_exit`."""
    panel = panel.sort(["symbol", "date"])
    global_max = panel["date"].max()
    last_seen = panel.group_by("symbol").agg(pl.col("date").max().alias("last_date"))
    exited = last_seen.filter(pl.col("last_date") < global_max)

    rows: list[dict[str, object]] = []
    known_symbols: set[str] = set(exits["symbol"].to_list()) if not exits.is_empty() else set()
    for sym, last in zip(exited["symbol"].to_list(), exited["last_date"].to_list(), strict=True):
        if sym in known_symbols:
            erow = exits.filter(pl.col("symbol") == sym)
            reason = str(erow["exit_reason"][0])
            terminal_known = bool(erow["terminal_return_known"][0])
            classification = classify_exit(reason=reason, terminal_return_known=terminal_known)
            terminal_value = (
                float(erow["terminal_return_value"][0])
                if "terminal_return_value" in erow.columns
                and erow["terminal_return_value"][0] is not None
                else None
            )
        else:
            classification = "unknown_exit"
            reason = "unknown"
            terminal_known = False
            terminal_value = None
        rows.append(
            {
                "symbol": sym,
                "exit_date": last,
                "exit_reason": reason,
                "terminal_return_captured": terminal_known,
                "terminal_return_value": terminal_value,
                "classification_source": "exits_feed" if sym in known_symbols else "panel_inferred",
                "classification": classification,
            }
        )
    audit_df = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "symbol": pl.Utf8,
            "exit_date": pl.Date,
            "exit_reason": pl.Utf8,
            "terminal_return_captured": pl.Boolean,
            "terminal_return_value": pl.Float64,
            "classification_source": pl.Utf8,
            "classification": pl.Utf8,
        }
    )
    counters = {
        k: int((audit_df["classification"] == k).sum()) if not audit_df.is_empty() else 0
        for k in (
            "delisted_captured",
            "delisted_missing",
            "merger_captured",
            "merger_missing",
            "ticker_changed",
            "unknown_exit",
        )
    }
    return DelistingAuditResult(counters=counters, audit_table=audit_df)
