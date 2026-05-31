"""Borrow drag for the strict backtest (spec §5.7)."""

from __future__ import annotations

import polars as pl

from quant_research_stack.alpha_eq.data.borrow_proxy import apply_borrow_charges


def apply_borrow_drag(positions: pl.DataFrame, *, multiplier: float) -> pl.DataFrame:
    out = apply_borrow_charges(positions, multiplier=multiplier)
    return out.rename({"borrow_cost": "borrow_drag"})
