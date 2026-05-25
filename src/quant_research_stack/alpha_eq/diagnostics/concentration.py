"""Concentration check on net PnL by stock / month / sector (spec §6.4-11)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class ConcentrationReport:
    stock_violation: bool
    month_violation: bool
    sector_violation: bool
    stock_top: dict[str, float]
    month_top: dict[str, float]
    sector_top: dict[str, float]


def check_concentration(
    *,
    pnl: pl.DataFrame,
    max_stock_frac: float,
    max_month_frac: float,
    max_sector_frac: float,
) -> ConcentrationReport:
    total = float(pnl["net_pnl"].sum())
    if total == 0.0:
        return ConcentrationReport(False, False, False, {}, {}, {})
    by_stock = pnl.group_by("symbol").agg(pl.col("net_pnl").sum().alias("v"))
    by_month = pnl.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("ym")).group_by("ym").agg(
        pl.col("net_pnl").sum().alias("v")
    )
    by_sector = pnl.group_by("sector").agg(pl.col("net_pnl").sum().alias("v"))

    def _fractions(df: pl.DataFrame, key: str) -> dict[str, float]:
        return {row[key]: float(row["v"]) / total for row in df.to_dicts()}

    s_top = _fractions(by_stock, "symbol")
    m_top = _fractions(by_month, "ym")
    sec_top = _fractions(by_sector, "sector")
    s_v = max(s_top.values(), default=0.0) > max_stock_frac
    m_v = max(m_top.values(), default=0.0) > max_month_frac
    sec_v = max(sec_top.values(), default=0.0) > max_sector_frac
    return ConcentrationReport(s_v, m_v, sec_v, s_top, m_top, sec_top)
