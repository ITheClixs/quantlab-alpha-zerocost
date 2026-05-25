"""Backtest runner — orchestrates per-date portfolio construction, fills,
PnL accounting, exposures, and metrics (spec §5)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_research_stack.alpha_eq.backtest.borrow import apply_borrow_drag
from quant_research_stack.alpha_eq.backtest.costs import (
    CostConfig,
    compute_commission_drag,
    compute_spread_drag,
)
from quant_research_stack.alpha_eq.backtest.exposure import compute_daily_exposures
from quant_research_stack.alpha_eq.backtest.fills import FillModel, pick_fill_prices
from quant_research_stack.alpha_eq.backtest.financing import compute_financing_drag
from quant_research_stack.alpha_eq.backtest.pnl import (
    PnLDecomposition,
    decompose_pnl,
)
from quant_research_stack.alpha_eq.backtest.portfolio import (
    PortfolioBuildConfig,
    build_target_positions,
)


@dataclass(frozen=True)
class BacktestConfig:
    portfolio: PortfolioBuildConfig
    fill_model: FillModel
    cohort: str
    borrow_multiplier: float
    financing_rate_annual: float
    cost: CostConfig = CostConfig()


@dataclass(frozen=True)
class BacktestResult:
    daily_returns: pl.DataFrame
    positions: pl.DataFrame
    exposures: pl.DataFrame
    decomposition: PnLDecomposition


def _step_one_day(
    *,
    today: pl.DataFrame,
    prior_positions: pl.DataFrame,
    config: BacktestConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    bars = today.select(
        ["execution_date", "symbol", "open", "high", "low", "close"]
    ).rename({"execution_date": "date"})
    fills = pick_fill_prices(bars, model=config.fill_model)
    today = today.join(
        fills.select(["date", "symbol", "fill_price"]).rename({"date": "execution_date"}),
        on=["execution_date", "symbol"],
        how="left",
    )

    book = build_target_positions(signals=today, config=config.portfolio, cohort=config.cohort)
    if book.is_empty():
        return prior_positions, pl.DataFrame(
            schema={
                "date": pl.Date, "net_return": pl.Float64, "gross_return": pl.Float64,
                "commission_drag": pl.Float64, "spread_drag": pl.Float64,
                "borrow_drag": pl.Float64, "financing_drag": pl.Float64,
            }
        )

    today_date = book["execution_date"].min()

    # Build close_today map for PnL
    close_today_map = today.select(["symbol", "close"])

    # Determine held vs new positions
    if not prior_positions.is_empty():
        merged = book.rename({"signed_target_notional": "signed_new"}).join(
            prior_positions.rename({"signed_notional": "signed_prev", "close": "close_prev"}),
            on="symbol", how="full", coalesce=True,
        ).with_columns(
            pl.col("signed_new").fill_null(0.0),
            pl.col("signed_prev").fill_null(0.0),
        ).join(close_today_map, on="symbol", how="left")
    else:
        merged = book.rename({"signed_target_notional": "signed_new"}).with_columns(
            pl.lit(0.0).alias("signed_prev"),
            pl.lit(None, dtype=pl.Float64).alias("close_prev"),
        ).join(close_today_map, on="symbol", how="left")

    # Price PnL on held (prev != 0 and we have close_prev)
    held_pnl = float(
        merged.filter(
            pl.col("signed_prev").abs() > 0.0,
        ).filter(
            pl.col("close_prev").is_not_null() & pl.col("close").is_not_null()
        ).with_columns(
            (pl.col("signed_prev") * (pl.col("close") / pl.col("close_prev") - 1.0)).alias("_p")
        )["_p"].sum()
    )

    # New-lot PnL: new shares = (signed_new - signed_prev) / fill_price; PnL = shares * (close - fill)
    new_lot_pnl = float(
        merged.filter(
            (pl.col("signed_new") - pl.col("signed_prev")).abs() > 0.0
        ).filter(
            pl.col("fill_price").is_not_null() & pl.col("close").is_not_null()
        ).with_columns(
            (
                ((pl.col("signed_new") - pl.col("signed_prev")) / pl.col("fill_price"))
                * (pl.col("close") - pl.col("fill_price"))
            ).alias("_p")
        )["_p"].sum()
    )

    price_pnl_total = held_pnl + new_lot_pnl

    # Costs: commission + spread on rebalance volume (|signed_new - signed_prev|)
    trades = merged.with_columns(
        (pl.col("signed_new") - pl.col("signed_prev")).abs().alias("trade_notional_abs"),
        pl.col("execution_date").alias("date"),
    ).filter(pl.col("trade_notional_abs") > 0.0)
    if "tier" not in trades.columns and "borrow_tier" in trades.columns:
        trades = trades.with_columns(pl.col("borrow_tier").alias("tier"))
    elif "tier" not in trades.columns:
        trades = trades.with_columns(pl.lit("general").alias("tier"))
    if "roll_spread_bps" not in trades.columns:
        trades = trades.with_columns(pl.lit(None, dtype=pl.Float64).alias("roll_spread_bps"))
    comm = compute_commission_drag(
        trades.select(["date", "symbol", "trade_notional_abs"]), cost=config.cost
    )
    spr = compute_spread_drag(
        trades.select(["date", "symbol", "trade_notional_abs", "roll_spread_bps", "tier"]),
        cost=config.cost,
    )
    comm_total = float(comm["commission_drag"].sum())
    spr_total = float(spr["spread_drag"].sum())

    # Borrow on target book (signed_new)
    borrow_pos = book.rename(
        {"signed_target_notional": "signed_notional", "execution_date": "date"}
    ).select(["date", "symbol", "signed_notional", "borrow_tier"]).rename(
        {"borrow_tier": "tier"}
    )
    borrow = apply_borrow_drag(borrow_pos, multiplier=config.borrow_multiplier)
    borrow_total = float(borrow["borrow_drag"].sum())

    # Financing on gross > 1
    gross = float(book["signed_target_notional"].abs().sum())
    fin_in = pl.DataFrame(
        {
            "date": [today_date],
            "gross_notional": [gross],
            "equity": [config.portfolio.equity],
        }
    )
    fin = compute_financing_drag(fin_in, rate_annual=config.financing_rate_annual)
    fin_total = float(fin["financing_drag"][0])

    net = price_pnl_total - comm_total - spr_total - borrow_total - fin_total
    equity = config.portfolio.equity
    daily = pl.DataFrame(
        {
            "date": [today_date],
            "gross_return": [price_pnl_total / equity],
            "net_return": [net / equity],
            "commission_drag": [comm_total],
            "spread_drag": [spr_total],
            "borrow_drag": [borrow_total],
            "financing_drag": [fin_total],
        }
    )
    next_positions = (
        book.select(["execution_date", "symbol", "signed_target_notional"])
        .rename({"execution_date": "date", "signed_target_notional": "signed_notional"})
        .join(close_today_map, on="symbol", how="left")
    )
    return next_positions, daily


def run_backtest(
    *,
    signals_with_bars: pl.DataFrame,
    config: BacktestConfig,
    dividends: pl.DataFrame | None,
) -> BacktestResult:
    all_dates = sorted(signals_with_bars["execution_date"].unique().to_list())
    positions = pl.DataFrame(
        schema={
            "date": pl.Date, "symbol": pl.Utf8,
            "signed_notional": pl.Float64, "close": pl.Float64,
        }
    )
    daily_frames: list[pl.DataFrame] = []
    for d in all_dates:
        today = signals_with_bars.filter(pl.col("execution_date") == d)
        positions, daily = _step_one_day(
            today=today, prior_positions=positions, config=config
        )
        if not daily.is_empty():
            daily_frames.append(daily)
    daily_all = pl.concat(daily_frames) if daily_frames else pl.DataFrame(
        schema={"date": pl.Date, "net_return": pl.Float64, "gross_return": pl.Float64,
                "commission_drag": pl.Float64, "spread_drag": pl.Float64,
                "borrow_drag": pl.Float64, "financing_drag": pl.Float64}
    )
    expo = (
        compute_daily_exposures(positions=positions)
        if not positions.is_empty()
        else pl.DataFrame()
    )

    dec = decompose_pnl(
        price_pnl=pl.DataFrame({"date": [], "symbol": [], "price_pnl": []}),
        cash_dividend_pnl=pl.DataFrame({"date": [], "symbol": [], "cash_dividend_pnl": []}),
        commission_drag=daily_all.select(["date", "commission_drag"]).with_columns(
            pl.lit("agg").alias("symbol")
        ),
        spread_drag=daily_all.select(["date", "spread_drag"]).with_columns(
            pl.lit("agg").alias("symbol")
        ),
        borrow_drag=daily_all.select(["date", "borrow_drag"]).with_columns(
            pl.lit("agg").alias("symbol")
        ),
        financing_drag=daily_all.select(["date", "financing_drag"]).with_columns(
            pl.lit("agg").alias("symbol")
        ),
        equity=config.portfolio.equity,
        n_days=daily_all.height,
    )
    return BacktestResult(
        daily_returns=daily_all,
        positions=positions,
        exposures=expo,
        decomposition=dec,
    )
