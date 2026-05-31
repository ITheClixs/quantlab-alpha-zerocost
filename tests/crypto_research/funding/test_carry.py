"""Unit tests for delta-neutral funding-carry math (Strategy A)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from quant_research_stack.crypto_research.funding import carry


def _panel(spot: list[float], perp: list[float], fund: list[float]) -> pl.DataFrame:
    days = [date(2024, 1, 1 + i) for i in range(len(spot))]
    return pl.DataFrame({"date": days, "spot_close": spot, "perp_close": perp,
                         "funding_day": fund})


def test_day0_collects_no_funding_only_entry_cost() -> None:
    r = carry.carry_returns(_panel([100, 100], [100, 100], [0.001, 0.001]),
                            spot_taker_bps=10, perp_taker_bps=5)
    rt = (10 + 5) * 1e-4
    assert r.funding[0] == 0.0
    assert r.net[0] == pytest.approx(-rt)          # entry cost only
    # day 1: flat prices, +0.001 funding, minus exit cost (last day)
    assert r.funding[1] == pytest.approx(0.001)
    assert r.net[1] == pytest.approx(0.001 - rt)   # exit on final day


def test_short_perp_receives_positive_funding() -> None:
    r = carry.carry_returns(_panel([100, 100, 100], [100, 100, 100],
                                   [0.0, 0.001, 0.001]), spot_taker_bps=0, perp_taker_bps=0)
    # no costs, flat prices -> net == funding on interior day
    assert r.net[1] == pytest.approx(0.001)
    assert r.metrics["total_return"] > 0


def test_invert_flips_sign() -> None:
    base = carry.carry_returns(_panel([100, 101, 102], [100, 101, 102],
                                      [0.0, 0.001, 0.001]), spot_taker_bps=0, perp_taker_bps=0)
    inv = carry.carry_returns(_panel([100, 101, 102], [100, 101, 102],
                                     [0.0, 0.001, 0.001]), spot_taker_bps=0,
                              perp_taker_bps=0, invert=True)
    assert inv.gross == pytest.approx(-base.gross)


def test_zero_funding_isolates_price() -> None:
    r = carry.carry_returns(_panel([100, 110, 121], [100, 110, 121],
                                   [0.0, 0.01, 0.01]), spot_taker_bps=0,
                            perp_taker_bps=0, zero_funding=True)
    # spot and perp move identically -> price term zero, funding dropped -> all zero
    assert np.allclose(r.gross, 0.0)


def test_price_term_captures_basis_convergence() -> None:
    # perp starts 1% rich then converges: short perp gains on price
    r = carry.carry_returns(_panel([100, 100], [101, 100], [0.0, 0.0]),
                            spot_taker_bps=0, perp_taker_bps=0)
    # day1: spot_ret=0, perp_ret=-0.0099 -> price = +0.0099 (short gains)
    assert r.price[1] > 0


def test_per_year_splits() -> None:
    dates = [date(2023, 6, 1), date(2023, 6, 2), date(2024, 6, 1), date(2024, 6, 2)]
    net = np.array([0.01, 0.01, -0.01, -0.01])
    py = carry.per_year(dates, net)
    assert set(py) == {2023, 2024}
    assert py[2023]["total_pct"] > 0
    assert py[2024]["total_pct"] < 0


def test_pooled_book_averages() -> None:
    a = carry.carry_returns(_panel([100, 100], [100, 100], [0.0, 0.002]),
                            spot_taker_bps=0, perp_taker_bps=0)
    b = carry.carry_returns(_panel([100, 100], [100, 100], [0.0, 0.004]),
                            spot_taker_bps=0, perp_taker_bps=0)
    pool = carry.pooled_book({"A": a, "B": b})
    assert pool.net[1] == pytest.approx((a.net[1] + b.net[1]) / 2)


def test_pooled_book_respects_periods_per_year() -> None:
    """Regression: pooled metrics must annualize at the bar frequency. A pooled 8h book
    annualized at 365 instead of 1095 understates Sharpe by sqrt(3) and return by ~3x."""
    rng = np.random.default_rng(0)
    net = rng.normal(0.0001, 0.0005, 3000)
    r = carry.CarryResult(dates=[date(2024, 1, 1)] * 3000, net=net, gross=net,
                          funding=net, price=np.zeros(3000), cost=np.zeros(3000),
                          metrics={})
    daily = carry.pooled_book({"x": r})
    eighth = carry.pooled_book({"x": r}, periods_per_year=carry.PPY_8H)
    assert eighth.metrics["sharpe"] == pytest.approx(daily.metrics["sharpe"] * np.sqrt(3), rel=1e-6)


def test_pnl_concentration_flags_single_year() -> None:
    dates = [date(2021, 1, 1), date(2022, 1, 1)]
    net = np.array([0.10, 0.0])  # all PnL in 2021
    c = carry.pnl_concentration(dates, net)
    assert c["top_year_share"] == pytest.approx(1.0)


def _panel_8h(n: int, basis_jitter: float = 0.0, fund: float = 0.0002) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    base = 100.0 * (1 + np.cumsum(rng.normal(0, 0.005, n)))
    perp = base * (1.0 + rng.normal(0, basis_jitter, n))
    ts = pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1) + __import__("datetime").timedelta(hours=8 * (n - 1)),
                           interval="8h", eager=True, time_zone="UTC")
    return pl.DataFrame({
        "ts": ts, "date": ts.dt.date(),
        "spot_close": base, "perp_close": perp,
        "spot_high": base * 1.001, "spot_low": base * 0.999,
        "perp_high": perp * 1.001, "perp_low": perp * 0.999,
        "funding_rate": np.full(n, fund), "basis": perp / base - 1.0,
    })


def test_carry_8h_higher_basis_jitter_lowers_sharpe() -> None:
    smooth = carry.carry_returns_8h(_panel_8h(300, basis_jitter=0.0), spot_taker_bps=0,
                                    perp_taker_bps=0, slip_bps=0)
    jittery = carry.carry_returns_8h(_panel_8h(300, basis_jitter=0.01), spot_taker_bps=0,
                                     perp_taker_bps=0, slip_bps=0)
    # real intrabar basis variance must deflate the (illusory) smooth Sharpe
    assert jittery.metrics["sharpe"] < smooth.metrics["sharpe"]


def test_carry_8h_annualizes_at_1095() -> None:
    assert carry.PPY_8H == pytest.approx(1095.0)


def test_liquidation_diag_counts_events_by_leverage() -> None:
    p = _panel_8h(100, basis_jitter=0.0)
    # force a large adverse intrabar excursion on one bar
    p = p.with_columns(
        pl.when(pl.int_range(pl.len()) == 50).then(pl.col("perp_high") * 1.30)
        .otherwise(pl.col("perp_high")).alias("perp_high"))
    diag = carry.liquidation_diag(p, leverages=(3.0, 10.0))
    assert diag["liq_events"]["10x"] >= 1           # 30% excursion liquidates a 10x short
    assert diag["max_adverse_dbasis_pct"] > 20.0


def test_liquidation_stress_reduces_return() -> None:
    p = _panel_8h(100, basis_jitter=0.0)
    p = p.with_columns(
        pl.when(pl.int_range(pl.len()) == 50).then(pl.col("perp_high") * 1.30)
        .otherwise(pl.col("perp_high")).alias("perp_high"))
    base = carry.carry_returns_8h(p, spot_taker_bps=2, perp_taker_bps=1, slip_bps=1)
    stressed = carry.carry_liquidation_stressed(p, leverage=10.0, spot_taker_bps=2,
                                                perp_taker_bps=1, slip_bps=1)
    assert stressed["n_liquidations"] >= 1
    assert stressed["total_return"] < base.metrics["total_return"]
