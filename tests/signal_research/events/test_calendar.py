from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from quant_research_stack.signal_research.events.calendar import (
    _in_earnings_season,
    attach_event_features,
    load_fomc_dates,
)


def _bars(start: date, n: int) -> pl.DataFrame:
    # business-day-ish index: skip weekends so windows resemble trading days
    days: list[date] = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return pl.DataFrame({"date": days, "close": [100.0 + i for i in range(n)]})


def test_event_day_and_adjacent_windows() -> None:
    bars = _bars(date(2024, 1, 1), 40)
    event = bars["date"][20]  # an interior trading day is the FOMC day
    out = attach_event_features(bars, fomc_dates=[event]).sort("date")
    row = out.with_row_index("i")
    e = row.filter(pl.col("date") == event)["i"][0]
    assert out["fomc_t0"][e] is True
    assert out["fomc_tm1"][e - 1] is True   # day before
    assert out["fomc_tp1"][e + 1] is True   # day after
    assert out["fomc_t0"][e - 1] is False
    # window of +/-2 covers e-2..e+2
    assert out["fomc_win2"][e - 2] is True
    assert out["fomc_win2"][e - 3] is False
    assert out["fomc_win5"][e - 5] is True


def test_days_to_and_since_event() -> None:
    bars = _bars(date(2024, 1, 1), 30)
    event = bars["date"][10]
    out = attach_event_features(bars, fomc_dates=[event])
    assert out["days_to_next_fomc"][8] == 2
    assert out["days_to_next_fomc"][10] == 0
    assert out["days_since_last_fomc"][13] == 3


def test_no_events_yields_no_window_flags() -> None:
    bars = _bars(date(2024, 1, 1), 10)
    out = attach_event_features(bars, fomc_dates=[])
    assert out["fomc_t0"].sum() == 0
    assert out["fomc_win2"].sum() == 0


def test_month_and_quarter_end_flags() -> None:
    bars = _bars(date(2024, 3, 25), 12)  # spans the Mar/Apr (quarter) boundary
    out = attach_event_features(bars, fomc_dates=[]).sort("date")
    me = out.filter(pl.col("is_month_end"))["date"].to_list()
    # last March trading day present is 2024-03-29 (Mar 30/31 are weekend)
    assert date(2024, 3, 29) in me
    qe = out.filter(pl.col("is_quarter_end"))["date"].to_list()
    assert date(2024, 3, 29) in qe  # end of Q1


def test_earnings_season_membership() -> None:
    assert _in_earnings_season(date(2024, 1, 20)) is True
    assert _in_earnings_season(date(2024, 4, 30)) is True
    assert _in_earnings_season(date(2024, 3, 1)) is False
    assert _in_earnings_season(date(2024, 12, 1)) is False


def test_load_fomc_dates_from_manifest() -> None:
    dates = load_fomc_dates()
    assert len(dates) >= 160
    assert date(2008, 9, 16) in dates  # a known scheduled FOMC decision day
    assert dates == sorted(dates)
