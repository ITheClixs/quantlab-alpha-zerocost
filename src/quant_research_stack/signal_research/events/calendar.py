"""Trading-day event-window features from the ex-ante event calendar.

Every feature is computable at the prior close from a fixed, known-in-advance
schedule, so conditioning on it introduces no look-ahead. Windows are measured in
**trading days** relative to the event (t-1 = the trading day before, t+1 = the
trading day after), which is the leak-free convention: a pre-open release (CPI/NFP,
when added) must therefore be acted on at the prior close, never the event-day open.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

MANIFEST_PATH = Path("manifests/event_calendar/event_calendar_manifest.json")

# Canonical broad reporting windows (month, day) inclusive — a calendar REGIME,
# not single-stock earnings.
EARNINGS_WINDOWS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((1, 15), (2, 15)),
    ((4, 15), (5, 15)),
    ((7, 15), (8, 15)),
    ((10, 15), (11, 15)),
)

_FAR = 9999


def load_fomc_dates(manifest_path: Path | str = MANIFEST_PATH) -> list[date]:
    manifest = json.loads(Path(manifest_path).read_text())
    return [datetime.strptime(s, "%Y-%m-%d").date() for s in manifest["families"]["fomc"]["dates"]]


def _in_earnings_season(d: date) -> bool:
    md = (d.month, d.day)
    return any(start <= md <= end for start, end in EARNINGS_WINDOWS)


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def attach_event_features(
    bars: pl.DataFrame, *, fomc_dates: Iterable[date], date_col: str = "date",
) -> pl.DataFrame:
    """Attach leak-free FOMC-window, period-end, and earnings-season features.

    Adds (per row, all computable at prior close):
      fomc_t0/fomc_tm1/fomc_tp1, fomc_win2 (t-2..t+2), fomc_win5 (t-5..t+5 diag),
      days_to_next_fomc, days_since_last_fomc, is_month_end, is_quarter_end,
      in_earnings_season.
    """
    if date_col not in bars.columns:
        raise ValueError(f"missing date column {date_col!r}")
    df = bars.sort(date_col)
    dates = [_as_date(v) for v in df[date_col].to_list()]
    n = len(dates)
    if n == 0:
        raise ValueError("empty bars frame")

    position = {d: i for i, d in enumerate(dates)}
    event_pos = np.array(sorted(position[d] for d in set(fomc_dates) if d in position), dtype=np.int64)
    idx = np.arange(n, dtype=np.int64)

    if event_pos.size:
        j = np.searchsorted(event_pos, idx, side="left")
        right_pos = np.where(j < event_pos.size, event_pos[np.clip(j, 0, event_pos.size - 1)], _FAR)
        left_pos = np.where(j > 0, event_pos[np.clip(j - 1, 0, event_pos.size - 1)], -_FAR)
        dist_next = right_pos - idx        # trading days until next event (>=0)
        dist_prev = idx - left_pos         # trading days since last event (>=0)
        nearest = np.minimum(dist_next, dist_prev)
        is_event = np.zeros(n, dtype=bool)
        is_event[event_pos] = True
    else:
        dist_next = np.full(n, _FAR, dtype=np.int64)
        dist_prev = np.full(n, _FAR, dtype=np.int64)
        nearest = np.full(n, _FAR, dtype=np.int64)
        is_event = np.zeros(n, dtype=bool)

    months = np.array([d.year * 12 + d.month for d in dates], dtype=np.int64)
    quarters = np.array([d.year * 4 + (d.month - 1) // 3 for d in dates], dtype=np.int64)
    is_month_end = np.append(months[:-1] != months[1:], True) if n > 1 else np.array([True])
    is_quarter_end = np.append(quarters[:-1] != quarters[1:], True) if n > 1 else np.array([True])

    return df.with_columns(
        pl.Series("fomc_t0", is_event),
        pl.Series("fomc_tm1", dist_next == 1),
        pl.Series("fomc_tp1", dist_prev == 1),
        pl.Series("fomc_win2", nearest <= 2),
        pl.Series("fomc_win5", nearest <= 5),
        pl.Series("days_to_next_fomc", np.minimum(dist_next, _FAR)),
        pl.Series("days_since_last_fomc", np.minimum(dist_prev, _FAR)),
        pl.Series("is_month_end", is_month_end),
        pl.Series("is_quarter_end", is_quarter_end),
        pl.Series("in_earnings_season", [_in_earnings_season(d) for d in dates]),
    )
