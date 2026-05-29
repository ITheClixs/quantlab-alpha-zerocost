"""Build the event-calendar manifest for event_conditioned_macro_v1.

Sources scheduled FOMC decision dates (the only cleanly-sourceable macro event in
this environment) and records which event families are active vs deferred. CPI and
NFP release dates could NOT be sourced timestamp-clean here (ALFRED and BLS return
403 to automated fetches); approximating them (CPI mid-month / NFP first-Friday)
would violate the intake's timestamp-clean gate, so those families are recorded as
`deferred` rather than faked. Earnings-season and period-end are deterministic
rules computed at runtime from the trading-date index.

FOMC provenance:
- 2006-2018: github.com/tobiasi/FOMCscrape FOMC_dates.csv, `Scheduled==1` End dates
  (exactly 8/year; emergency meetings excluded by the flag).
- 2019-2026: federalreserve.gov FOMC calendars/historical pages (scheduled decision
  dates; 2020 emergency cuts 03-03 & 03-15 excluded as not ex-ante scheduled, the
  originally-scheduled 03-18 retained).

Usage:
    PYTHONPATH=src uv run python scripts/build_event_calendar.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

_CSV_URL = "https://raw.githubusercontent.com/tobiasi/FOMCscrape/master/FOMC_dates.csv"

# Scheduled FOMC decision (second-day) dates, 2019-2026, from the Federal Reserve.
# 2020 excludes the emergency cuts (2020-03-03, 2020-03-15); the ex-ante-scheduled
# 2020-03-18 meeting date is retained (it was on the published calendar).
_FOMC_2019_2026: tuple[str, ...] = (
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29",  # partial: meetings concluded as of 2026-05-30
)


def fetch_fomc_2006_2018() -> list[str]:
    text = urllib.request.urlopen(_CSV_URL, timeout=60).read().decode()  # noqa: S310 - fixed raw GH host
    out: list[str] = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 4 or not row[2].count("/") == 2:
            continue
        end, scheduled = row[2], row[3].strip()
        day, month, year = end.split("/")
        if scheduled == "1" and 2006 <= int(year) <= 2018:
            out.append(f"{year}-{month}-{day}")
    return sorted(out)


def main() -> int:
    fomc = sorted(set(fetch_fomc_2006_2018()) | set(_FOMC_2019_2026))
    per_year = dict(sorted(Counter(d[:4] for d in fomc).items()))
    digest = hashlib.sha256("\n".join(fomc).encode()).hexdigest()

    manifest = {
        "name": "event_conditioned_macro_v1",
        "built_utc": datetime.now(UTC).isoformat(),
        "intake": "docs/research/intake/2026-05-30-event-conditioned-macro-calendar-v1.md",
        "families": {
            "fomc": {
                "status": "active",
                "release_time_et": "14:00",
                "pre_open_release": False,
                "execution_note": "decision ~14:00 ET; daily position set at prior close from ex-ante schedule",
                "dates": fomc,
                "count": len(fomc),
                "per_year": per_year,
                "sha256": digest,
                "provenance": {
                    "2006-2018": f"{_CSV_URL} (Scheduled==1 End dates)",
                    "2019-2026": "federalreserve.gov FOMC calendars/historical (scheduled decision dates)",
                    "exclusions": "2020-03-03 and 2020-03-15 emergency cuts excluded (not ex-ante scheduled)",
                },
            },
            "cpi": {
                "status": "deferred",
                "reason": "ALFRED (rid=10) and BLS archive return 403 to automated fetch in this environment; "
                          "approximating mid-month release would violate the timestamp-clean gate.",
            },
            "nfp": {
                "status": "deferred",
                "reason": "ALFRED (rid=50) and BLS empsit archive return 403; first-Friday rule has documented "
                          "exceptions and is not timestamp-clean.",
            },
            "earnings_season": {
                "status": "deterministic_rule",
                "rule": "broad regime: trading days within the canonical reporting windows "
                        "(approx Jan 15-Feb 15, Apr 15-May 15, Jul 15-Aug 15, Oct 15-Nov 15)",
                "note": "calendar regime only; NOT single-stock earnings",
            },
            "period_end": {
                "status": "deterministic_rule",
                "rule": "last trading day of each month (month_end) and of each quarter (quarter_end), "
                        "derived from the SPY/QQQ trading-date index",
            },
        },
    }
    out_dir = Path("manifests/event_calendar")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "event_calendar_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"FOMC dates: {len(fomc)}  per-year: {per_year}")
    print(f"sha256(fomc): {digest}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
