"""Sharadar data-audit logic: delisted probe, CIK-mapping loss, coverage, actions.

Pure functions operating on loaded polars tables, so they are unit-testable on tiny
synthetic fixtures. The script `audit_sharadar_data.py` wires these to real tables
and the EDGAR universe and renders the report.
"""

from __future__ import annotations

import re
from typing import Any

import polars as pl

# Eight delisted/merged/failed probes with their 2019-2023 exit context.
EIGHT_NAMES: dict[str, str] = {
    "TWTR": "Twitter — acquired (Musk) Oct 2022",
    "CELG": "Celgene — merged into BMY 2019",
    "XLNX": "Xilinx — acquired by AMD 2022",
    "CERN": "Cerner — acquired by Oracle 2022",
    "ATVI": "Activision Blizzard — acquired by Microsoft 2023",
    "SIVB": "SVB Financial — failed Mar 2023",
    "FRC": "First Republic — failed May 2023",
    "AABA": "Altaba — liquidated 2019",
}
_DELIST_ACTIONS = ("delisted", "merger", "acquisition", "acquired", "liquidation", "bankruptcy")
_CONCENTRATION_PCT_KILL = 90.0


def _norm_name(name: str) -> str:
    n = (name or "").lower()
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    for suf in (" inc", " corp", " corporation", " co", " ltd", " plc", " holdings", " group",
                " company", " the", " sa", " nv", " /de/", " /mo/"):
        n = n.replace(suf, " ")
    return re.sub(r"\s+", " ", n).strip()


def eight_name_probe(tickers: pl.DataFrame | None, actions: pl.DataFrame | None,
                     sep: pl.DataFrame | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tk, context in EIGHT_NAMES.items():
        row: dict[str, Any] = {"ticker": tk, "context": context, "found": False,
                               "permaticker": None, "isdelisted": None, "last_price_date": None,
                               "delisting_action": None, "final_return_computable": False}
        if tickers is not None and "ticker" in tickers.columns:
            hit = tickers.filter(pl.col("ticker") == tk)
            if hit.height:
                row["found"] = True
                r0 = hit.row(0, named=True)
                row["permaticker"] = r0.get("permaticker")
                row["isdelisted"] = r0.get("isdelisted")
                row["last_price_date"] = str(r0.get("lastpricedate")) if r0.get("lastpricedate") is not None else None
        if actions is not None and {"ticker", "action"}.issubset(actions.columns):
            act = actions.filter(
                (pl.col("ticker") == tk)
                & pl.col("action").cast(pl.Utf8).str.to_lowercase().str.contains("|".join(_DELIST_ACTIONS))
            )
            if act.height:
                a0 = act.row(0, named=True)
                row["delisting_action"] = {"action": a0.get("action"), "date": str(a0.get("date")),
                                           "value": a0.get("value")}
        has_price = sep is not None and "ticker" in sep.columns and sep.filter(pl.col("ticker") == tk).height > 0
        # a final return is computable if we have a price path (last price) and/or a delisting action value
        row["final_return_computable"] = bool(has_price or row["delisting_action"])
        out.append(row)
    return out


def cik_mapping_loss(tickers: pl.DataFrame | None,
                     edgar: pl.DataFrame) -> dict[str, Any]:
    """edgar: DataFrame with columns cik (+ company). Estimate mappability to Sharadar."""
    total = int(edgar["cik"].n_unique()) if "cik" in edgar.columns else edgar.height
    if tickers is None:
        return {"method": "no_tickers_table", "total_edgar": total, "mapped": 0,
                "unmapped": total, "mapped_pct": 0.0, "passes_90pct": False, "unmapped_examples": []}
    if "cik" in tickers.columns:
        sh_ciks = set(int(x) for x in tickers["cik"].drop_nulls().to_list())
        ed_ciks = set(int(x) for x in edgar["cik"].drop_nulls().to_list())
        mapped = ed_ciks & sh_ciks
        unmapped = sorted(ed_ciks - sh_ciks)
        method = "direct_cik"
        n_total = len(ed_ciks)
        n_mapped = len(mapped)
        examples = unmapped[:10]
    else:
        # degraded: name-normalized bridge (rough estimate only)
        sh_names = {_norm_name(n) for n in tickers["name"].to_list()} if "name" in tickers.columns else set()
        ed = edgar.with_columns(pl.col("company").cast(pl.Utf8)) if "company" in edgar.columns else edgar
        ed_names = ed["company"].to_list() if "company" in ed.columns else []
        mapped_names = [n for n in ed_names if _norm_name(n) in sh_names]
        method = "name_bridge_degraded"
        n_total = len(ed_names) or total
        n_mapped = len(mapped_names)
        examples = [n for n in ed_names if _norm_name(n) not in sh_names][:10]
    pct = 100.0 * n_mapped / n_total if n_total else 0.0
    return {"method": method, "total_edgar": n_total, "mapped": n_mapped,
            "unmapped": n_total - n_mapped, "mapped_pct": round(pct, 1),
            "passes_90pct": pct >= _CONCENTRATION_PCT_KILL, "unmapped_examples": examples}


def coverage_check(sep: pl.DataFrame | None, windows: dict[str, tuple[str, str]]) -> dict[str, Any]:
    if sep is None or "date" not in sep.columns:
        return {w: {"covered": False, "reason": "no SEP table"} for w in windows}
    d = sep.with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("d"))
    raw_min, raw_max = d["d"].min(), d["d"].max()
    if raw_min is None or raw_max is None:
        return {w: {"covered": False, "reason": "empty SEP"} for w in windows}
    dmin, dmax = str(raw_min), str(raw_max)
    out: dict[str, Any] = {}
    for w, (lo, hi) in windows.items():
        in_win = d.filter((pl.col("d") >= lo) & (pl.col("d") <= hi))
        out[w] = {"covered": bool(dmin <= lo and dmax >= hi),
                  "data_min": dmin, "data_max": dmax,
                  "symbols_in_window": int(in_win["ticker"].n_unique()) if in_win.height else 0}
    return out


def actions_summary(actions: pl.DataFrame | None) -> dict[str, Any]:
    if actions is None or "action" not in actions.columns:
        return {"present": False, "by_action": {}}
    counts = (actions.group_by("action").len().sort("len", descending=True))
    return {"present": True, "by_action": {r["action"]: r["len"] for r in counts.to_dicts()},
            "has_splits": actions.filter(pl.col("action").cast(pl.Utf8).str.contains("split")).height > 0,
            "has_dividends": actions.filter(pl.col("action").cast(pl.Utf8).str.contains("dividend")).height > 0,
            "has_delistings": actions.filter(
                pl.col("action").cast(pl.Utf8).str.to_lowercase().str.contains("|".join(_DELIST_ACTIONS))
            ).height > 0}
