from __future__ import annotations

import json

import polars as pl
import pytest

from quant_research_stack.data.sharadar.audit import cik_mapping_loss, eight_name_probe
from quant_research_stack.data.sharadar.loaders import load_table
from quant_research_stack.data.sharadar.manifest import build_manifest, write_manifest
from quant_research_stack.data.sharadar.return_panel import build_return_panel
from quant_research_stack.data.sharadar.schema import SchemaError, validate_schema


def _sep() -> pl.DataFrame:
    # AAA trades all days; ZZZ (delisted) stops early -> must NOT be dropped
    rows = []
    for d in ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]:
        rows.append({"ticker": "AAA", "date": d, "open": 10.0, "high": 11.0, "low": 9.0,
                     "close": 10.0, "volume": 1000, "closeadj": 9.5, "dividends": 0.0})
    for d in ["2020-01-02", "2020-01-03"]:  # ZZZ delists after 01-03
        rows.append({"ticker": "ZZZ", "date": d, "open": 5.0, "high": 5.0, "low": 5.0,
                     "close": 5.0, "volume": 500, "closeadj": 5.0, "dividends": 0.0})
    return pl.DataFrame(rows)


def _tickers() -> pl.DataFrame:
    return pl.DataFrame({
        "permaticker": [100, 200, 300], "ticker": ["AAA", "ZZZ", "TWTR"],
        "name": ["Alpha Inc", "Zeta Corp", "Twitter, Inc."],
        "isdelisted": ["N", "Y", "Y"], "lastpricedate": ["2025-01-01", "2020-01-03", "2022-10-27"],
        "cik": [111, 222, 333],
    })


def _actions() -> pl.DataFrame:
    return pl.DataFrame({
        "date": ["2020-01-03", "2021-06-01", "2022-10-27"],
        "action": ["delisted", "split", "merger"],
        "ticker": ["ZZZ", "AAA", "TWTR"], "value": [None, 2.0, 54.20],
    })


def test_missing_required_columns_fails() -> None:
    bad = _sep().drop("close")
    with pytest.raises(SchemaError):
        validate_schema(bad, "SEP")


def test_unknown_table_fails() -> None:
    with pytest.raises(SchemaError):
        validate_schema(_sep(), "NOPE")


def test_load_and_manifest_writes_hash(tmp_path) -> None:
    (tmp_path / "sep.parquet").write_bytes(b"")  # placeholder to be overwritten
    _sep().write_parquet(tmp_path / "sep.parquet")
    _tickers().write_parquet(tmp_path / "tickers.parquet")
    loaded = {"SEP": load_table(tmp_path, "SEP"), "TICKERS": load_table(tmp_path, "TICKERS")}
    loaded = {k: v for k, v in loaded.items() if v is not None}
    assert loaded["SEP"].metadata["sha256"]
    assert loaded["SEP"].metadata["rows"] == 6
    manifest = build_manifest(loaded)
    out = write_manifest(manifest, tmp_path / "m.json")
    written = json.loads(out.read_text())
    assert written["tables"]["SEP"]["sha256"] == loaded["SEP"].metadata["sha256"]
    assert written["status"] in ("partial", "complete")  # ACTIONS missing -> partial


def test_eight_name_probe_report_produced() -> None:
    probe = eight_name_probe(_tickers(), _actions(), _sep())
    assert len(probe) == 8
    twtr = next(r for r in probe if r["ticker"] == "TWTR")
    assert twtr["found"] is True and twtr["isdelisted"] == "Y"
    assert twtr["delisting_action"]["action"] == "merger"


def test_cik_mapping_report_direct() -> None:
    edgar = pl.DataFrame({"cik": [111, 222, 999], "company": ["Alpha Inc", "Zeta Corp", "Ghost Co"]})
    rep = cik_mapping_loss(_tickers(), edgar)
    assert rep["method"] == "direct_cik"
    assert rep["total_edgar"] == 3 and rep["mapped"] == 2
    assert rep["mapped_pct"] == pytest.approx(66.7, abs=0.1)
    assert rep["passes_90pct"] is False


def test_cik_mapping_name_bridge_when_no_cik() -> None:
    edgar = pl.DataFrame({"cik": [111, 222], "company": ["Alpha Inc", "Zeta Corp"]})
    rep = cik_mapping_loss(_tickers().drop("cik"), edgar)
    assert rep["method"] == "name_bridge_degraded"
    assert rep["mapped"] >= 1


def test_return_panel_keeps_delisted_and_no_survival_filter() -> None:
    panel = build_return_panel(_sep(), tickers=_tickers(), actions=_actions())
    # ZZZ delisted after 2020-01-03 (before AAA's last date) must still be present
    assert "ZZZ" in panel["ticker"].to_list()
    assert panel["ticker"].n_unique() == 2
    assert "ret_adj" in panel.columns and "permaticker" in panel.columns
    # permaticker stable id attached (string-normalized for a consistent index)
    assert str(panel.filter(pl.col("ticker") == "ZZZ")["permaticker"][0]) == "200"
