from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime

import polars as pl
import pytest

from quant_research_stack.crypto_research.perps.trade_flow import (
    AGGTRADES_RAW_COLUMNS,
    _stream_aggtrades_zip,
    aggtrades_day_url,
    build_trade_flow_features,
    normalize_aggtrades,
    trade_flow_feature_columns,
)

_RAW = pl.DataFrame(
    {
        "agg_trade_id": [10, 11, 12, 13],
        "price": [100.0, 100.0, 101.0, 102.0],
        "quantity": [1.0, 2.0, 1.5, 0.5],
        "first_trade_id": [1, 2, 3, 4],
        "last_trade_id": [1, 2, 3, 4],
        "transact_time": [1711929600000, 1711929600000, 1711929600001, 1711929600002],
        "is_buyer_maker": ["True", "False", "False", "True"],
        "is_best_match": ["True", "True", "True", "True"],
    }
)


def test_normalize_signs_aggressor_and_makes_unique_times() -> None:
    out = normalize_aggtrades(_RAW, symbol="btcusdt")
    assert out.columns == ["symbol", "event_time", "price", "size", "aggressor_sign"]
    assert out["symbol"][0] == "BTCUSDT"
    # buyer-maker=True -> seller initiated -> -1 ; False -> +1
    assert out["aggressor_sign"].to_list() == [-1.0, 1.0, 1.0, -1.0]
    # first two share a ms -> strictly increasing unique event_time
    times = out["event_time"].to_list()
    assert times[0] == datetime.fromtimestamp(1711929600000 / 1000.0, tz=UTC)
    assert times[1] > times[0]
    assert len(set(times)) == 4


def test_normalize_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        normalize_aggtrades(_RAW.drop("price"), symbol="BTCUSDT")


def test_day_url_points_at_spot_aggtrades() -> None:
    url = aggtrades_day_url("BTCUSDT", "2024-04-01")
    assert url.endswith("/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-04-01.zip")


def test_feature_columns_listing() -> None:
    cols = trade_flow_feature_columns((10, 50))
    assert "price_return_1" in cols
    assert {"ofi_10", "ret_10", "realized_vol_10", "signed_count_imb_10"} <= set(cols)
    assert {"ofi_50", "ret_50"} <= set(cols)


def test_features_are_causal_and_synthesize_spread() -> None:
    norm = normalize_aggtrades(_RAW, symbol="BTCUSDT")
    feats = build_trade_flow_features(norm, horizons=(1, 2), windows=(2,), half_spread_bps=1.0)
    feats = feats.sort("event_time")
    # future label uses the FUTURE price
    assert feats["future_mid_return_1"][0] == pytest.approx(feats["price"][1] / feats["price"][0] - 1.0)
    # past micro-return is null at the first row
    assert feats["price_return_1"][0] is None
    # modeled spread: best_bid/ask straddle price by 1 bp
    assert feats["best_ask"][0] == pytest.approx(feats["price"][0] * (1.0 + 1e-4))
    assert feats["best_bid"][0] == pytest.approx(feats["price"][0] * (1.0 - 1e-4))
    assert feats["relative_spread"][0] == pytest.approx(2e-4)
    # OFI in [-1, 1]
    ofi = [v for v in feats["ofi_2"].to_list() if v is not None]
    assert all(-1.0 <= v <= 1.0 for v in ofi)


def test_stream_reader_parses_headerless_aggtrades_zip() -> None:
    csv = (
        "10,100.0,1.0,1,1,1711929600000,True,True\n"
        "11,100.0,2.0,2,2,1711929600000,False,True\n"
        "12,101.0,1.5,3,3,1711929600001,False,True\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("BTCUSDT-aggTrades-2024-04-01.csv", csv)
    df = _stream_aggtrades_zip(io.BytesIO(buf.getvalue()), max_rows=2)
    assert df.columns == AGGTRADES_RAW_COLUMNS
    assert df.height == 2
    assert df["price"][0] == 100.0
