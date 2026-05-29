"""Mode A — aggressor-signed trade-flow features on Binance spot aggTrades.

The audited aggTrades stream (`trade_only_clean`) has trade prints only — no
bid/ask. This module:
  1. loads/streams spot aggTrades (`data/spot/daily/aggTrades/<SYMBOL>/`),
  2. signs each trade by aggressor (is_buyer_maker=True -> seller-initiated),
  3. builds causal order-flow-imbalance features, and
  4. reconstructs midprice from the trade price and synthesizes a *modeled*
     constant half-spread (since no quotes exist), so the existing
     `perps.run_event_backtest` can charge a realistic taker spread crossing.

The synthesized spread is a MODEL, not observed data — flagged in every report.
InformationSource.microstructure_tick. research_only.
"""

from __future__ import annotations

import io
import struct
import zlib
from typing import Any, BinaryIO

import polars as pl

from quant_research_stack.crypto_research.perps.binance_bookticker import _read_exact

_SPOT_BASE = "https://data.binance.vision/data/spot/daily/aggTrades"

AGGTRADES_RAW_COLUMNS: list[str] = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
]

NORMALIZED_COLUMNS: list[str] = ["symbol", "event_time", "price", "size", "aggressor_sign"]


def aggtrades_day_url(symbol: str, date: str) -> str:
    sym = symbol.upper()
    return f"{_SPOT_BASE}/{sym}/{sym}-aggTrades-{date}.zip"


def normalize_aggtrades(raw: pl.DataFrame, *, symbol: str, max_rows: int | None = None) -> pl.DataFrame:
    """Sign trades by aggressor and build a strictly-increasing unique event_time.

    Binance packs many trades into the same millisecond; we order by the
    monotonic agg_trade_id and bump within-ms duplicates by 1µs so (symbol,
    event_time) is a unique key (needed for the post-prediction join-back).
    ``aggressor_sign``: +1 buyer-initiated, -1 seller-initiated.
    """
    missing = [column for column in AGGTRADES_RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"aggTrades frame missing columns: {missing}")
    maker = pl.col("is_buyer_maker").cast(pl.Utf8).str.to_lowercase().is_in(["true", "1"])
    out = (
        raw.lazy()
        .with_columns(
            pl.lit(symbol.upper()).alias("symbol"),
            pl.col("price").cast(pl.Float64).alias("price"),
            pl.col("quantity").cast(pl.Float64).alias("size"),
            # buyer is maker  => trade was seller-initiated => aggressor sells => -1
            pl.when(maker).then(-1.0).otherwise(1.0).alias("aggressor_sign"),
            (pl.col("transact_time").cast(pl.Int64) * 1000).cast(pl.Datetime("us", "UTC")).alias("_t"),
        )
        .filter((pl.col("price") > 0.0) & (pl.col("size") > 0.0))
        .sort(["symbol", "agg_trade_id"])
        .with_columns(
            (pl.col("_t") + pl.duration(microseconds=pl.int_range(0, pl.len()).over("symbol", "_t"))).alias("event_time")
        )
        .select(NORMALIZED_COLUMNS)
        .sort(["symbol", "event_time"])
        .collect()
    )
    if max_rows is not None:
        out = out.head(max_rows)
    return out


def _stream_aggtrades_zip(stream: BinaryIO, *, max_rows: int | None) -> pl.DataFrame:
    """Inflate the first member of an aggTrades zip incrementally, stop early."""
    if _read_exact(stream, 4) != b"PK\x03\x04":
        raise ValueError("stream is not a zip local-file header")
    header = _read_exact(stream, 26)
    method = struct.unpack("<H", header[4:6])[0]
    name_len = struct.unpack("<H", header[22:24])[0]
    extra_len = struct.unpack("<H", header[24:26])[0]
    _read_exact(stream, name_len + extra_len)
    decompressor = zlib.decompressobj(-15) if method == 8 else None
    leftover = ""
    lines: list[str] = []
    limit = (max_rows + 1) if max_rows is not None else None
    while True:
        compressed = stream.read(1 << 16)
        if not compressed:
            tail = decompressor.flush() if decompressor is not None else b""
            lines.extend((leftover + tail.decode("utf-8", errors="replace")).split("\n"))
            break
        raw = decompressor.decompress(compressed) if decompressor is not None else compressed
        parts = (leftover + raw.decode("utf-8", errors="replace")).split("\n")
        leftover = parts.pop()
        lines.extend(parts)
        if limit is not None and len(lines) >= limit:
            break
    rows = [line for line in lines if line]
    if not rows:
        return pl.DataFrame()
    has_header = rows[0].lower().startswith("agg_trade_id") or rows[0].lower().startswith("aggtradeid")
    data = rows[1:] if has_header else rows
    if max_rows is not None:
        data = data[:max_rows]
    schema: dict[str, Any] = {
        "agg_trade_id": pl.Int64, "price": pl.Float64, "quantity": pl.Float64,
        "first_trade_id": pl.Int64, "last_trade_id": pl.Int64, "transact_time": pl.Int64,
        "is_buyer_maker": pl.Utf8, "is_best_match": pl.Utf8,
    }
    return pl.read_csv(
        io.BytesIO(("\n".join(data) + "\n").encode("utf-8")),
        has_header=False, new_columns=AGGTRADES_RAW_COLUMNS, schema_overrides=schema,
    )


def load_aggtrades_day(symbol: str, date: str, *, max_rows: int | None = None, read_timeout: float = 90.0) -> pl.DataFrame:
    """Stream a capped aggTrades day from Binance and normalize it."""
    import urllib.request

    url = aggtrades_day_url(symbol, date)
    with urllib.request.urlopen(url, timeout=read_timeout) as response:  # noqa: S310 - fixed Binance host
        raw = _stream_aggtrades_zip(response, max_rows=max_rows)
    return normalize_aggtrades(raw, symbol=symbol)


def trade_flow_feature_columns(windows: tuple[int, ...]) -> list[str]:
    cols = ["price_return_1"]
    for window in sorted(set(windows)):
        cols += [f"ofi_{window}", f"ret_{window}", f"realized_vol_{window}", f"signed_count_imb_{window}"]
    return cols


def build_trade_flow_features(
    trades: pl.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5, 15, 60),
    windows: tuple[int, ...] = (10, 50, 200),
    half_spread_bps: float = 1.0,
) -> pl.DataFrame:
    """Causal order-flow features + future-return labels + a modeled-spread quote.

    Midprice is the trade price (no resting book). Features use only current and
    past rows. ``half_spread_bps`` synthesizes best_bid/ask = price * (1 -/+ hs)
    so the taker backtest pays a realistic spread crossing.
    """
    required = {"symbol", "event_time", "price", "size", "aggressor_sign"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"missing trade columns: {sorted(missing)}")
    hs = half_spread_bps * 1e-4

    out = trades.sort(["symbol", "event_time"]).with_columns(
        (pl.col("size") * pl.col("aggressor_sign")).alias("signed_size"),
        (pl.col("price") / pl.col("price").shift(1).over("symbol") - 1.0).alias("price_return_1"),
    )
    feat_exprs: list[pl.Expr] = []
    for window in sorted(set(windows)):
        vol = pl.col("size").rolling_sum(window_size=window, min_samples=1).over("symbol")
        signed_vol = pl.col("signed_size").rolling_sum(window_size=window, min_samples=1).over("symbol")
        feat_exprs += [
            pl.when(vol > 0.0).then(signed_vol / vol).otherwise(None).alias(f"ofi_{window}"),
            (pl.col("price") / pl.col("price").shift(window).over("symbol") - 1.0).alias(f"ret_{window}"),
            pl.col("price_return_1").rolling_std(window_size=window, min_samples=2).over("symbol").alias(f"realized_vol_{window}"),
            pl.col("aggressor_sign").rolling_mean(window_size=window, min_samples=1).over("symbol").alias(f"signed_count_imb_{window}"),
        ]
    out = out.with_columns(feat_exprs)

    # Modeled constant-spread quotes + forward labels.
    quote_exprs: list[pl.Expr] = [
        (pl.col("price") * (1.0 - hs)).alias("best_bid"),
        (pl.col("price") * (1.0 + hs)).alias("best_ask"),
        pl.lit(2.0 * hs).alias("relative_spread"),
        pl.lit(1.0).alias("best_bid_size"),
        pl.lit(1.0).alias("best_ask_size"),
    ]
    for horizon in sorted(set(horizons)):
        future_price = pl.col("price").shift(-horizon).over("symbol")
        quote_exprs += [
            (future_price / pl.col("price") - 1.0).alias(f"future_mid_return_{horizon}"),
            (future_price * (1.0 - hs)).alias(f"future_best_bid_{horizon}"),
            (future_price * (1.0 + hs)).alias(f"future_best_ask_{horizon}"),
        ]
    return out.with_columns(quote_exprs)
