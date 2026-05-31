"""Loader for Binance USDT-M futures bookTicker public archives.

bookTicker carries best bid/ask price and size per book update — the L1-quote
channel the perps pipeline (`build_l1_features`, `run_event_backtest`) consumes.
Source: https://data.binance.vision/data/futures/um/daily/bookTicker/<SYMBOL>/

Free, downloadable (no entitlement wall — unlike Massive.com flat files). Files
are large decompressed (tens of millions of rows/day), so all readers accept a
``max_rows`` cap and the reader streams the zip member line-by-line to bound
memory.
"""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Any, BinaryIO

import polars as pl

_BASE = "https://data.binance.vision/data/futures/um/daily/bookTicker"

BOOKTICKER_RAW_COLUMNS: list[str] = [
    "update_id",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "transaction_time",
    "event_time",
]

NORMALIZED_COLUMNS: list[str] = [
    "symbol",
    "event_time",
    "best_bid",
    "best_ask",
    "best_bid_size",
    "best_ask_size",
]

_SCHEMA_OVERRIDES: dict[str, Any] = {
    "update_id": pl.Int64,
    "best_bid_price": pl.Float64,
    "best_bid_qty": pl.Float64,
    "best_ask_price": pl.Float64,
    "best_ask_qty": pl.Float64,
    "transaction_time": pl.Int64,
    "event_time": pl.Int64,
}


def _raw_frame_from_lines(lines: list[str], max_rows: int | None) -> pl.DataFrame:
    rows = [line for line in lines if line]
    if not rows:
        return pl.DataFrame()
    has_header = rows[0].lower().startswith("update_id")
    data = rows[1:] if has_header else rows
    if max_rows is not None:
        data = data[:max_rows]
    body = ("\n".join(data) + "\n").encode("utf-8")
    return pl.read_csv(
        io.BytesIO(body),
        has_header=False,
        new_columns=BOOKTICKER_RAW_COLUMNS,
        schema_overrides=_SCHEMA_OVERRIDES,
    )


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def bookticker_day_url(symbol: str, date: str) -> str:
    sym = symbol.upper()
    return f"{_BASE}/{sym}/{sym}-bookTicker-{date}.zip"


def normalize_bookticker(raw: pl.DataFrame, *, symbol: str, max_rows: int | None = None) -> pl.DataFrame:
    """Map raw bookTicker columns to the perps L1 event schema.

    Drops crossed/locked books (best_bid >= best_ask) and non-positive prices,
    sorts by event_time, and optionally caps to the first ``max_rows`` rows.
    """
    missing = [column for column in BOOKTICKER_RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"bookTicker frame missing columns: {missing}")
    out = (
        raw.lazy()
        .with_columns(
            pl.lit(symbol.upper()).alias("symbol"),
            (pl.col("event_time").cast(pl.Int64) * 1000).cast(pl.Datetime("us", "UTC")).alias("event_time"),
            pl.col("best_bid_price").cast(pl.Float64).alias("best_bid"),
            pl.col("best_ask_price").cast(pl.Float64).alias("best_ask"),
            pl.col("best_bid_qty").cast(pl.Float64).alias("best_bid_size"),
            pl.col("best_ask_qty").cast(pl.Float64).alias("best_ask_size"),
        )
        .filter(
            (pl.col("best_bid") > 0.0)
            & (pl.col("best_ask") > 0.0)
            & (pl.col("best_bid") < pl.col("best_ask"))
        )
        .select(NORMALIZED_COLUMNS)
        .sort("event_time")
        .collect()
    )
    if max_rows is not None:
        out = out.head(max_rows)
    return out


def read_bookticker_zip_bytes(blob: bytes, *, max_rows: int | None = None) -> pl.DataFrame:
    """Read the single CSV member of a bookTicker zip into the raw schema.

    Streams the member line-by-line so a ``max_rows`` cap bounds memory even
    though the decompressed CSV can be multiple GB. Handles files with or
    without a header row (Binance has shipped both).
    """
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8")
            first = text.readline()
            has_header = first.lower().startswith("update_id")
            lines: list[str] = []
            if not has_header and first:
                lines.append(first)
            limit = max_rows if max_rows is not None else None
            for line in text:
                if limit is not None and len(lines) >= limit:
                    break
                lines.append(line)
    body = "".join(lines)
    return pl.read_csv(
        io.BytesIO(body.encode("utf-8")),
        has_header=False,
        new_columns=BOOKTICKER_RAW_COLUMNS,
        schema_overrides={"best_bid_price": pl.Float64, "best_ask_price": pl.Float64,
                          "best_bid_qty": pl.Float64, "best_ask_qty": pl.Float64,
                          "event_time": pl.Int64, "transaction_time": pl.Int64, "update_id": pl.Int64},
    )


def read_bookticker_zip_stream(stream: BinaryIO, *, max_rows: int | None = None) -> pl.DataFrame:
    """Read a bookTicker zip from a forward-only byte stream, stopping early.

    Parses the zip's first local-file header, inflates the deflate payload
    incrementally, and stops once ``max_rows`` data lines are available — so a
    capped read over an HTTP response transfers only the few MB it needs instead
    of the whole multi-tens-of-MB daily file. Assumes a single stored/deflated
    member (the Binance bookTicker layout).
    """
    if _read_exact(stream, 4) != b"PK\x03\x04":
        raise ValueError("stream is not a zip local-file header")
    header = _read_exact(stream, 26)  # remaining 26 bytes of the 30-byte local header
    method = struct.unpack("<H", header[4:6])[0]
    name_len = struct.unpack("<H", header[22:24])[0]
    extra_len = struct.unpack("<H", header[24:26])[0]
    _read_exact(stream, name_len + extra_len)  # skip filename + extra field

    decompressor = zlib.decompressobj(-15) if method == 8 else None
    leftover = ""
    lines: list[str] = []
    limit = (max_rows + 1) if max_rows is not None else None  # +1 to absorb a header row
    while True:
        compressed = stream.read(1 << 16)
        if not compressed:
            tail = decompressor.flush() if decompressor is not None else b""
            text = leftover + tail.decode("utf-8", errors="replace")
            lines.extend(text.split("\n"))
            break
        raw = decompressor.decompress(compressed) if decompressor is not None else compressed
        text = leftover + raw.decode("utf-8", errors="replace")
        parts = text.split("\n")
        leftover = parts.pop()
        lines.extend(parts)
        if limit is not None and len(lines) >= limit:
            break
    return _raw_frame_from_lines(lines, max_rows)


def load_bookticker_day(
    symbol: str, date: str, *, max_rows: int | None = None, read_timeout: float = 90.0
) -> pl.DataFrame:
    """Stream a capped bookTicker day directly from Binance and normalize it.

    Avoids downloading the full daily file: only the bytes needed for
    ``max_rows`` are transferred. Returns the perps L1 event schema.
    """
    import urllib.request

    url = bookticker_day_url(symbol, date)
    with urllib.request.urlopen(url, timeout=read_timeout) as response:  # noqa: S310 - fixed Binance host
        raw = read_bookticker_zip_stream(response, max_rows=max_rows)
    return normalize_bookticker(raw, symbol=symbol)


def download_bookticker_day(
    symbol: str,
    date: str,
    dest_dir: Path | str,
    *,
    overwrite: bool = False,
    chunk_bytes: int = 1 << 20,
    max_attempts: int = 4,
    read_timeout: float = 60.0,
) -> Path:
    """Download one daily bookTicker zip to ``dest_dir`` (skip if already present).

    Streams in chunks to a ``.part`` file then atomically renames, so an
    interrupted/slow download never leaves a truncated zip in the cache. Retries
    a few times because the Binance public endpoint can be slow or flaky.
    """
    import time
    import urllib.request

    dest = Path(dest_dir).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{symbol.upper()}-bookTicker-{date}.zip"
    if target.exists() and not overwrite:
        return target
    url = bookticker_day_url(symbol, date)
    part = target.with_suffix(".zip.part")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=read_timeout) as response, part.open("wb") as handle:  # noqa: S310 - fixed Binance host
                while True:
                    chunk = response.read(chunk_bytes)
                    if not chunk:
                        break
                    handle.write(chunk)
            part.replace(target)
            return target
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < max_attempts:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"failed to download {url} after {max_attempts} attempts: {last_error}")
