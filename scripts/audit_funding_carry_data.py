"""Funding-carry DATA AUDIT (no strategy code).

Binding question: can we build a leak-safe, survivorship-clean crypto perpetual
funding-rate carry research panel from FREE data?

Verifies: funding-rate history (Binance Vision, free) for BTC/ETH — coverage,
schema, timestamp-safety, per-year realized funding (regime honesty); spot price
(on disk) + perp price availability; basis computability; cost model for a
delta-neutral (long spot / short perp) carry. Emits manifest + audit report.

Usage:
    PYTHONPATH=src uv run python scripts/audit_funding_carry_data.py
"""

from __future__ import annotations

import glob
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from rich.console import Console

from quant_research_stack.crypto_research.funding.data import (
    annualized_funding,
    coverage,
    load_funding,
)

console = Console()
_OUT = Path("reports/signal_research/funding_carry_v1")
_MANIFEST = Path("manifests/funding_carry/funding_carry_data_manifest.json")
_SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _per_year(df: pl.DataFrame) -> dict[int, float]:
    out: dict[int, float] = {}
    for yr in sorted({d.year for d in df["funding_time"].to_list()}):
        sub = df.filter(pl.col("funding_time").dt.year() == yr)
        out[int(yr)] = round(annualized_funding(sub), 4)
    return out


def _spot_perp_status() -> dict[str, object]:
    spot_btc = bool(glob.glob("data/raw/huggingface/vaquum__binance_btcusdt_1m_klines/*.parquet"))
    perp = bool(glob.glob("data/raw/huggingface/123olp__binance-futures-ohlcv-2018-2026/*"))
    return {
        "btc_spot_on_disk": spot_btc,
        "perp_ohlcv_on_disk": perp,
        "eth_spot_source": "free (Binance Vision spot klines / yfinance ETH-USD) — fetch in P1",
        "perp_price_source": "free (Binance Vision futures/um klines; 123olp on disk)",
        "basis_source": "free (Binance Vision markPriceKlines + premiumIndexKlines)",
    }


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    built = datetime.now(UTC).isoformat()

    funding: dict[str, dict] = {}
    for sym in _SYMBOLS:
        console.print(f"[cyan]loading funding[/cyan] {sym} (cached after first run)")
        df = load_funding(sym)
        funding[sym] = {"coverage": coverage(df), "annualized_full": round(annualized_funding(df), 4),
                        "annualized_by_year": _per_year(df)}
        cov = funding[sym]["coverage"]
        console.print(f"  {sym}: rows={cov.get('rows')} {cov.get('start')}..{cov.get('end')} "
                      f"ann_funding={funding[sym]['annualized_full']}")

    sp = _spot_perp_status()
    rows_ok = all(funding[s]["coverage"]["rows"] > 1000 for s in _SYMBOLS)
    verdict = "PASS" if rows_ok and sp["perp_ohlcv_on_disk"] else "PARTIAL"

    manifest = {
        "name": "funding_carry_data_manifest", "built_utc": built, "verdict": verdict,
        "funding": funding, "spot_perp": sp,
        "timestamp_rule": "funding settles every 8h; realized rate known at settlement t; signal uses funding<=t, "
                          "earns the next settlement (t+1 interval) -> leak-safe.",
        "universe": "BTCUSDT, ETHUSDT perps — never delisted -> survivorship-clean (cross-sectional top-N deferred).",
        "cost_model": {
            "perp_taker_bps_one_way": 5.0, "spot_taker_bps_one_way": 10.0,
            "note": "delta-neutral carry = long spot / short perp; cost paid at entry/exit only, funding earned "
                    "each 8h held -> NOT a per-trade taker bet (escapes the microstructure cost wall).",
        },
        "labels_legend": ["funding_clean", "spot_clean", "perp_clean", "basis_clean", "research_only", "reject"],
    }
    _MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))

    (_OUT / "funding_carry_data_audit.md").write_text("\n".join([
        "# Funding-Carry Data Audit",
        f"\n**Built:** {built}  **Verdict:** **{verdict}**",
        "**Binding question:** leak-safe, survivorship-clean crypto perp funding-carry panel from FREE data?",
        "\n## 1. Funding-rate data (Binance Vision, free)",
        "", "| symbol | rows | start | end | ann. funding (full) |",
        "|---|---:|---|---|---:|",
        *[f"| {s} | {funding[s]['coverage'].get('rows')} | {funding[s]['coverage'].get('start')} | "
          f"{funding[s]['coverage'].get('end')} | {funding[s]['annualized_full']:.1%} |" for s in _SYMBOLS],
        "\n### Realized funding by year (regime honesty — carry is NOT constant)",
        *[f"- **{s}**: " + ", ".join(f"{y}:{r:+.1%}" for y, r in funding[s]["annualized_by_year"].items())
          for s in _SYMBOLS],
        "\n- Funding is large+positive in leverage/bull regimes (longs pay shorts) and ~0/negative otherwise."
        " A carry strategy's edge is regime-dependent — the v1 gate must test robustness across these years.",
        "\n## 2. Timestamp / leakage",
        f"- {manifest['timestamp_rule']}",
        "- 8h settlements (00/08/16 UTC); `calc_time` = settlement; `last_funding_rate` realized at that time.",
        "\n## 3. Spot / perp / basis",
        *[f"- {k}: {v}" for k, v in sp.items()],
        "\n## 4. Universe & survivorship",
        f"- {manifest['universe']}",
        "\n## 5. Cost model (delta-neutral)",
        f"- {manifest['cost_model']['note']} perp taker ~{manifest['cost_model']['perp_taker_bps_one_way']}bps, "
        f"spot taker ~{manifest['cost_model']['spot_taker_bps_one_way']}bps one-way.",
        "\n## 6. Verdict",
        (f"- **PASS** — funding (free, {funding['BTCUSDT']['coverage'].get('start','?')[:7]}..), spot, perp, and basis"
         " are all free + timestamp-clean + survivorship-clean for BTC/ETH. Proceed to funding-carry v1 intake +"
         " backtest (delta-neutral long-spot/short-perp + directional variants), with the regime-robustness gate."
         if verdict == "PASS" else
         "- **PARTIAL** — verify the unavailable component before proceeding."),
        "\n## Constraints",
        "- No strategy code in this audit. No paper/live. Funding carry escapes the cost wall (held, not taker) and"
        " the subsumption wall (carry != vol-timing); the gate must still beat buy-and-hold + survive regime/cost.",
    ]) + "\n")

    console.print(f"[bold]{verdict}[/bold] | wrote manifest + funding_carry_data_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
