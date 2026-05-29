"""Futures carry / term-structure DATA AUDIT (no strategy code).

Answers one question: can we compute futures carry / roll yield from clean,
timestamp-safe front AND deferred contract data, without relying on a single
back-adjusted continuous series?

Findings (probed 2026-05-30, this environment):
- Massive flat files us_futures_{cme,cbot,comex,nymex}/{session,minute,trades,
  quotes}_v1: catalogue LISTS (2017-2024) but GetObject returns 403 (paid
  entitlement). The one source that would give per-contract curve data is not
  downloadable.
- yfinance: continuous front-month only ('CL=F','ES=F','GC=F' work). Dated
  contracts ('CLF26.NYM') return empty/404; 'ESM26.CME' just aliases the
  continuous front (same close as ES=F). No distinct next-contract prices, no
  expiry metadata -> cannot build a curve.
- ETF proxies download cleanly via yfinance (VXX/VXZ VIX term structure;
  USO/USL oil curve ladder; UNG). These EMBED roll yield but do not expose
  front/next contract prices + expiries -> proxy_not_native_futures.

Emits manifests/futures/futures_data_manifest.json and
reports/signal_research/futures_carry_v1/futures_data_audit.md.
Verdict: PARTIAL_PASS (no native curve_clean; proxy-only research test possible).

Usage:
    PYTHONPATH=src uv run python scripts/build_futures_data_audit.py [--live]
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_MANIFEST = Path("manifests/futures/futures_data_manifest.json")
_AUDIT = Path("reports/signal_research/futures_carry_v1/futures_data_audit.md")

# Per-market data-quality classification from the §3 requirement.
# label in {curve_clean, continuous_only, proxy_not_native_futures, metadata_incomplete, reject}
MARKETS: dict[str, dict] = {
    "ES (e-mini S&P futures)": {"class": "equity_index", "label": "continuous_only",
        "front": "yes (yfinance ES=F)", "next": "no", "expiry": "no", "note": "continuous front only; usable for trend baselines, NOT carry"},
    "NQ (e-mini Nasdaq futures)": {"class": "equity_index", "label": "continuous_only",
        "front": "yes (yfinance NQ=F)", "next": "no", "expiry": "no", "note": "continuous front only"},
    "CL (WTI crude futures)": {"class": "commodity", "label": "continuous_only",
        "front": "yes (yfinance CL=F)", "next": "no", "expiry": "no", "note": "continuous front only; dated NYMEX contracts not served by yfinance"},
    "GC (gold futures)": {"class": "commodity", "label": "continuous_only",
        "front": "yes (yfinance GC=F)", "next": "no", "expiry": "no", "note": "continuous front only"},
    "SI/NG/ZC/ZS/ZW (commodity futures)": {"class": "commodity", "label": "continuous_only",
        "front": "partial (yfinance =F)", "next": "no", "expiry": "no", "note": "continuous front at best; no curve"},
    "Rates futures (ZN/ZB/ZF)": {"class": "rates", "label": "reject",
        "front": "weak/none", "next": "no", "expiry": "no", "note": "no clean free source for the curve in this environment"},
    "FX futures (6E/6J/6B)": {"class": "fx", "label": "reject",
        "front": "weak/none", "next": "no", "expiry": "no", "note": "no clean free source for the curve"},
    "Massive us_futures_* flat files": {"class": "all", "label": "reject",
        "front": "catalog-only", "next": "catalog-only", "expiry": "catalog-only",
        "note": "session/minute/trades/quotes per-contract data EXISTS in catalogue (2017-2024) but GetObject=403 (paid entitlement); not downloadable"},
    "VXX/VXZ (VIX term-structure ETNs)": {"class": "vol_proxy", "label": "proxy_not_native_futures",
        "front": "VXX (short-term VIX futures)", "next": "VXZ (mid-term)", "expiry": "embedded (not exposed)",
        "note": "ETN price embeds VIX-futures roll yield; clean free history; term-structure SLOPE proxy, NOT native front/next prices"},
    "USO/USL (WTI oil curve ETFs)": {"class": "commodity_proxy", "label": "proxy_not_native_futures",
        "front": "USO (front WTI)", "next": "USL (12-month ladder)", "expiry": "embedded (not exposed)",
        "note": "USO/USL relationship reflects WTI curve shape; clean free history; proxy, NOT native curve"},
    "UNG (natgas ETF)": {"class": "commodity_proxy", "label": "proxy_not_native_futures",
        "front": "UNG (front natgas)", "next": "none", "expiry": "embedded", "note": "single front proxy; contango bleed only"},
}

CARRY_FORMULA = {
    "native_intended": "carry_t = ln(F_front_t / F_next_t) * (365 / (expiry_next - expiry_front in days))",
    "sign_convention": "positive => backwardation (front > next) => positive roll yield to a long holder",
    "annualization": "365 / (calendar days between front and next expiry)",
    "contracts_used": "front and first deferred (next) contract; raw (unadjusted) contract prices",
    "missing_next_handling": "no carry signal on dates lacking a clean next-contract price",
    "near_expiry_handling": "enforce min_days_to_expiry on the front (e.g. 5 trading days); roll before front expiry; never compute carry from an expiring front inside the threshold",
    "leakage_rule": "signal at date t uses only contracts and expiries known at date t; carry computed from RAW prices BEFORE any back-adjustment",
    "proxy_intended": "term-structure SLOPE proxy = ln(P_mid / P_short) e.g. ln(VXZ/VXX) or ln(USL/USO); labeled proxy_not_native_futures; NOT a native roll yield",
}

COST_MODEL = {
    "native_futures": "NOT applicable in v1 (no native data). Reference: ES ~0.25-tick spread + ~$2-4/contract commission; CL ~1-tick; roll cost ~1 spread/roll.",
    "etf_proxies": "VXX/VXZ/USO ~1-2 bps spread, ~0 commission (liquid); USL/UNG slightly wider. 2x cost stress + 1-bar delay stress feasible.",
    "verdict": "native cost model cannot be exercised (no native data); proxy cost model is feasible -> proxy markets are research-only at best.",
}


def verify_live() -> dict:
    """Best-effort re-check of the two decisive facts. Network-tolerant."""
    result: dict = {}
    try:
        import os
        import boto3
        from botocore.config import Config
        for line in open(".env"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)
        c = boto3.client("s3", endpoint_url=os.environ["MASSIVE_S3_ENDPOINT_URL"],
                         aws_access_key_id=os.environ["MASSIVE_S3_ACCESS_KEY_ID"],
                         aws_secret_access_key=os.environ["MASSIVE_S3_SECRET_ACCESS_KEY"],
                         config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                       connect_timeout=8, read_timeout=15, retries={"max_attempts": 1}))
        r = c.list_objects_v2(Bucket="flatfiles", Prefix="us_futures_cme/session_aggs_v1/2024/01/", MaxKeys=1)
        key = r.get("Contents", [{}])[0].get("Key")
        try:
            c.get_object(Bucket="flatfiles", Key=key, Range="bytes=0-10")
            result["massive_futures_get"] = "200_OK_UNEXPECTED"
        except Exception as exc:
            result["massive_futures_get"] = getattr(exc, "response", {}).get("Error", {}).get("Code", "error")
    except Exception as exc:
        result["massive_futures_get"] = f"unverified ({type(exc).__name__})"
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import yfinance as yf
        rows = len(yf.Ticker("VXZ").history(period="5d"))
        result["yfinance_proxy_VXZ_rows"] = rows
    except Exception as exc:
        result["yfinance_proxy_VXZ_rows"] = f"unverified ({type(exc).__name__})"
    return result


def main() -> int:
    live = verify_live() if "--live" in sys.argv else {"note": "static (use --live to re-verify)"}
    curve_clean = [m for m, d in MARKETS.items() if d["label"] == "curve_clean"]
    proxies = [m for m, d in MARKETS.items() if d["label"] == "proxy_not_native_futures"]
    continuous = [m for m, d in MARKETS.items() if d["label"] == "continuous_only"]
    verdict = "PASS" if curve_clean else ("PARTIAL_PASS" if proxies else "REJECT_ON_DATA")

    manifest = {
        "name": "futures_carry_term_structure_v1",
        "built_utc": datetime.now(UTC).isoformat(),
        "intake": "docs/research/intake/2026-05-30-futures-carry-term-structure-v1.md",
        "binding_question": "Can we compute futures carry from clean front+next contract prices+expiries, "
                            "without relying on a single back-adjusted continuous series?",
        "verdict": verdict,
        "markets": MARKETS,
        "carry_formula": CARRY_FORMULA,
        "cost_model": COST_MODEL,
        "live_verification": live,
        "labels_legend": ["curve_clean", "continuous_only", "proxy_not_native_futures",
                          "metadata_incomplete", "research_only", "reject"],
    }
    _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST.write_text(json.dumps(manifest, indent=2))

    lines = [
        "# Futures Carry / Term-Structure v1 — Data Audit",
        "",
        f"**Built:** {manifest['built_utc']}  **Intake:** `{manifest['intake']}`",
        "**Binding question:** Can we compute futures carry / roll yield from clean, timestamp-safe **front AND",
        "deferred** contract data, without relying on a single back-adjusted continuous series?",
        f"**Live re-verification:** `{live}`",
        "",
        "## 1-3. Data availability + curve classification (liquid markets first)",
        "",
        "| market | class | label | front | next | expiry | note |",
        "|---|---|---|---|---|---|---|",
        *[f"| {m} | {d['class']} | **{d['label']}** | {d['front']} | {d['next']} | {d['expiry']} | {d['note']} |"
          for m, d in MARKETS.items()],
        "",
        "**Curve requirement:** carry needs front + next contract prices + expiries. A single back-adjusted",
        "continuous series is NOT enough (usable for trend baselines only).",
        "",
        "## 4. Roll & back-adjustment audit",
        "- No native front/next contract data is downloadable here, so a leakage-safe roll cannot be constructed.",
        "- yfinance `=F` series are continuous/stitched front-month — exactly the back-adjusted single series the",
        "  intake forbids as a carry signal. Their roll/back-adjustment convention is undocumented (cannot verify",
        "  whether future roll info leaks).",
        "- Massive per-contract flat files would carry raw prices + a contract ticker encoding expiry (curve_clean",
        "  if downloadable), but `GetObject` = 403 (paid). Not usable.",
        "- Conclusion: the §4 leakage checks cannot be satisfied for any native market in this environment.",
        "",
        "## 5. Carry / roll-yield formula (pre-registered)",
        *[f"- **{k}**: {v}" for k, v in CARRY_FORMULA.items()],
        "",
        "## 6. Cost model",
        *[f"- **{k}**: {v}" for k, v in COST_MODEL.items()],
        "",
        "## 7. ETF proxy fallback",
        f"- Proxies available + downloadable (clean free history): {', '.join(proxies)}.",
        "- These EMBED roll yield (VIX-futures roll for VXX/VXZ; WTI curve for USO/USL) but do NOT expose",
        "  front/next contract prices + expiries. A research-only term-structure-SLOPE test is possible",
        "  (labeled `proxy_not_native_futures`); it must NOT claim native futures carry.",
        "",
        "## 8. Data-quality labels (summary)",
        f"- `curve_clean`: {curve_clean or 'NONE'}",
        f"- `continuous_only`: {len(continuous)} markets (yfinance =F) — trend baselines only, NOT carry",
        f"- `proxy_not_native_futures`: {len(proxies)} (VIX term ETNs, oil curve ETFs)",
        "- `reject`: rates/FX curve, Massive flat files (403)",
        "",
        "## 9. Audit decision",
        "",
        f"### VERDICT: **{verdict}**",
        "",
        "- **No `curve_clean` native futures market exists** in this environment: Massive per-contract data is",
        "  403-paywalled, and yfinance gives only continuous front-month (no next contract, no expiries).",
        "- **Proxy-only research test IS possible** (VIX term structure via VXX/VXZ; oil curve via USO/USL),",
        "  which is exactly the `PARTIAL_PASS` condition: only proxy/limited markets exist, research-only,",
        "  **no promotion language**.",
        "",
        "### Decision-rule consequence",
        "- Per the intake §10 and the operator decision rule: **PARTIAL_PASS → ASK before implementing a",
        "  proxy-only research run.** No strategy code is written. Native futures-carry v1 (promotion-grade) is",
        "  rejected on data grounds; only a clearly-labeled `proxy_not_native_futures` research probe remains",
        "  on the table, pending operator approval.",
        "- If the operator declines the proxy run, futures carry closes on data grounds and we return to the",
        "  program-review decision tree (CPI/NFP only if a timestamp-clean calendar source is supplied).",
        "",
        "## Constraints honored",
        "- No strategy backtest run. No single back-adjusted series used as a carry signal. No post-hoc roll",
        "  convention selection. No universe expansion. No options-IV / FinBERT. No CPI/NFP.",
    ]
    _AUDIT.parent.mkdir(parents=True, exist_ok=True)
    _AUDIT.write_text("\n".join(lines) + "\n")
    print(f"VERDICT: {verdict}  | curve_clean={len(curve_clean)} proxies={len(proxies)} continuous={len(continuous)}")
    print(f"live: {live}")
    print(f"Wrote {_MANIFEST}\nWrote {_AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
