"""P0 — zero-cost data loaders + macro-feature timestamp audit (no strategy code).

Loads the 4 directly-traded instruments (SPY/QQQ disk; BTC/ETH yfinance) and the
allowed timestamp-safe macro series (market-priced ETFs/VIX + daily Treasury CMT),
caches them, and emits the P0 deliverables. Forbidden revised aggregates are NOT
fetched (rejected by policy). Best-effort network: unavailable series are recorded,
not fatal.

Emits manifests/zero_cost/zero_cost_data_manifest.json and, under
reports/signal_research/zero_cost_v1/: zero_cost_macro_timestamp_audit.md,
zero_cost_feature_availability_report.md, zero_cost_instrument_coverage_report.md.

Usage:
    PYTHONPATH=src uv run python scripts/build_zero_cost_data_audit.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from quant_research_stack.signal_research.zero_cost.data import (
    DERIVED_FEATURES,
    FORBIDDEN_SERIES,
    INSTRUMENTS,
    MACRO_REGISTRY,
    load_instrument,
    load_macro,
)

console = Console()
_OUT = Path("reports/signal_research/zero_cost_v1")
_MANIFEST = Path("manifests/zero_cost/zero_cost_data_manifest.json")


def _coverage(df) -> dict[str, Any]:
    if df is None or df.height == 0:
        return {"available": False, "rows": 0, "date_min": None, "date_max": None}
    return {"available": True, "rows": df.height,
            "date_min": str(df["date"].min()), "date_max": str(df["date"].max())}


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    built = datetime.now(UTC).isoformat()

    instruments: dict[str, dict[str, Any]] = {}
    for name, spec in INSTRUMENTS.items():
        try:
            cov = _coverage(load_instrument(name))
        except Exception as exc:
            cov = {"available": False, "error": f"{type(exc).__name__}: {str(exc)[:80]}"}
        instruments[name] = {**spec, **cov}
        console.print(f"  instrument {name}: {cov.get('available')} {cov.get('date_min')}..{cov.get('date_max')}")

    macro: dict[str, dict[str, Any]] = {}
    for s in MACRO_REGISTRY:
        try:
            cov = _coverage(load_macro(s))
        except Exception as exc:
            cov = {"available": False, "error": f"{type(exc).__name__}: {str(exc)[:80]}"}
        macro[s.name] = {"source": s.source, "ref": s.ref, "classification": s.classification,
                         "rationale": s.rationale, **cov}
        console.print(f"  macro {s.name} ({s.classification}): {cov.get('available')}")

    # common basket window (intersection of instrument coverage)
    starts = [v["date_min"] for v in instruments.values() if v.get("available")]
    ends = [v["date_max"] for v in instruments.values() if v.get("available")]
    basket_start = max(starts) if starts else None
    basket_end = min(ends) if ends else None

    instruments_ok = all(v.get("available") for v in instruments.values())
    macro_avail = [k for k, v in macro.items() if v.get("available")]
    # timestamp-safety: every registry series is allowed by construction; confirm none is revision_risk/reject
    unsafe = [k for k, v in macro.items() if v["classification"] in ("revision_risk", "reject")]
    verdict = "PASS_TIMESTAMP_SAFE" if (instruments_ok and macro_avail and not unsafe) else "PARTIAL"

    manifest = {
        "name": "zero_cost_data_manifest", "built_utc": built, "verdict": verdict,
        "instruments": instruments, "macro": macro,
        "forbidden_series": FORBIDDEN_SERIES, "derived_features": DERIVED_FEATURES,
        "basket_common_window": {"start": basket_start, "end": basket_end},
        "timestamp_rule": "all features observed at close t, used only at t+1",
    }
    _MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))

    # ---- macro timestamp audit ----
    (_OUT / "zero_cost_macro_timestamp_audit.md").write_text("\n".join([
        "# Zero-Cost Macro-Feature Timestamp Audit (P0)",
        f"\n**Built:** {built}  **Rule:** every feature observed at close t, used only at **t+1**.",
        f"**Verdict:** **{verdict}**",
        "\n## Allowed macro features (timestamp-safe)",
        "", "| feature | source | ref | classification | available | coverage | rationale |",
        "|---|---|---|---|:---:|---|---|",
        *[f"| {k} | {v['source']} | {v['ref']} | **{v['classification']}** | {v.get('available')} | "
          f"{v.get('date_min')}..{v.get('date_max')} | {v['rationale']} |" for k, v in macro.items()],
        "\n## Forbidden (revised aggregates — NOT fetched, NOT usable without PIT vintages)",
        *[f"- `{k}`: {why}" for k, why in FORBIDDEN_SERIES.items()],
        "\n## Classification legend",
        "- `market_price_clean`: daily market close, not revised → safe at t+1.",
        "- `daily_next_day_only`: published EOD t (e.g. Treasury CMT), not revised → safe at t+1.",
        "- `revision_risk` / `reject`: revised aggregates without PIT vintages → forbidden.",
        "\n## Decision",
        ("- **PASS** — all instruments covered and all available macro features are timestamp-safe; no forbidden"
         " series used. Proceed to P1 (zero_cost_riskalloc_v1)."
         if verdict == "PASS_TIMESTAMP_SAFE" else
         f"- **PARTIAL** — instruments_ok={instruments_ok}, macro_available={macro_avail}, unsafe={unsafe}."
         " Re-run after the unavailable series fetch, or drop them; do NOT proceed with unsafe features."),
    ]) + "\n")

    # ---- feature availability ----
    (_OUT / "zero_cost_feature_availability_report.md").write_text("\n".join([
        "# Zero-Cost Feature Availability (P0)",
        f"\n**Built:** {built}",
        "\n## Per-instrument OHLCV features (always available from instrument bars)",
        "- trailing realized vol (20/60d), downside vol, trend (12-1 momentum, SMA50/200 state),",
        "  drawdown-from-peak, return autocorrelation. Computed at close t, used t+1.",
        "\n## Macro features available now",
        *[f"- {k}: {'available' if v.get('available') else 'UNAVAILABLE'} ({v['classification']})"
          for k, v in macro.items()],
        "\n## Derived features (built in P1)",
        *[f"- `{k}`: {desc}" for k, desc in DERIVED_FEATURES.items()],
        "\n## Regime features",
        "- HMM risk-on/off (reuse `signal_research/strategies/hmm_single_index.py`), past-data-only;",
        "  vol-regime fallback (trailing vol vs rolling median).",
    ]) + "\n")

    # ---- instrument coverage ----
    (_OUT / "zero_cost_instrument_coverage_report.md").write_text("\n".join([
        "# Zero-Cost Instrument Coverage (P0)",
        f"\n**Built:** {built}",
        "", "| instrument | source | available | rows | date range |",
        "|---|---|:---:|---:|---|",
        *[f"| {k} | {v['source']} | {v.get('available')} | {v.get('rows', 0)} | "
          f"{v.get('date_min')}..{v.get('date_max')} |" for k, v in instruments.items()],
        f"\n**Basket common window (intersection):** `{basket_start}` .. `{basket_end}`",
        "- ETH-USD (~2017 start) is the binding constraint on the 4-instrument basket window;",
        "  SPY/QQQ alone cover 2010-2026. The basket validation runs on the common window.",
        "- Long-flat, weekly rebalance, equal-risk; decisions at close t, execution t+1.",
    ]) + "\n")

    console.print(f"[bold]{verdict}[/bold] | basket window {basket_start}..{basket_end}")
    console.print(f"Wrote manifest + 3 reports under {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
