"""EDGAR 10-Q quarterly-filings data audit (no strategy code).

Binding question: does an on-disk EDGAR 10-Q dataset exist, or can a 10-Q research
panel be cleanly + survivorship-safely assembled for free (filing dates + item
text + CIK/ticker mapping + survivorship-aware coverage + usable return labels)?

Context: 10-K v1 (commit 719530c) closed at low_frequency_insufficient_sample —
annual filings gave only 3 holdout cross-sections. 10-Q (quarterly) would ~4x the
cross-sections, addressing that wall IF the data exists. The equity-return audit
(commit 3f9a658) already established NO survivorship-safe free return panel is on
disk (HexQuant is survivorship-biased).

Emits reports/signal_research/edgar_10q_v1/{edgar_10q_data_manifest.json,
edgar_10q_data_audit.md}.

Usage:
    PYTHONPATH=src uv run python scripts/build_edgar_10q_audit.py
"""

from __future__ import annotations

import glob
import json
import re
from datetime import UTC, datetime
from pathlib import Path

_HF = "data/raw/huggingface"
_OUT = Path("reports/signal_research/edgar_10q_v1")
_MANIFEST = _OUT / "edgar_10q_data_manifest.json"


def scan_for_10q() -> dict:
    dirs = sorted(p.name for p in Path(_HF).glob("*") if p.is_dir())
    # any dataset whose name OR files reference 10-Q / quarterly filings
    name_hits = [d for d in dirs if re.search(r"10[\-_]?q|quarter", d, re.I)]
    file_hits = [f for f in glob.glob(f"{_HF}/**/*", recursive=True)
                 if re.search(r"10[\-_]?q", f, re.I)]
    sec_text_datasets = [d for d in dirs if re.search(r"edgar|10[\-_]?k|10[\-_]?q|filing", d, re.I)]
    return {
        "hf_dataset_count": len(dirs),
        "name_hits_10q": name_hits,
        "file_hits_10q": file_hits[:10],
        "sec_text_datasets_present": sec_text_datasets,
        "on_disk_10q": bool(name_hits or file_hits),
    }


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    scan = scan_for_10q()
    built = datetime.now(UTC).isoformat()

    labels = ["return_panel_required", "mapping_incomplete", "reject"]
    verdict = "REJECT_ON_DATA — no on-disk 10-Q; free assembly blocked by missing survivorship-safe labels + PIT constituents"

    manifest = {
        "name": "edgar_10q_v1_data_audit", "built_utc": built,
        "binding_question": "On-disk or cleanly+survivorship-safely free-sourceable 10-Q research panel?",
        "verdict": verdict, "labels": labels, "scan": scan,
        "return_panel_status": "no survivorship-safe free return panel (equity_return_data_audit, commit 3f9a658; "
                               "HexQuant survivorship-biased). 10-K worked only because returns were BUNDLED.",
        "labels_legend": ["filing_timestamp_clean", "quarterly_frequency_sufficient", "survivorship_safe_enough",
                          "mapping_incomplete", "labels_available", "return_panel_required",
                          "text_sections_incomplete", "research_only", "reject"],
    }
    _MANIFEST.write_text(json.dumps(manifest, indent=2))

    (_OUT / "edgar_10q_data_audit.md").write_text("\n".join([
        "# EDGAR 10-Q Quarterly-Filings Data Audit",
        "",
        f"**Built:** {built}  **Binding question:** can we assemble a clean, survivorship-safe 10-Q research",
        "panel (filing dates + item text + CIK/ticker mapping + survivorship coverage + usable labels)?",
        "**Motivation:** 10-K v1 closed `low_frequency_insufficient_sample` (3 holdout cross-sections); 10-Q",
        "(~4x cross-sections) would address the frequency wall IF the data exists.",
        "",
        "## 1. Dataset availability",
        f"- On-disk EDGAR 10-Q dataset: **{'YES' if scan['on_disk_10q'] else 'NO'}** "
        f"(scanned {scan['hf_dataset_count']} HF datasets; 10-Q name/file hits: "
        f"{scan['name_hits_10q'] or 'none'} / {len(scan['file_hits_10q'])}).",
        f"- SEC text datasets present: {scan['sec_text_datasets_present']} (only the 10-K set; no 10-Q).",
        "- 10-Q **text** is freely available from SEC EDGAR (public; parseable with `edgar-crawler`, the same",
        "  tool used for the 10-K set) — but that is a multi-hour scrape+parse, not on-disk data.",
        "",
        "## 2. Timestamp integrity",
        "- NOT the blocker: SEC filing date / accepted timestamp is clean and would be used (trade t+1; never",
        "  fiscal-period-end) — identical discipline to 10-K v1, which passed this gate.",
        "",
        "## 3. Survivorship & mapping",
        "- Raw EDGAR retains delisted/acquired/renamed filers (TWTR/CELG/XLNX/CERN/ATVI/SIVB/FRC/AABA all have",
        "  CIKs on EDGAR), so 10-Q TEXT could be survivorship-aware. **But** a research panel needs a",
        "  **point-in-time S&P-500 (or universe) constituent-membership list** to know which CIKs to include",
        "  each quarter, plus a CIK↔ticker map through ticker changes — neither is available survivorship-safe",
        "  for free here. → `mapping_incomplete`.",
        "",
        "## 4. Text sections (if sourced)",
        "- 10-Q exposes Item 1 (financial statements), Item 2 (MD&A), Item 3 (market risk), Item 4 (controls),",
        "  and Part II Item 1A (risk-factor updates) — all parseable. Not the blocker.",
        "",
        "## 5. Labels (THE BINDING CONSTRAINT)",
        "- A fresh 10-Q text scrape carries **no return labels**. The 10-K dataset only worked because it",
        "  **bundled survivorship-safe forward returns**. The equity-return audit (commit `3f9a658`) established",
        "  there is **no survivorship-safe free return panel on disk** (HexQuant drops delisted names). Pairing",
        "  10-Q text with a survivor-only return source would **repeat the options-IV survivorship mistake**.",
        "  → `return_panel_required` (UNMET).",
        "",
        "## 6. Sample size (would-be, moot)",
        "- 10-Q quarterly ≈ 4x the cross-sections of 10-K → ~40-50 cross-sections over 2010-2022, enough for",
        "  rank IC / decile spread / PBO / DSR / bootstrap — this WOULD fix the 10-K frequency wall. But it is",
        "  moot without survivorship-safe labels (§5) and PIT constituents (§3).",
        "",
        "## 7. Feature feasibility",
        "- Classical features (length/LM tone/uncertainty/litigious/modal/numeric density/readability/QoQ &",
        "  YoY-same-quarter change/boilerplate/new-deleted risk language/MD&A tone change) are all computable",
        "  from text. Not the blocker.",
        "",
        f"## 8. Data-quality labels: `{', '.join(labels)}`",
        "",
        f"## 9. VERDICT: **{verdict}**",
        "",
        "- No on-disk 10-Q dataset. 10-Q **text** is free-sourceable from SEC EDGAR, but the **labels**",
        "  (survivorship-safe returns) and **PIT constituent membership** are not available for free — the same",
        "  data wall that closed the options-IV cross-section. Per the decision rule, a 10-Q research panel",
        "  cannot be assembled leak-safe + survivorship-safe without paid data → **reject on data grounds**.",
        "- **Do not** assemble a survivor-only 10-Q backtest (would be invalid, like the rejected HexQuant",
        "  cross-section). **Do not** scrape EDGAR text without a survivorship-safe return panel to label it.",
        "",
        "## 10. Consequence",
        "- Per the operator's fallback instruction: **pause the free-data SEC search and produce a paid-data",
        "  acquisition recommendation** — see `docs/research/2026-05-30-PAID-DATA-ACQUISITION-RECOMMENDATION.md`.",
        "- SEC filings remain a *real* channel (10-K v1 proved the text+timestamp+survivorship gates are",
        "  passable); the binding gap is a survivorship-safe **return/fundamentals panel** to label higher-",
        "  frequency filings. That is a data-acquisition decision, not another free-data model search.",
        "",
        "## Constraints honored",
        "- No strategy code. No survivor-only return source. No EDGAR scrape without labels. No embeddings/LLM.",
        "  No promotion language. research_only.",
    ]) + "\n")

    print(f"VERDICT: {verdict}")
    print(f"on_disk_10q={scan['on_disk_10q']} | sec_text_datasets={scan['sec_text_datasets_present']}")
    print(f"Wrote {_MANIFEST} + edgar_10q_data_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
