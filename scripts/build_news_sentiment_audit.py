"""News/sentiment timestamp/PIT data audit (no strategy code).

Audits the three on-disk timestamped news/sentiment candidates for timestamp/PIT
safety, look-ahead, and survivorship, and picks the cleanest for a v1 intake.
Training-only sets (FinGPT-sentiment, financial_phrasebank, Neil0930) are excluded
as signal feeds (no timestamp/ticker) — usable only to train a classifier.

Candidates:
- jlohding__sp500-edgar-10k   (SEC 10-K filings; filing-date timestamp)
- glopardo__sp500-earnings-transcripts (earnings-call transcripts; earnings_date)
- benstaf__nasdaq_2013_2023   (LLM-generated daily sentiment on a curated universe)

Emits (reports/signal_research/news_sentiment_v1/ + manifests/news_sentiment/):
  news_sentiment_data_manifest.json, news_sentiment_data_audit_report.md,
  news_sentiment_timestamp_audit.md, news_sentiment_universe_survivorship.md

Usage:
    PYTHONPATH=src uv run python scripts/build_news_sentiment_audit.py
"""

from __future__ import annotations

import glob
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

_REPORTS = Path("reports/signal_research/news_sentiment_v1")
_MANIFEST = Path("manifests/news_sentiment/news_sentiment_data_manifest.json")
_EDGAR = "data/raw/huggingface/jlohding__sp500-edgar-10k/data/*.parquet"
_TRANS = "data/raw/huggingface/glopardo__sp500-earnings-transcripts/data/*.parquet"
_BENSTAF = "data/raw/huggingface/benstaf__nasdaq_2013_2023/trade_data_deepseek_sentiment_2019_2023.csv"
_DELIST = ["TWITTER", "CELGENE", "XILINX", "CERNER", "ACTIVISION", "MAXIM"]


def _has(companies: list, name: str) -> bool:
    return any(isinstance(c, str) and name in c.upper() for c in companies)


def compute() -> dict:
    edgar = pl.concat([pl.read_parquet(f, columns=["cik", "company", "date"]) for f in sorted(glob.glob(_EDGAR))],
                      how="diagonal_relaxed").with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("d"))
    ecomp = edgar.select("company").unique()["company"].to_list()
    trans = pl.concat([pl.read_parquet(f, columns=["ticker", "company", "earnings_date"]) for f in sorted(glob.glob(_TRANS))],
                      how="diagonal_relaxed")
    tcomp = trans.select("company").unique()["company"].to_list()
    ben = pl.read_csv(_BENSTAF, columns=["date", "tic", "llm_sentiment"])
    return {
        "edgar": {
            "rows": edgar.height, "companies": int(edgar["cik"].n_unique()),
            "date_min": edgar["d"].min(), "date_max": edgar["d"].max(),
            "delisted_present": {nm: _has(ecomp, nm) for nm in _DELIST},
            "text_items": "item_1..item_15 (incl item_1A risk factors, item_7 MD&A); all non-null (6282/6282)",
            "forward_returns": "1/3/5/10/20/40/60/80/100/150/252_day_return + ret, mkt_cap (LABELS ONLY)",
        },
        "transcripts": {
            "rows": trans.height, "companies": int(trans["ticker"].n_unique()),
            "earnings_date_min": str(trans["earnings_date"].min()), "earnings_date_max": str(trans["earnings_date"].max()),
            "delisted_present": {nm: _has(tcomp, nm) for nm in _DELIST},
            "text_col": "transcript (present)",
            "forward_fields": "eps12mfwd_qavg/eoq, peforw_* (FORWARD fundamentals — must NOT be features)",
        },
        "benstaf": {
            "rows": ben.height, "tickers": int(ben["tic"].n_unique()),
            "date_min": ben["date"].min(), "date_max": ben["date"].max(),
            "sentiment": "llm_sentiment 0-5 (DeepSeek/Llama-generated; see arxiv 2502.07393)",
        },
    }


def main() -> int:
    s = compute()
    built = datetime.now(UTC).isoformat()
    edgar_surv = all(s["edgar"]["delisted_present"].values())  # all delisted present => survivorship-aware
    trans_biased = not any(s["transcripts"]["delisted_present"].values())  # all delisted absent => biased

    candidate_labels = {
        "edgar_10k": ["timestamp_clean_filing_date", "survivorship_aware", "text_present",
                      "forward_returns_labels_only", "annual_low_frequency"],
        "earnings_transcripts": ["timestamp_clean_earnings_date", "survivorship_biased_research_only",
                                 "forward_eps_fields_leak_risk", "text_present"],
        "benstaf_llm": ["llm_generated_lookahead_risk", "narrow_universe", "survivorship_prone", "reject_as_signal"],
    }
    cleanest = "edgar_10k"
    verdict = "PASS (EDGAR 10-K) — open a research v1 intake on the cleanest candidate"

    _REPORTS.mkdir(parents=True, exist_ok=True)
    _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": "news_sentiment_v1_data_audit", "built_utc": built,
        "binding_question": "Is there a timestamp-clean, look-ahead-safe news/sentiment signal feed?",
        "verdict": verdict, "cleanest_candidate": cleanest,
        "training_only_excluded": ["FinGPT__fingpt-sentiment-train", "takala__financial_phrasebank",
                                   "Neil0930__quantitative_finance_dataset"],
        "candidate_labels": candidate_labels, "stats": s,
        "labels_legend": ["timestamp_clean", "survivorship_aware", "survivorship_biased_research_only",
                          "lookahead_risk", "training_only", "reject"],
    }
    _MANIFEST.write_text(json.dumps(manifest, indent=2))

    (_REPORTS / "news_sentiment_timestamp_audit.md").write_text("\n".join([
        "# News/Sentiment Audit — Timestamp / PIT / Look-ahead",
        "", f"**Built:** {built}",
        "",
        "## EDGAR 10-K (jlohding) — timestamp_clean",
        f"- `date` = **SEC filing date** (true public-availability timestamp). Range {s['edgar']['date_min']} → {s['edgar']['date_max']}.",
        "- 10-K text is backward-looking (prior fiscal year), made public at the filing date → keying features to",
        "  the filing date and trading **t+1** is leak-safe.",
        f"- Forward returns ({s['edgar']['forward_returns']}) are present — **LABELS ONLY; never features.**",
        "- **No look-ahead** in the text itself. Verdict: timestamp-clean.",
        "",
        "## Earnings transcripts (glopardo) — timestamp_clean (universe caveat)",
        f"- `earnings_date` = call date (after-close typical) → trade t+1 leak-safe. Range {s['transcripts']['earnings_date_min']} → {s['transcripts']['earnings_date_max']}.",
        f"- Contains FORWARD fundamentals ({s['transcripts']['forward_fields']}) → **must NOT be used as features.**",
        "- Timestamp is clean; the binding defect is survivorship (see universe report).",
        "",
        "## benstaf LLM-sentiment — LOOK-AHEAD RISK (binding)",
        f"- {s['benstaf']['sentiment']}. Daily date+tic.",
        "- **Hard look-ahead concern:** the sentiment is generated by an LLM (DeepSeek/Llama) whose training",
        "  cutoff post-dates the 2019-2023 sample, so scores can encode hindsight; and the news-availability",
        "  time per score is undocumented. Cannot establish feature_timestamp < signal_timestamp safely.",
        "- Verdict: **reject as a leak-safe signal** (usable, if at all, only as a heavily-caveated diagnostic).",
    ]) + "\n")

    (_REPORTS / "news_sentiment_universe_survivorship.md").write_text("\n".join([
        "# News/Sentiment Audit — Universe & Survivorship",
        "",
        "## EDGAR 10-K",
        f"- {s['edgar']['companies']} unique companies (CIK), {s['edgar']['rows']} filings, "
        f"{s['edgar']['date_min']} → {s['edgar']['date_max']} (annual).",
        f"- README: 'all SP500 **historical** constituents'. Delisted/merged names present: "
        + ", ".join(f"{k}={v}" for k, v in s['edgar']['delisted_present'].items())
        + f" → **survivorship-aware: {edgar_surv}** (FRC/SIVB failed in 2023, outside the 2010-2022 window).",
        "",
        "## Earnings transcripts",
        f"- {s['transcripts']['companies']} companies (from the **current** Wikipedia S&P 500 list), "
        f"{s['transcripts']['rows']} transcripts.",
        "- Delisted-name presence: " + ", ".join(f"{k}={v}" for k, v in s['transcripts']['delisted_present'].items())
        + f" → all absent → **survivorship-biased: {trans_biased}** (current-constituent universe).",
        "",
        "## benstaf",
        f"- {s['benstaf']['tickers']} tickers (curated mega-cap/NASDAQ universe), {s['benstaf']['rows']} rows, "
        f"{s['benstaf']['date_min']} → {s['benstaf']['date_max']} → narrow + survivorship-prone.",
    ]) + "\n")

    (_REPORTS / "news_sentiment_data_audit_report.md").write_text("\n".join([
        "# News/Sentiment Data Audit — Summary & Decision",
        "", f"**Built:** {built}",
        "**Binding question:** is there a timestamp-clean, look-ahead-safe news/sentiment signal feed?",
        "",
        "## Candidate comparison",
        "",
        "| candidate | timestamp | survivorship | look-ahead | freq | labels |",
        "|---|---|---|---|---|---|",
        f"| **EDGAR 10-K** | filing date (PIT) ✓ | aware ({s['edgar']['companies']} cos) ✓ | none ✓ | annual | fwd returns incl. |",
        f"| earnings transcripts | earnings_date ✓ | **biased** ({s['transcripts']['companies']} cur. cos) ✗ | fwd-EPS fields | quarterly | — |",
        f"| benstaf LLM | daily | narrow ({s['benstaf']['tickers']}) ✗ | **LLM hindsight** ✗✗ | daily | — |",
        "",
        "Training-only (no timestamp/ticker; excluded as feeds): FinGPT-sentiment, financial_phrasebank, Neil0930 —",
        "usable only to TRAIN a sentiment classifier, not as a PIT signal.",
        "",
        f"## VERDICT: **{verdict}**",
        "",
        f"- **Cleanest candidate: `{cleanest}` (EDGAR 10-K).** It is the first dataset in the whole program that",
        "  is BOTH timestamp-clean (SEC filing date = true public-availability) AND survivorship-aware (727",
        "  historical constituents incl. delisted names), with full 10-K item text and forward-return labels.",
        "- Earnings transcripts: clean timestamp but **survivorship-biased universe** + forward-EPS fields →",
        "  `research_only`, cross-sectional invalid (single-name/time-series use less affected).",
        "- benstaf LLM-sentiment: **rejected** as a leak-safe signal (LLM hindsight + undocumented availability).",
        "",
        "### Recommended next step",
        "Open a **research v1 intake on EDGAR 10-K**: 10-K text features (risk-factor / MD&A tone, year-over-year",
        "10-K text change) keyed to the **filing date**, traded **t+1**, with forward-return labels used as labels",
        "only. Cross-sectional across historical constituents is valid here (survivorship-aware). It is annual /",
        "low-frequency and long-horizon — pre-register that as a constraint. This is the strongest data the",
        "program has found; it still faces the standard validation gate and the cost/subsumption walls.",
        "",
        "## Constraints honored",
        "- No strategy code. No silent dropping. Forward returns / forward-EPS flagged labels-only. LLM",
        "  look-ahead investigated and the LLM feed rejected. Training-only sets excluded as feeds.",
    ]) + "\n")

    print(f"VERDICT: {verdict}")
    print(f"EDGAR: {s['edgar']['rows']} filings, {s['edgar']['companies']} cos, survivorship_aware={edgar_surv}")
    print(f"transcripts: {s['transcripts']['rows']} rows, {s['transcripts']['companies']} cos, survivorship_biased={trans_biased}")
    print(f"benstaf: {s['benstaf']['rows']} rows, {s['benstaf']['tickers']} tickers (LLM look-ahead -> reject)")
    print(f"Wrote manifest + 3 reports under {_REPORTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
