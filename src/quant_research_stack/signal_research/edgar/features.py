"""Build the filing-date-aligned classical feature frame for EDGAR 10-K v1.

Leakage discipline (intake §5, §14):
- signal timestamp = SEC filing date; cross-section keyed to filing year.
- year-over-year features use ONLY the same CIK's strictly-earlier filing.
- forward-return columns are converted to LABELS (net = gross ratio - 1); they are
  NEVER emitted as features. A guard asserts no label leaks into the feature set.
"""

from __future__ import annotations

import glob
import math

import polars as pl

from quant_research_stack.signal_research.edgar.text_features import (
    cosine_similarity,
    jaccard_new_deleted,
    lm_tone_ratios,
    section_features,
)

_EDGAR_GLOB = "data/raw/huggingface/jlohding__sp500-edgar-10k/data/*.parquet"
# dataset horizon (gross-ratio col) -> pre-registered label name (nearest available)
_LABEL_MAP = {"20_day_return": "fwd_ret_21", "60_day_return": "fwd_ret_63",
              "150_day_return": "fwd_ret_126", "252_day_return": "fwd_ret_252"}
_PRIMARY_LABEL = "fwd_ret_63"
_LABELS = list(_LABEL_MAP.values())
# columns that must never become features
_FORBIDDEN_FEATURE_SUBSTR = ("fwd_ret", "_day_return", "label", "mkt_cap_raw")


def load_edgar() -> pl.DataFrame:
    df = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(_EDGAR_GLOB))], how="diagonal_relaxed")
    return df.with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("filing_date")).sort(
        ["cik", "filing_date"]
    )


def build_feature_frame() -> pl.DataFrame:
    edgar = load_edgar()
    rows = edgar.to_dicts()
    # prior-filing lookup per cik (strictly earlier) for YoY features
    prev_by_cik: dict[object, dict] = {}
    out: list[dict] = []
    for r in rows:
        cik = r["cik"]
        rf, mda, biz = r.get("item_1A") or "", r.get("item_7") or "", r.get("item_1") or ""
        feats: dict[str, object] = {
            "cik": cik,
            "company": r.get("company"),
            "sic": str(r.get("sic")),
            "sector2": str(r.get("sic"))[:2] if r.get("sic") is not None else "NA",
            "filing_date": r["filing_date"],
            "filing_year": int(r["filing_date"][:4]),
            "size_log_mktcap": math.log(float(r["mkt_cap"])) if r.get("mkt_cap") else 0.0,
            "event_ret": float(r["ret"]) if r.get("ret") is not None else 0.0,
        }
        feats.update(section_features(rf, prefix="rf"))
        feats.update(section_features(mda, prefix="mda"))
        feats.update(section_features(biz, prefix="biz"))
        feats["full_char_len"] = float(sum(len(x) for x in (rf, mda, biz)))

        prev = prev_by_cik.get(cik)
        if prev is not None and prev["filing_date"] < r["filing_date"]:
            prf, pmda = prev.get("item_1A") or "", prev.get("item_7") or ""
            rf_cos = cosine_similarity(rf, prf)
            mda_cos = cosine_similarity(mda, pmda)
            rf_jac = jaccard_new_deleted(rf, prf)
            feats["rf_yoy_cosine"] = rf_cos
            feats["rf_yoy_change"] = 1.0 - rf_cos
            feats["mda_yoy_cosine"] = mda_cos
            feats["mda_yoy_change"] = 1.0 - mda_cos
            feats["rf_new_word_frac"] = rf_jac["new_word_frac"]
            feats["rf_deleted_word_frac"] = rf_jac["deleted_word_frac"]
            feats["mda_tone_change"] = lm_tone_ratios(mda)["net_tone"] - lm_tone_ratios(pmda)["net_tone"]
            prev_len = float(len(prf) + len(pmda)) or 1.0
            feats["abnormal_len_change"] = (float(len(rf) + len(mda)) - prev_len) / prev_len
            feats["has_prior_filing"] = 1.0
        else:
            for k in ("rf_yoy_cosine", "rf_yoy_change", "mda_yoy_cosine", "mda_yoy_change",
                      "rf_new_word_frac", "rf_deleted_word_frac", "mda_tone_change", "abnormal_len_change"):
                feats[k] = None
            feats["has_prior_filing"] = 0.0

        # labels (gross ratio -> net return); LABELS ONLY
        for col, name in _LABEL_MAP.items():
            v = r.get(col)
            feats[name] = (float(v) - 1.0) if v is not None else None
        prev_by_cik[cik] = r
        out.append(feats)

    frame = pl.DataFrame(out)
    _assert_no_label_leak(frame)
    return frame


def feature_columns(frame: pl.DataFrame) -> list[str]:
    meta = {"cik", "company", "sic", "sector2", "filing_date", "filing_year", "has_prior_filing"}
    cols = []
    for name, dtype in frame.schema.items():
        if name in meta or name in _LABELS:
            continue
        if any(s in name for s in _FORBIDDEN_FEATURE_SUBSTR):
            continue
        if dtype.is_numeric():
            cols.append(name)
    return cols


def _assert_no_label_leak(frame: pl.DataFrame) -> None:
    # 1) any return/label-named column in the RAW frame must be an expected label.
    suspicious = [c for c in frame.columns
                  if any(s in c for s in _FORBIDDEN_FEATURE_SUBSTR) and c not in _LABELS]
    if suspicious:
        raise AssertionError(f"unexpected return/label-named column(s) present: {suspicious}")
    # 2) the model's feature set must contain no label.
    overlap = set(feature_columns(frame)) & set(_LABELS)
    if overlap:
        raise AssertionError(f"labels present in feature set: {overlap}")
