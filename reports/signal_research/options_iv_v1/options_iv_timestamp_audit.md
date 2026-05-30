# Options-IV Data Audit — §1 Timestamp Integrity

**Built:** 2026-05-30T08:17:17.435367+00:00  **Rows:** 3,161,661  **Range:** 2019-10-14 → 2023-07-28

- Only a daily `date` column exists — **no intraday/observability timestamp**. Values are end-of-day
  (VIX is the EOD index level; `hv_*` are trailing historical vols; IV buckets are market-implied EOD).
- **Same-day signal use is NOT safe** (we cannot prove the IV was observable before a same-day close
  decision). **EOD-features-for-next-day (t → t+1) IS safe** by shifting features one trading day.
- `feature_timestamp < signal_timestamp` can be enforced ONLY under the next-day convention (feature at
  EOD t, decision/execution at t+1). Document and enforce a 1-day shift in any downstream research.
- No revised/recomputed field is documented; IV is market-implied (not derived from future realized
  vol per the dataset README), and `hv_*` are explicitly *historical* (trailing) — no future-RV leak
  evident. Caveat: the author's exact computation is undocumented beyond the README → **timestamp_uncertain**.

**§1 verdict: timestamp_uncertain — research_only, next-day features only.**
