# Massive.com Data-Feed Entitlement Audit

**Audit date:** 2026-05-29
**Auditor:** QuantLab research
**Method:** live probes against the credentials in `.env` (gitignored; secrets
never printed or committed). REST probes respected the 5-calls/min budget.
**Purpose:** establish exactly what the supplied Massive.com plan can access
before any ingestion or strategy work depends on it.

## 0. Binding question

> What market data can the credentials in `.env` actually retrieve, and is it
> sufficient to train a model?

## 1. Provider identity

Massive.com is a **Polygon.io-compatible** provider: identical S3 flat-file
prefix layout (`us_stocks_sip`, `us_options_opra`, `us_futures_*`, `us_indices`,
`global_crypto`, `global_forex`) and an identical REST schema
(`api.massive.com` mirrors the Polygon REST surface).

Credential note: `MASSIVE_S3_SECRET_ACCESS_KEY` is byte-identical to
`MASSIVE_REST_API_KEY`. This is **not** the cause of any failure — that value
still authenticates S3 *listing*.

## 2. S3 flat files (`files.massive.com`, bucket `flatfiles`)

| Operation | Result |
|---|---|
| `ListObjectsV2` (all prefixes) | ✅ works — full catalogue visible, trades+quotes back to **2003** |
| `GetObject` / `HeadObject` (every asset class, every date tested) | ❌ **403 Forbidden** |

**Two findings:**

1. **Addressing-style bug (fixed).** boto3 defaults to virtual-hosted-style
   addressing against a custom endpoint, which times out / 403s. The endpoint
   only accepts **path-style** (`Config(s3={"addressing_style": "path"})`).
   This — not the credentials — caused the blanket 403 in the original
   `data_audit_report.md` exploration. `scripts/explore_massive_flatfiles.py`
   is now patched.
2. **Downloads are not entitled.** With addressing fixed, listing works but
   every `GetObject` still 403s. Bulk historical flat-file download (the only
   channel that could supply training-grade tick/quote history) requires a
   **paid plan tier** this key does not hold.

## 3. REST API (`api.massive.com`, Bearer auth, 5 calls/min)

| Endpoint | Result |
|---|---|
| `/v1/marketstatus/now` | ✅ 200 OK |
| `/v2/aggs/ticker/{T}/prev` (previous-day EOD bar) | ✅ 200 OK |
| `/v2/aggs/ticker/{T}/range/...` (historical aggregates) | ❌ 403 `NOT_AUTHORIZED` |
| `/v2/snapshot/...` (live snapshot) | ❌ 403 `NOT_AUTHORIZED` |
| `/v3/reference/tickers` | ❌ timeout / not authorized |

Live confirmation (2026-05-29): market open; `SPY` prev-close `754.6`,
`QQQ` prev-close `735.6` for trading day 2026-05-28.

## 4. Verdict

**`rest_eod_only` — not training-grade.** The plan authorizes live market
status and previous-day EOD bars per ticker at 5 calls/min. It cannot backfill
history (range = 403) and cannot stream/snapshot (snapshot = 403), and S3
downloads are 403. Building a cross-sectional research dataset from it would
mean accumulating one prior-day bar per ticker per day, forward-only — strictly
worse than the 2003–2026 daily history the program already holds.

## 5. What was built against this verdict

Scoped to the reachable surface, no overclaiming:

- `src/quant_research_stack/feeds/rate_limit.py` — sliding-window `RateLimiter`
  (5/min), injectable clock for deterministic tests.
- `src/quant_research_stack/feeds/massive_rest.py` — `MassiveREST` client
  (`market_status`, `previous_close`), Polygon-compatible parsers, parquet
  panel upsert, and a `NotAuthorizedError` that surfaces the paywall explicitly.
- `scripts/massive_ingest.py` — CLI for `status` and `prev-close` accumulation.
- 18 unit tests (network-free via `httpx.MockTransport`); `ruff` + `mypy` clean.

## 6. Unlocks (operator decision, not in scope here)

The microstructure / order-book v1 direction (program review §8.A) needs clean
tick/quote data. On the **current** plan that is unreachable for US equities.
Two paths: (a) upgrade the Massive plan to unlock flat-file downloads / REST
history, or (b) run microstructure v1 on the already-audited **Binance BTCUSDT
public archive** (verdict `trade_only_clean`, `data_audit_report.md`), which is
free and downloadable. No promotion, paper-trading, or live trading is
authorized by this audit.
