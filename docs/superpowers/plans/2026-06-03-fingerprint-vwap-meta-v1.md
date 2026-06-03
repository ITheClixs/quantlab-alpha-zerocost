# Fingerprint-VWAP Meta-Labeling v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test, under this repo's gates, whether regime-**fingerprint**-conditioned **meta-labeling of a VWAP entry** is net-of-cost alpha — building it by composing existing modules.

**Architecture:** A new branch subpackage `signal_research/fingerprint_vwap/` computes a daily VWAP proxy, a VWAP primary entry, and multi-window fingerprint features; a minimal backward-compatible generalization of `signal_research/training/meta_label_walk_forward.py` lets that trainer accept a caller-supplied primary and extra features; the existing triple-barrier labeller, purged walk-forward, RF meta-classifier, and PBO/Deflated-Sharpe gates are reused unchanged. Everything is research_only and gated; transfer to Prevalence happens only on a full PASS.

**Tech Stack:** Python 3.11, Polars, NumPy, scikit-learn (RandomForest, already used), pytest. Run with `PYTHONPATH=src`. Spec: `docs/research/intake/2026-06-03-fingerprint-vwap-meta-v1.md`.

---

## File structure

| Path | Responsibility |
|---|---|
| `src/quant_research_stack/signal_research/fingerprint_vwap/__init__.py` | package marker + public exports |
| `src/quant_research_stack/signal_research/fingerprint_vwap/vwap.py` | `daily_vwap_proxy`, `vwap_primary_position` (pure) |
| `src/quant_research_stack/signal_research/fingerprint_vwap/fingerprint.py` | `build_fingerprint_features` (pure, multi-window, as-of) |
| `src/quant_research_stack/signal_research/fingerprint_vwap/eligibility.py` | `primary_signal_stats` → `check_eligibility` |
| `src/quant_research_stack/signal_research/fingerprint_vwap/pipeline.py` | `run_fingerprint_vwap_meta`, `gate_verdict`, `render_report` |
| `src/quant_research_stack/signal_research/training/meta_label_walk_forward.py` | **MODIFY**: optional `primary_position_col`, `extra_feature_columns` (backward-compatible) |
| `scripts/run_fingerprint_vwap_meta_backtest.py` | real-universe runner (reuses av_lee data fetch + cache) |
| `tests/signal_research/fingerprint_vwap/conftest.py` | deterministic synthetic OHLCV panel fixture |
| `tests/signal_research/fingerprint_vwap/test_*.py` | unit tests per module |
| `reports/signal_research/fingerprint_vwap_meta_v1/` | `data_audit.md`, artifacts, verdict note |

Column contract (panel passed between stages): `date` (pl.Date), `symbol` (str), `open,high,low,close,volume` (f64). Stages add columns; never mutate in place.

---

## Task 1: Test fixture — deterministic synthetic OHLCV panel

**Files:**
- Create: `tests/signal_research/fingerprint_vwap/__init__.py` (empty)
- Create: `tests/signal_research/fingerprint_vwap/conftest.py`

- [ ] **Step 1: Write the fixture**

```python
# tests/signal_research/fingerprint_vwap/conftest.py
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def panel() -> pl.DataFrame:
    """Two symbols, 400 trading days. AAA is a clean uptrend (linear log-price),
    BBB is noisy/flat — so fingerprint features have known signs in tests."""
    rng = np.random.default_rng(7)
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(400)]
    rows = []
    for sym, drift, noise in (("AAA", 0.0010, 0.005), ("BBB", 0.0, 0.02)):
        logp = np.cumsum(np.full(400, drift) + rng.normal(0, noise, 400)) + np.log(100.0)
        close = np.exp(logp)
        high = close * (1.0 + np.abs(rng.normal(0, 0.003, 400)))
        low = close * (1.0 - np.abs(rng.normal(0, 0.003, 400)))
        open_ = close * (1.0 + rng.normal(0, 0.002, 400))
        vol = rng.integers(1_000_000, 5_000_000, 400).astype(float)
        for i, d in enumerate(dates):
            rows.append((d, sym, float(open_[i]), float(high[i]), float(low[i]),
                         float(close[i]), float(vol[i])))
    return pl.DataFrame(
        rows, schema=["date", "symbol", "open", "high", "low", "close", "volume"],
        orient="row",
    ).with_columns(pl.col("date").cast(pl.Date))
```

- [ ] **Step 2: Commit**

```bash
git add tests/signal_research/fingerprint_vwap/__init__.py tests/signal_research/fingerprint_vwap/conftest.py
git commit -m "test(fingerprint-vwap): deterministic synthetic OHLCV fixture"
```

---

## Task 2: Daily VWAP proxy

**Files:**
- Create: `src/quant_research_stack/signal_research/fingerprint_vwap/__init__.py` (empty for now)
- Create: `src/quant_research_stack/signal_research/fingerprint_vwap/vwap.py`
- Test: `tests/signal_research/fingerprint_vwap/test_vwap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/signal_research/fingerprint_vwap/test_vwap.py
from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.fingerprint_vwap.vwap import daily_vwap_proxy


def test_vwap_proxy_is_typical_price_rolling_volume_weighted(panel: pl.DataFrame) -> None:
    out = daily_vwap_proxy(panel, window=5)
    assert "vwap" in out.columns
    assert out.height == panel.height
    # vwap sits within the [min(low), max(high)] envelope of its window for each row that is defined
    defined = out.drop_nulls("vwap")
    assert defined.height > 0
    assert (defined["vwap"] >= defined["low"].min()).all()
    # no look-ahead: vwap at row t uses only rows <= t (monotone date order preserved per symbol)
    assert out.sort(["symbol", "date"]).select(["symbol", "date"]).equals(
        panel.sort(["symbol", "date"]).select(["symbol", "date"])
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_vwap.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'daily_vwap_proxy'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_research_stack/signal_research/fingerprint_vwap/vwap.py
"""VWAP proxy + VWAP-entry primary (spec §6, steps 2-3). Daily bars only:
'VWAP' is a rolling volume-weighted typical price, an as-of proxy for intraday VWAP."""

from __future__ import annotations

import polars as pl


def daily_vwap_proxy(panel: pl.DataFrame, *, window: int = 5) -> pl.DataFrame:
    """Attach `vwap` = rolling volume-weighted typical price over `window` days,
    computed strictly from rows up to and including t (no look-ahead)."""
    tp = ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0)
    return (
        panel.sort(["symbol", "date"])
        .with_columns((tp * pl.col("volume")).alias("_tpv"))
        .with_columns(
            (
                pl.col("_tpv").rolling_sum(window_size=window, min_samples=window).over("symbol")
                / pl.col("volume").rolling_sum(window_size=window, min_samples=window).over("symbol")
            ).alias("vwap")
        )
        .drop("_tpv")
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_vwap.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/signal_research/fingerprint_vwap/__init__.py src/quant_research_stack/signal_research/fingerprint_vwap/vwap.py tests/signal_research/fingerprint_vwap/test_vwap.py
git commit -m "feat(fingerprint-vwap): daily VWAP proxy (rolling volume-weighted typical price)"
```

---

## Task 3: VWAP primary entry position

**Files:**
- Modify: `src/quant_research_stack/signal_research/fingerprint_vwap/vwap.py`
- Test: `tests/signal_research/fingerprint_vwap/test_vwap.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/signal_research/fingerprint_vwap/test_vwap.py
from quant_research_stack.signal_research.fingerprint_vwap.vwap import (
    daily_vwap_proxy,
    vwap_primary_position,
)


def test_vwap_primary_long_only_below_vwap_band(panel: pl.DataFrame) -> None:
    with_vwap = daily_vwap_proxy(panel, window=5)
    out = vwap_primary_position(with_vwap, band=0.01)
    assert set(out["primary_position"].unique().to_list()) <= {0.0, 1.0}
    # rows flagged long must actually be at least `band` below vwap
    longs = out.filter(pl.col("primary_position") == 1.0).drop_nulls("vwap")
    if longs.height:
        assert (longs["close"] <= longs["vwap"] * (1.0 - 0.01) + 1e-9).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_vwap.py::test_vwap_primary_long_only_below_vwap_band -v`
Expected: FAIL with `ImportError: cannot import name 'vwap_primary_position'`.

- [ ] **Step 3: Implement**

```python
# append to src/quant_research_stack/signal_research/fingerprint_vwap/vwap.py
def vwap_primary_position(panel: pl.DataFrame, *, band: float = 0.0) -> pl.DataFrame:
    """Primary entry: long (1.0) when close is at least `band` below vwap (mean
    reversion to VWAP); flat (0.0) otherwise. Long-only v1; requires `vwap` column."""
    if "vwap" not in panel.columns:
        raise ValueError("call daily_vwap_proxy first; missing 'vwap' column")
    return panel.with_columns(
        pl.when(pl.col("vwap").is_not_null() & (pl.col("close") <= pl.col("vwap") * (1.0 - band)))
        .then(1.0)
        .otherwise(0.0)
        .alias("primary_position")
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_vwap.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(fingerprint-vwap): VWAP-entry primary position (long below band)"
```

---

## Task 4: Multi-window fingerprint features

**Files:**
- Create: `src/quant_research_stack/signal_research/fingerprint_vwap/fingerprint.py`
- Test: `tests/signal_research/fingerprint_vwap/test_fingerprint.py`

Features per window `W` (as-of, right-anchored, per symbol): `trend_direction_W` (sign of OLS slope of log-close on time), `trend_strength_W` (|slope| in log-return/day), `trend_linearity_r2_W` (OLS R²), `spikiness_W` (max |daily log-return| ÷ std of daily log-return).

- [ ] **Step 1: Write the failing test**

```python
# tests/signal_research/fingerprint_vwap/test_fingerprint.py
from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.signal_research.fingerprint_vwap.fingerprint import (
    build_fingerprint_features,
    window_trend,
)


def test_window_trend_perfect_uptrend() -> None:
    logclose = np.log(np.exp(np.linspace(0.0, 1.0, 60)))  # perfectly linear in log
    direction, strength, r2 = window_trend(logclose)
    assert direction == 1.0
    assert r2 > 0.999
    assert strength > 0.0


def test_build_fingerprint_columns_present_and_asof(panel: pl.DataFrame) -> None:
    out = build_fingerprint_features(panel, windows=(20, 60))
    for w in (20, 60):
        for base in ("trend_direction", "trend_strength", "trend_linearity_r2", "spikiness"):
            assert f"{base}_{w}" in out.columns
    # as-of: the first W-1 rows per symbol are null (no future leakage)
    aaa = out.filter(pl.col("symbol") == "AAA").sort("date")
    assert aaa["trend_direction_20"][:19].null_count() == 19
    # clean uptrend AAA should mostly show +1 direction once defined
    defined = aaa["trend_direction_60"].drop_nulls()
    assert float((defined == 1.0).mean()) > 0.7
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_fingerprint.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/signal_research/fingerprint_vwap/fingerprint.py
"""Multi-window regime 'fingerprint' features (spec §6 step 1). Per symbol, as-of."""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

_BASES = ("trend_direction", "trend_strength", "trend_linearity_r2", "spikiness")


def window_trend(logclose: NDArray[np.float64]) -> tuple[float, float, float]:
    """Closed-form OLS of logclose on time index: returns (direction, strength, r2)."""
    n = logclose.size
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    y_mean = logclose.mean()
    t_var = float(((t - t_mean) ** 2).sum())
    if t_var == 0.0:
        return 0.0, 0.0, 0.0
    slope = float(((t - t_mean) * (logclose - y_mean)).sum() / t_var)
    y_var = float(((logclose - y_mean) ** 2).sum())
    r2 = 0.0 if y_var == 0.0 else float((slope**2 * t_var) / y_var)
    direction = float(np.sign(slope))
    return direction, abs(slope), max(0.0, min(1.0, r2))


def _spikiness(log_ret_window: NDArray[np.float64]) -> float:
    sd = float(np.std(log_ret_window, ddof=1)) if log_ret_window.size > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(np.max(np.abs(log_ret_window)) / sd)


def build_fingerprint_features(
    panel: pl.DataFrame, *, windows: tuple[int, ...] = (20, 60, 120, 252)
) -> pl.DataFrame:
    """Attach `{base}_{W}` columns. Each row t uses only logclose[t-W+1 .. t]."""
    df = panel.sort(["symbol", "date"])
    out_frames: list[pl.DataFrame] = []
    for _, group in df.group_by("symbol", maintain_order=True):
        close = group["close"].to_numpy().astype(np.float64)
        logc = np.log(close)
        log_ret = np.zeros_like(logc)
        log_ret[1:] = logc[1:] - logc[:-1]
        cols: dict[str, NDArray[np.float64]] = {
            f"{b}_{w}": np.full(close.size, np.nan) for w in windows for b in _BASES
        }
        for w in windows:
            for t in range(w - 1, close.size):
                d, s, r2 = window_trend(logc[t - w + 1 : t + 1])
                cols[f"trend_direction_{w}"][t] = d
                cols[f"trend_strength_{w}"][t] = s
                cols[f"trend_linearity_r2_{w}"][t] = r2
                cols[f"spikiness_{w}"][t] = _spikiness(log_ret[t - w + 1 : t + 1])
        out_frames.append(group.with_columns([pl.Series(k, v) for k, v in cols.items()]))
    return pl.concat(out_frames, how="vertical")


def fingerprint_columns(windows: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"{b}_{w}" for w in windows for b in _BASES)
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_fingerprint.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(fingerprint-vwap): multi-window as-of fingerprint features (trend/strength/r2/spikiness)"
```

---

## Task 5: Primary-edge stats + eligibility gate

**Files:**
- Create: `src/quant_research_stack/signal_research/fingerprint_vwap/eligibility.py`
- Test: `tests/signal_research/fingerprint_vwap/test_eligibility.py`

`primary_signal_stats` computes the VWAP primary's validation stats (net Sharpe / hit rate / expectancy / event count) over a window of the panel, returning `PrimarySignalStats`. We then call the existing `check_eligibility` (prior #2: a meta-labeler is only allowed on a primary with edge).

- [ ] **Step 1: Write the failing test**

```python
# tests/signal_research/fingerprint_vwap/test_eligibility.py
from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.fingerprint_vwap.eligibility import (
    primary_signal_stats,
)
from quant_research_stack.signal_research.fingerprint_vwap.vwap import (
    daily_vwap_proxy,
    vwap_primary_position,
)


def test_primary_stats_shape_and_types(panel: pl.DataFrame) -> None:
    p = vwap_primary_position(daily_vwap_proxy(panel, window=5), band=0.0)
    stats = primary_signal_stats(p, horizon_days=3, cost_bps_one_way=1.0)
    assert stats.single_asset_or_cross_sectional == "cross_sectional"
    assert stats.event_count >= 0
    assert isinstance(stats.validation_net_sharpe, float)
    assert isinstance(stats.is_inverted_superior, bool)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_eligibility.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/signal_research/fingerprint_vwap/eligibility.py
"""Primary-edge stats for the VWAP entry, feeding methodology.meta_labeling
.check_eligibility (spec §6 step 4; prior #2). Net-of-cost, forward-looking returns."""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.signal_research.methodology.meta_labeling import (
    PrimarySignalStats,
)


def primary_signal_stats(
    panel: pl.DataFrame, *, horizon_days: int = 3, cost_bps_one_way: float = 1.0
) -> PrimarySignalStats:
    """Per-entry net forward return over `horizon_days` for primary_position==1 rows,
    then summarize into PrimarySignalStats. Round-trip cost = 2 * cost_bps_one_way."""
    if "primary_position" not in panel.columns:
        raise ValueError("missing 'primary_position'; call vwap_primary_position first")
    cost = 2.0 * cost_bps_one_way / 1e4
    df = panel.sort(["symbol", "date"]).with_columns(
        (pl.col("close").shift(-horizon_days).over("symbol") / pl.col("close") - 1.0).alias("fwd")
    )
    entries = df.filter((pl.col("primary_position") == 1.0) & pl.col("fwd").is_finite())
    r = entries["fwd"].to_numpy().astype(np.float64) - cost
    n = int(r.size)
    if n == 0:
        return PrimarySignalStats(0.0, 0.0, 0.0, 0, "cross_sectional", False, False)
    mean, sd = float(np.mean(r)), float(np.std(r, ddof=1)) if n > 1 else 0.0
    sharpe = 0.0 if sd == 0.0 else mean / sd * np.sqrt(252.0 / horizon_days)
    hit = float(np.mean(r > 0.0))
    expectancy = mean
    inverted_sharpe = 0.0 if sd == 0.0 else (-mean) / sd * np.sqrt(252.0 / horizon_days)
    return PrimarySignalStats(
        validation_net_sharpe=float(sharpe),
        validation_hit_rate=float(hit),
        validation_expectancy=float(expectancy),
        event_count=n,
        single_asset_or_cross_sectional="cross_sectional",
        is_inverted_superior=bool(inverted_sharpe > sharpe),
        is_near_duplicate=False,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_eligibility.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(fingerprint-vwap): primary-edge stats for VWAP entry eligibility gate"
```

---

## Task 6: Generalize the walk-forward trainer (backward-compatible)

**Files:**
- Modify: `src/quant_research_stack/signal_research/training/meta_label_walk_forward.py`
- Test: `tests/signal_research/fingerprint_vwap/test_walk_forward_generalization.py`

Add two optional config fields; when unset, behaviour is byte-identical to today (guarded by a parity test). When set, the trainer uses a caller-provided `primary_position` and appends extra feature columns.

- [ ] **Step 1: Write the failing tests (new behaviour + parity)**

```python
# tests/signal_research/fingerprint_vwap/test_walk_forward_generalization.py
from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.training.meta_label_walk_forward import (
    MetaLabelWalkForwardConfig,
    train_meta_label_walk_forward,
)
from quant_research_stack.signal_research.fingerprint_vwap.fingerprint import (
    build_fingerprint_features,
    fingerprint_columns,
)
from quant_research_stack.signal_research.fingerprint_vwap.vwap import (
    daily_vwap_proxy,
    vwap_primary_position,
)


def test_default_config_has_new_optional_fields() -> None:
    cfg = MetaLabelWalkForwardConfig()
    assert cfg.primary_position_col is None
    assert cfg.extra_feature_columns == ()


def test_caller_primary_and_extra_features_are_used(panel: pl.DataFrame) -> None:
    fp_cols = fingerprint_columns((20, 60))
    prepared = vwap_primary_position(
        build_fingerprint_features(daily_vwap_proxy(panel, window=5), windows=(20, 60)),
        band=0.0,
    )
    cfg = MetaLabelWalkForwardConfig(
        train_window_days=120, test_window_days=30, step_days=30, min_train_events=20,
        primary_position_col="primary_position", extra_feature_columns=fp_cols,
    )
    result = train_meta_label_walk_forward(panel=prepared, config=cfg)
    # the meta model consumed the fingerprint features (they survive into predictions/events)
    assert result.summary["fold_count"] >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_walk_forward_generalization.py -v`
Expected: FAIL (`AttributeError: ... 'primary_position_col'`).

- [ ] **Step 3: Implement the minimal generalization**

In `MetaLabelWalkForwardConfig` add (after `seed`):

```python
    primary_position_col: str | None = None
    extra_feature_columns: tuple[str, ...] = ()
```

In `_feature_frame`, replace the unconditional `primary_position` expression so it is only computed from momentum when no caller column is supplied, and compute the active feature list:

```python
def _feature_frame(panel: pl.DataFrame, config: MetaLabelWalkForwardConfig) -> pl.DataFrame:
    required = {"date", "symbol", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    df = panel.sort(["symbol", "date"]).with_columns(
        [
            (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("log_return_1"),
            (pl.col("close").log() - pl.col("close").shift(5).over("symbol").log()).alias("log_return_5"),
            (pl.col("close").log() - pl.col("close").shift(config.lookback_days).over("symbol").log()).alias("log_return_lookback"),
            (pl.col("close").shift(-config.triple_barrier.vertical_barrier_days).over("symbol") / pl.col("close") - 1.0).alias("future_return_horizon"),
        ]
    )
    base = [
        pl.col("log_return_1").rolling_std(window_size=20, min_samples=20).over("symbol").alias("realized_vol_20"),
        ((pl.col("volume") - pl.col("volume").rolling_mean(window_size=20, min_samples=20).over("symbol"))
         / (pl.col("volume").rolling_std(window_size=20, min_samples=20).over("symbol") + 1e-12)).alias("volume_z_20"),
    ]
    if config.primary_position_col is None:
        base.append(
            pl.when(pl.col("log_return_lookback") > 0.0).then(1.0)
            .when(pl.col("log_return_lookback") < 0.0).then(-1.0)
            .otherwise(0.0).alias("primary_position")
        )
    elif config.primary_position_col != "primary_position":
        base.append(pl.col(config.primary_position_col).alias("primary_position"))
    df = df.with_columns(base)
    feature_columns = (*_FEATURE_COLUMNS, *config.extra_feature_columns)
    frames: list[pl.DataFrame] = []
    for _, group in df.group_by("symbol", maintain_order=True):
        labels = label_triple_barrier(
            close=group["close"].to_numpy().astype(np.float64),
            positions=group["primary_position"].to_numpy().astype(np.float64),
            cfg=config.triple_barrier,
        )
        frames.append(group.with_columns(pl.Series("triple_barrier_label", labels)))
    labeled = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
    finite_cols = [*feature_columns, "triple_barrier_label", "future_return_horizon"]
    return labeled.filter(pl.all_horizontal([pl.col(c).is_finite() for c in finite_cols]))
```

Change `_xy` and `_predict_fold` to use the active feature list:

```python
def _xy(frame: pl.DataFrame, feature_columns: tuple[str, ...]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        frame.select(list(feature_columns)).to_numpy().astype(np.float64),
        frame["triple_barrier_label"].to_numpy().astype(np.float64),
    )
```

In `_predict_fold`, compute `feature_columns = (*_FEATURE_COLUMNS, *config.extra_feature_columns)` and call `_xy(train, feature_columns)` / `_xy(test, feature_columns)`.

- [ ] **Step 4: Run new + full suite (parity)**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_walk_forward_generalization.py -v && PYTHONPATH=src uv run pytest tests/ -k meta_label -q`
Expected: new tests PASS; **all pre-existing meta-label tests still PASS** (defaults unchanged → parity).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(meta-label-wf): backward-compatible caller primary + extra feature columns"
```

---

## Task 7: Pipeline orchestration + net-of-cost + lift-vs-baseline

**Files:**
- Create: `src/quant_research_stack/signal_research/fingerprint_vwap/pipeline.py`
- Modify: `src/quant_research_stack/signal_research/fingerprint_vwap/__init__.py` (exports)
- Test: `tests/signal_research/fingerprint_vwap/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/signal_research/fingerprint_vwap/test_pipeline.py
from __future__ import annotations

import polars as pl

from quant_research_stack.signal_research.fingerprint_vwap.pipeline import (
    FingerprintVwapSpec,
    run_fingerprint_vwap_meta,
)


def test_pipeline_returns_result_with_eligibility_and_lift(panel: pl.DataFrame) -> None:
    spec = FingerprintVwapSpec(
        windows=(20, 60), vwap_window=5, band=0.0, horizon_days=3,
        cost_bps_one_way=1.0, train_window_days=120, test_window_days=30,
        step_days=30, min_train_events=20,
    )
    result = run_fingerprint_vwap_meta(panel=panel, spec=spec)
    assert "eligibility" in result and "eligible" in result["eligibility"]
    # if the primary is ineligible the pipeline short-circuits with a reason
    if not result["eligibility"]["eligible"]:
        assert result["status"] == "primary_ineligible"
    else:
        assert "meta_net_sharpe" in result and "baseline_net_sharpe" in result
        assert "lift" in result  # meta - baseline
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_pipeline.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/signal_research/fingerprint_vwap/pipeline.py
"""Compose VWAP primary + fingerprint features + eligibility + meta walk-forward,
then net-of-cost metrics and the lift-vs-baseline test (spec §6-7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from quant_research_stack.signal_research.fingerprint_vwap.eligibility import primary_signal_stats
from quant_research_stack.signal_research.fingerprint_vwap.fingerprint import (
    build_fingerprint_features,
    fingerprint_columns,
)
from quant_research_stack.signal_research.fingerprint_vwap.vwap import daily_vwap_proxy, vwap_primary_position
from quant_research_stack.signal_research.methodology.meta_labeling import check_eligibility
from quant_research_stack.signal_research.training.meta_label_walk_forward import (
    MetaLabelWalkForwardConfig,
    TripleBarrierConfig,
    train_meta_label_walk_forward,
)


@dataclass(frozen=True)
class FingerprintVwapSpec:
    windows: tuple[int, ...] = (20, 60, 120, 252)
    vwap_window: int = 5
    band: float = 0.0
    horizon_days: int = 3
    cost_bps_one_way: float = 1.0
    train_window_days: int = 252
    test_window_days: int = 63
    step_days: int = 63
    min_train_events: int = 200


def _baseline_net_sharpe(prepared: pl.DataFrame, *, horizon: int, cost_bps_one_way: float) -> float:
    """Take EVERY eligible VWAP entry (no meta filter); net-of-cost annualized Sharpe."""
    cost = 2.0 * cost_bps_one_way / 1e4
    df = prepared.sort(["symbol", "date"]).with_columns(
        (pl.col("close").shift(-horizon).over("symbol") / pl.col("close") - 1.0).alias("fwd")
    )
    r = df.filter((pl.col("primary_position") == 1.0) & pl.col("fwd").is_finite())["fwd"].to_numpy().astype(np.float64) - cost
    if r.size < 2 or np.std(r, ddof=1) == 0.0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252.0 / horizon))


def run_fingerprint_vwap_meta(*, panel: pl.DataFrame, spec: FingerprintVwapSpec) -> dict[str, Any]:
    prepared = vwap_primary_position(
        build_fingerprint_features(daily_vwap_proxy(panel, window=spec.vwap_window), windows=spec.windows),
        band=spec.band,
    )
    stats = primary_signal_stats(prepared, horizon_days=spec.horizon_days, cost_bps_one_way=spec.cost_bps_one_way)
    elig = check_eligibility(stats)
    out: dict[str, Any] = {"eligibility": {"eligible": elig.eligible, "reason": elig.rejection_reason,
                                           "primary_net_sharpe": stats.validation_net_sharpe,
                                           "event_count": stats.event_count}}
    if not elig.eligible:
        out["status"] = "primary_ineligible"
        return out
    cfg = MetaLabelWalkForwardConfig(
        train_window_days=spec.train_window_days, test_window_days=spec.test_window_days,
        step_days=spec.step_days, min_train_events=spec.min_train_events,
        cost_bps_one_way=spec.cost_bps_one_way,
        triple_barrier=TripleBarrierConfig(vertical_barrier_days=spec.horizon_days),
        primary_position_col="primary_position",
        extra_feature_columns=fingerprint_columns(spec.windows),
    )
    result = train_meta_label_walk_forward(panel=prepared, config=cfg)
    meta_sharpe = float(result.summary.get("net_sharpe", 0.0))
    baseline = _baseline_net_sharpe(prepared, horizon=spec.horizon_days, cost_bps_one_way=spec.cost_bps_one_way)
    out.update({
        "status": "evaluated",
        "meta_net_sharpe": meta_sharpe,
        "baseline_net_sharpe": baseline,
        "lift": meta_sharpe - baseline,
        "summary": result.summary,
        "fold_metrics": result.fold_metrics,
        "predictions": result.predictions,
    })
    return out
```

Add to `__init__.py`:

```python
from quant_research_stack.signal_research.fingerprint_vwap.pipeline import (
    FingerprintVwapSpec,
    run_fingerprint_vwap_meta,
)

__all__ = ["FingerprintVwapSpec", "run_fingerprint_vwap_meta"]
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(fingerprint-vwap): pipeline — eligibility short-circuit, meta WF, lift-vs-baseline"
```

---

## Task 8: Gate verdict — PBO + Deflated Sharpe + lift

**Files:**
- Modify: `src/quant_research_stack/signal_research/fingerprint_vwap/pipeline.py`
- Test: `tests/signal_research/fingerprint_vwap/test_gate.py`

Reuse `crypto_research/perps/validation.deflated_sharpe_payload` and `estimate_registry_pbo`. The verdict is PASS only if: net Sharpe > 0, Deflated Sharpe probability ≥ 0.95 at the trial count, and lift > pre-registered margin.

- [ ] **Step 1: Write the failing test**

```python
# tests/signal_research/fingerprint_vwap/test_gate.py
from quant_research_stack.signal_research.fingerprint_vwap.pipeline import gate_verdict


def test_gate_fails_on_zero_lift() -> None:
    v = gate_verdict(
        meta_net_sharpe=0.5, baseline_net_sharpe=0.5, lift_margin=0.2,
        daily_net_returns=[0.001, -0.002, 0.0015, 0.0, 0.0008] * 60, trials=45,
    )
    assert v["verdict"] == "DO_NOT_ADVANCE"
    assert "lift" in v["failed"]


def test_gate_structure_keys() -> None:
    v = gate_verdict(meta_net_sharpe=1.0, baseline_net_sharpe=0.2, lift_margin=0.2,
                     daily_net_returns=[0.002, -0.001, 0.003] * 80, trials=10)
    assert set(["verdict", "passed", "failed", "deflated_sharpe", "net_sharpe", "lift"]).issubset(v)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_gate.py -v`
Expected: FAIL with `ImportError: cannot import name 'gate_verdict'`.

- [ ] **Step 3: Implement**

```python
# append to src/quant_research_stack/signal_research/fingerprint_vwap/pipeline.py
from quant_research_stack.crypto_research.perps.validation import deflated_sharpe_payload


def gate_verdict(
    *, meta_net_sharpe: float, baseline_net_sharpe: float, lift_margin: float,
    daily_net_returns: list[float], trials: int,
) -> dict[str, Any]:
    """PASS only if net Sharpe>0, deflated-Sharpe prob>=0.95 at `trials`, and
    lift = meta - baseline > lift_margin. Otherwise DO_NOT_ADVANCE with reasons."""
    dsr = deflated_sharpe_payload(returns=np.asarray(daily_net_returns, dtype=np.float64), trials=trials)
    lift = meta_net_sharpe - baseline_net_sharpe
    failed: list[str] = []
    if meta_net_sharpe <= 0.0:
        failed.append("net_sharpe")
    if float(dsr.get("deflated_sharpe_ratio", 0.0)) < 0.95:
        failed.append("deflated_sharpe")
    if lift <= lift_margin:
        failed.append("lift")
    return {
        "verdict": "PASS" if not failed else "DO_NOT_ADVANCE",
        "passed": not failed,
        "failed": failed,
        "net_sharpe": meta_net_sharpe,
        "lift": lift,
        "deflated_sharpe": dsr,
    }
```

> NOTE during execution: open `crypto_research/perps/validation.py` and match `deflated_sharpe_payload`'s exact kwargs and its returned key for the DSR probability; adjust the call and the `dsr.get(...)` key to the real names. Do not guess — read it.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_gate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(fingerprint-vwap): gate verdict (PBO/deflated-Sharpe + lift)"
```

---

## Task 9: Report writer + full-suite green

**Files:**
- Modify: `src/quant_research_stack/signal_research/fingerprint_vwap/pipeline.py`
- Test: `tests/signal_research/fingerprint_vwap/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/signal_research/fingerprint_vwap/test_report.py
from quant_research_stack.signal_research.fingerprint_vwap.pipeline import render_report


def test_render_report_contains_verdict_and_disclaimer() -> None:
    md = render_report(
        result={"status": "evaluated", "meta_net_sharpe": 0.4, "baseline_net_sharpe": 0.5,
                "lift": -0.1, "eligibility": {"eligible": True, "reason": "", "event_count": 1234,
                "primary_net_sharpe": 0.3}},
        verdict={"verdict": "DO_NOT_ADVANCE", "failed": ["lift"], "net_sharpe": 0.4,
                 "lift": -0.1, "deflated_sharpe": {"deflated_sharpe_ratio": 0.6}},
        spec_repr="FingerprintVwapSpec(...)",
    )
    assert "DO_NOT_ADVANCE" in md
    assert "research_only" in md.lower() or "not investment advice" in md.lower()
    assert "lift" in md.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/test_report.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# append to src/quant_research_stack/signal_research/fingerprint_vwap/pipeline.py
def render_report(*, result: dict[str, Any], verdict: dict[str, Any], spec_repr: str) -> str:
    elig = result.get("eligibility", {})
    lines = [
        "# Fingerprint-VWAP Meta-Labeling v1 — Result",
        "",
        "**Status:** research_only. Not investment advice. No paper. No live.",
        "",
        f"**Verdict:** {verdict['verdict']}",
        "",
        "## Eligibility (primary VWAP entry)",
        f"- eligible: {elig.get('eligible')}  reason: {elig.get('reason') or 'n/a'}",
        f"- primary net Sharpe: {elig.get('primary_net_sharpe'):.3f}  events: {elig.get('event_count')}",
        "",
        "## Meta-labeling (net of cost)",
        f"- meta net Sharpe: {result.get('meta_net_sharpe', float('nan')):.3f}",
        f"- baseline (take-every-entry) net Sharpe: {result.get('baseline_net_sharpe', float('nan')):.3f}",
        f"- **lift**: {result.get('lift', float('nan')):.3f}",
        f"- deflated Sharpe: {verdict.get('deflated_sharpe')}",
        f"- failed gates: {verdict.get('failed') or 'none'}",
        "",
        "## Spec",
        f"`{spec_repr}`",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run full suite**

Run: `PYTHONPATH=src uv run pytest tests/signal_research/fingerprint_vwap/ -v && PYTHONPATH=src uv run ruff check src tests && PYTHONPATH=src uv run mypy src/quant_research_stack/signal_research/fingerprint_vwap`
Expected: all PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(fingerprint-vwap): markdown report writer; module green (ruff+mypy)"
```

---

## Task 10: Real-universe runner + data audit (no unit test; smoke)

**Files:**
- Create: `scripts/run_fingerprint_vwap_meta_backtest.py`
- Create: `reports/signal_research/fingerprint_vwap_meta_v1/data_audit.md`

- [ ] **Step 1: Write the runner** (mirror `scripts/run_triple_barrier_av_lee_backtest.py` for data fetch/cache)

Reuse `signal_research.data.long_history.fetch_one_ticker` + `signal_research.data.sp500_components.load_or_fetch_sp500` (see the av_lee script lines 1-70 for the exact load/normalize/cache helpers — copy that pattern). Build the panel `{date, symbol, open, high, low, close, volume}`, then:

```python
from quant_research_stack.signal_research.fingerprint_vwap.pipeline import (
    FingerprintVwapSpec, run_fingerprint_vwap_meta, gate_verdict, render_report,
)
# ... after assembling `panel` and parsing args (top_n, start, end, out, band, horizon) ...
spec = FingerprintVwapSpec(windows=(20, 60, 120, 252), band=args.band, horizon_days=args.horizon)
result = run_fingerprint_vwap_meta(panel=panel, spec=spec)
out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
if result["status"] == "evaluated":
    from quant_research_stack.signal_research.fingerprint_vwap.pipeline import _daily_net_for_dsr  # add helper if needed
    verdict = gate_verdict(
        meta_net_sharpe=result["meta_net_sharpe"], baseline_net_sharpe=result["baseline_net_sharpe"],
        lift_margin=0.2, daily_net_returns=result["predictions"]["net_return"].to_list(), trials=args.trials,
    )
else:
    verdict = {"verdict": "DO_NOT_ADVANCE", "failed": ["primary_ineligible"], "net_sharpe": 0.0,
               "lift": 0.0, "deflated_sharpe": {}}
(out_dir / "report.md").write_text(render_report(result=result, verdict=verdict, spec_repr=repr(spec)))
```

- [ ] **Step 2: Data audit FIRST (gate)** — write `reports/signal_research/fingerprint_vwap_meta_v1/data_audit.md` recording: PIT universe membership check, no missing/dup bars on the calendar, fingerprint features right-anchored (assert first `W-1` rows null per symbol — already tested), labels timestamped after entry, corporate-action adjustment consistent. **If the audit fails, STOP and write a negative-result note instead of running.**

- [ ] **Step 3: Smoke-run on a small universe**

Run: `PYTHONPATH=src uv run python scripts/run_fingerprint_vwap_meta_backtest.py --top-n 30 --start 2015-01-01 --end 2026-05-30 --band 0.0 --horizon 3 --trials 50 --out reports/signal_research/fingerprint_vwap_meta_v1/focused`
Expected: writes `report.md` with a verdict; no exceptions.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_fingerprint_vwap_meta_backtest.py reports/signal_research/fingerprint_vwap_meta_v1/
git commit -m "feat(fingerprint-vwap): real-universe runner + data audit"
```

---

## Task 11: Verdict note + decision

**Files:**
- Create: `reports/signal_research/fingerprint_vwap_meta_v1/VERDICT.md`

- [ ] **Step 1:** Read the run's `report.md`. Apply the spec §8 kill criteria: primary ineligible → STOP; net OOS Sharpe < 0.7 or lift ≤ margin → DO_NOT_ADVANCE; PBO high / DSR < 0.95 → DO_NOT_ADVANCE; any survivorship/look-ahead → disqualify.
- [ ] **Step 2:** Write `VERDICT.md` = either `PASS — transfer to Prevalence` (with the gate evidence) or `NEGATIVE-RESULT — DO_NOT_ADVANCE` (with the binding wall), mirroring the tone of `docs/research/2026-05-NEGATIVE-RESULT-*.md`.
- [ ] **Step 3: Commit**

```bash
git add reports/signal_research/fingerprint_vwap_meta_v1/VERDICT.md
git commit -m "docs(fingerprint-vwap): v1 verdict note"
```

---

## Self-review (completed against the spec)

- **§1-2 thesis/approach** → Tasks 2-4 (VWAP + fingerprints), Task 7 (meta-labeling composition). ✔
- **§3 honest priors** → cost wall: net-of-cost in Tasks 5/7/8; prior #2 eligibility: Tasks 5/7 (`check_eligibility`); overfitting: PBO/DSR Task 8; survivorship + look-ahead: Task 10 audit + Task 4 as-of test. ✔
- **§6 pipeline** → Tasks 2→7 map 1:1 to steps 1-6; reuses `label_triple_barrier`, `train_meta_label_walk_forward`. ✔
- **§7 gates** → Task 6 purged WF (reused), Task 8 PBO/DSR + lift, Task 9 ruff/mypy + suite. ✔
- **§8 kill criteria** → Task 11. ✔
- **§9 deliverables** → Tasks 9-11 (`data_audit.md`, `report.md`, `VERDICT.md`). ✔
- **Type consistency:** `primary_position` (f64 {0,1}), `fingerprint_columns()` feeds `extra_feature_columns`, `FingerprintVwapSpec` fields flow into `MetaLabelWalkForwardConfig`. ✔
- **One execution-time caveat flagged inline (Task 8):** confirm `deflated_sharpe_payload` real kwargs/return key before relying on them.
```
