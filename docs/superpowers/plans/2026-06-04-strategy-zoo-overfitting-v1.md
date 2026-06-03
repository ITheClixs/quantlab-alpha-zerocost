# Strategy-Zoo Backtest-Overfitting Demonstration v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale the existing `strategy_benchmark` framework to ~100k single-asset strategy configurations and empirically prove the backtest-overfitting thesis (best-of-N Sharpe ≈ √(2 ln N) by chance, PBO≈1, DSR pass≈0, OOS collapse, permutation-null match), with advanced figures embedded in the README.

**Architecture:** Reuse `strategy_benchmark/{signals,backtest,pbo,dsr}.py`. Reach ~100k by composing the existing 15 (+6 new cited) single-asset signal families with **generic grid axes** — volatility estimator (×3), position mode (×2), holding period (×3) — over ~10 universes × 8 lookbacks × 8 thresholds. Add a purged IS/OOS split, a permutation null control, and a deterministic figures script. All research_only; nothing transfers to Prevalence.

**Tech Stack:** Python 3.11, Polars, NumPy, scikit-free, matplotlib, pytest. Run with `PYTHONPATH=src`. Spec: `docs/research/intake/2026-06-04-strategy-zoo-overfitting-v1.md`.

---

## File structure

| Path | Responsibility |
|---|---|
| `src/quant_research_stack/strategy_benchmark/zoo/__init__.py` | package marker + public exports |
| `…/zoo/vol_estimators.py` | `rolling_vol(bars, *, window, estimator)` — close-to-close / Parkinson / Rogers-Satchell |
| `…/zoo/transforms.py` | `apply_position_mode`, `apply_vol_target`, `apply_holding`, `build_strategy_signal` |
| `…/zoo/grid.py` | `GridConfig`, `ZooStrategySpec`, `enumerate_zoo`, `DEFAULT_GRID` |
| `…/zoo/runner.py` | `run_zoo` (enumerate → backtest → (T,N) float32 matrix → IS metrics + PBO + purged OOS) |
| `…/zoo/analysis.py` | `deflate_top_k`, `expected_vs_empirical_tiers`, `oos_decay` |
| `…/zoo/permutation.py` | `permutation_control` (seeded time-shuffle null) |
| `src/quant_research_stack/strategy_benchmark/signals.py` | **MODIFY**: add 6 cited families + register in `SIGNAL_FAMILIES` |
| `src/quant_research_stack/strategy_benchmark/data.py` | **MODIFY**: add universes (IWM, DIA, XLK, XLF, XLE + EW baskets) |
| `scripts/run_strategy_zoo_overfitting.py` | CLI: data audit + tiered run (1k/10k/100k) + write artifacts |
| `scripts/make_strategy_zoo_figures.py` | F1–F5 from artifacts |
| `tests/strategy_benchmark/zoo/test_*.py` | unit tests |
| `reports/signal_research/strategy_zoo_overfitting_v1/` | data_audit, artifacts, report, VERDICT, figures |

Signal contract (unchanged): `signal_fn(bars: pl.DataFrame, *, lookback: int, threshold: float) -> pl.Series`. Bars columns: `date,symbol,open,high,low,close,volume`.

---

## Task 1: Volatility estimators

**Files:** Create `…/zoo/__init__.py` (empty), `…/zoo/vol_estimators.py`; Test `tests/strategy_benchmark/zoo/__init__.py` (empty) + `tests/strategy_benchmark/zoo/test_vol_estimators.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/strategy_benchmark/zoo/test_vol_estimators.py
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_research_stack.strategy_benchmark.zoo.vol_estimators import rolling_vol


def _bars(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * 1.01
    low = close * 0.99
    open_ = close * (1 + rng.normal(0, 0.002, n))
    return pl.DataFrame({"date": list(range(n)), "symbol": ["X"] * n,
                         "open": open_, "high": high, "low": low, "close": close,
                         "volume": [1e6] * n})


@pytest.mark.parametrize("est", ["close_to_close", "parkinson", "rogers_satchell"])
def test_rolling_vol_positive_and_asof(est: str) -> None:
    out = rolling_vol(_bars(), window=20, estimator=est)
    assert isinstance(out, pl.Series)
    assert out.len() == 60
    # first window-1 are null (as-of); later values strictly positive & finite
    assert out[:19].null_count() == 19
    defined = out.drop_nulls()
    assert (defined > 0).all() and np.isfinite(defined.to_numpy()).all()


def test_unknown_estimator_raises() -> None:
    with pytest.raises(ValueError):
        rolling_vol(_bars(), window=20, estimator="bogus")
```

- [ ] **Step 2: Run — expect FAIL (ImportError).**
`PYTHONPATH=src uv run pytest tests/strategy_benchmark/zoo/test_vol_estimators.py -v`

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/strategy_benchmark/zoo/vol_estimators.py
"""Rolling volatility estimators (daily, as-of). Each returns annualised-agnostic
per-day vol (std of log scale). Estimators: close-to-close, Parkinson (1980),
Rogers-Satchell (1991)."""

from __future__ import annotations

import numpy as np
import polars as pl

_ESTIMATORS = ("close_to_close", "parkinson", "rogers_satchell")


def rolling_vol(bars: pl.DataFrame, *, window: int, estimator: str) -> pl.Series:
    if estimator not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; choose from {_ESTIMATORS}")
    df = bars.sort(["symbol", "date"])
    if estimator == "close_to_close":
        r = (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log())
        vol = r.rolling_std(window_size=window, min_samples=window).over("symbol")
        return df.with_columns(vol.alias("_v"))["_v"]
    if estimator == "parkinson":
        hl = (pl.col("high").log() - pl.col("low").log()) ** 2
        mean_hl = hl.rolling_mean(window_size=window, min_samples=window).over("symbol")
        vol = (mean_hl / (4.0 * np.log(2.0))).sqrt()
        return df.with_columns(vol.alias("_v"))["_v"]
    # rogers_satchell: drift-independent
    rs = (
        (pl.col("high").log() - pl.col("close").log()) * (pl.col("high").log() - pl.col("open").log())
        + (pl.col("low").log() - pl.col("close").log()) * (pl.col("low").log() - pl.col("open").log())
    )
    vol = rs.rolling_mean(window_size=window, min_samples=window).over("symbol").clip(lower_bound=0.0).sqrt()
    return df.with_columns(vol.alias("_v"))["_v"]
```

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(zoo): rolling vol estimators (close-to-close/Parkinson/Rogers-Satchell)"`

---

## Task 2: Signal transforms (position mode, vol-target, holding)

**Files:** Create `…/zoo/transforms.py`; Test `tests/strategy_benchmark/zoo/test_transforms.py`.

- [ ] **Step 1: Failing test**

```python
# tests/strategy_benchmark/zoo/test_transforms.py
from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.strategy_benchmark.zoo.transforms import (
    apply_holding, apply_position_mode, apply_vol_target,
)


def test_position_mode_long_only_clips_negatives() -> None:
    s = pl.Series("s", [1.0, -1.0, 0.5, -0.3])
    assert apply_position_mode(s, mode="long_only").to_list() == [1.0, 0.0, 0.5, 0.0]
    assert apply_position_mode(s, mode="long_short").to_list() == [1.0, -1.0, 0.5, -0.3]


def test_apply_holding_forward_fills_nonzero_for_h_days() -> None:
    s = pl.Series("s", [1.0, 0.0, 0.0, -1.0, 0.0])
    held = apply_holding(s, holding=2)
    # a new position persists for `holding` days (hold the last nonzero entry)
    assert held.to_list()[0] == 1.0 and held.to_list()[1] == 1.0
    assert held.len() == 5


def test_apply_vol_target_scales_inverse_to_vol() -> None:
    s = pl.Series("s", [1.0, 1.0, 1.0])
    vol = pl.Series("v", [0.01, 0.02, None])
    out = apply_vol_target(s, vol=vol, target_daily_vol=0.01)
    # position halves when vol doubles; null vol -> 0 position
    assert abs(out.to_list()[0] - 1.0) < 1e-9
    assert abs(out.to_list()[1] - 0.5) < 1e-9
    assert out.to_list()[2] == 0.0
```

- [ ] **Step 2: Run — expect FAIL (ImportError).**

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/strategy_benchmark/zoo/transforms.py
"""Composable transforms turning a raw family signal into a final position series.
Order: family signal -> position_mode -> vol_target -> holding."""

from __future__ import annotations

import polars as pl

_POSITION_CAP = 3.0  # never lever a single asset beyond 3x in this demo


def apply_position_mode(signal: pl.Series, *, mode: str) -> pl.Series:
    if mode == "long_short":
        return signal
    if mode == "long_only":
        return signal.clip(lower_bound=0.0)
    raise ValueError(f"unknown position mode {mode!r}")


def apply_vol_target(signal: pl.Series, *, vol: pl.Series, target_daily_vol: float) -> pl.Series:
    df = pl.DataFrame({"s": signal, "v": vol}).with_columns(
        pl.when(pl.col("v").is_not_null() & (pl.col("v") > 0.0))
        .then((pl.col("s") * (target_daily_vol / pl.col("v"))).clip(-_POSITION_CAP, _POSITION_CAP))
        .otherwise(0.0)
        .alias("scaled")
    )
    return df["scaled"]


def apply_holding(signal: pl.Series, *, holding: int) -> pl.Series:
    if holding <= 1:
        return signal
    # Hold the last nonzero position for `holding` days: forward-fill nonzeros within horizon.
    df = pl.DataFrame({"s": signal}).with_columns(
        pl.when(pl.col("s") != 0.0).then(pl.col("s")).otherwise(None).alias("_nz")
    ).with_columns(
        pl.col("_nz").fill_null(strategy="forward", limit=holding - 1).alias("held")
    ).with_columns(pl.col("held").fill_null(0.0))
    return df["held"]
```

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(zoo): signal transforms (position mode, vol target, holding)"`

---

## Task 3: Grid config + enumeration

**Files:** Create `…/zoo/grid.py`; Test `tests/strategy_benchmark/zoo/test_grid.py`.

- [ ] **Step 1: Failing test**

```python
# tests/strategy_benchmark/zoo/test_grid.py
from __future__ import annotations

from quant_research_stack.strategy_benchmark.zoo.grid import (
    DEFAULT_GRID, GridConfig, ZooStrategySpec, enumerate_zoo,
)


def test_default_grid_cardinality_and_uniqueness() -> None:
    specs = enumerate_zoo(universes=("U1", "U2"), grid=DEFAULT_GRID)
    expected = (
        2 * len(DEFAULT_GRID.families) * len(DEFAULT_GRID.lookbacks)
        * len(DEFAULT_GRID.thresholds) * len(DEFAULT_GRID.vol_estimators)
        * len(DEFAULT_GRID.position_modes) * len(DEFAULT_GRID.holdings)
    )
    assert len(specs) == expected
    assert len({s.strategy_id for s in specs}) == expected  # ids unique


def test_max_strategies_caps_deterministically() -> None:
    grid = GridConfig(max_strategies=100, seed=7)
    a = enumerate_zoo(universes=("U1", "U2"), grid=grid)
    b = enumerate_zoo(universes=("U1", "U2"), grid=grid)
    assert len(a) == 100 and [s.strategy_id for s in a] == [s.strategy_id for s in b]
```

- [ ] **Step 2: Run — expect FAIL (ImportError).**

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/strategy_benchmark/zoo/grid.py
"""Configurable strategy grid → up to ~100k single-asset configurations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from quant_research_stack.strategy_benchmark.signals import SIGNAL_FAMILIES


@dataclass(frozen=True)
class GridConfig:
    families: tuple[str, ...] = field(default_factory=lambda: tuple(sorted(SIGNAL_FAMILIES.keys())))
    lookbacks: tuple[int, ...] = (5, 10, 20, 40, 60, 120, 180, 252)
    thresholds: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    vol_estimators: tuple[str, ...] = ("close_to_close", "parkinson", "rogers_satchell")
    position_modes: tuple[str, ...] = ("long_only", "long_short")
    holdings: tuple[int, ...] = (1, 5, 10)
    max_strategies: int | None = None
    seed: int = 42


DEFAULT_GRID = GridConfig()


@dataclass(frozen=True)
class ZooStrategySpec:
    strategy_id: str
    universe: str
    family: str
    lookback: int
    threshold: float
    vol_estimator: str
    position_mode: str
    holding: int


def enumerate_zoo(*, universes: Iterable[str], grid: GridConfig) -> list[ZooStrategySpec]:
    specs: list[ZooStrategySpec] = []
    for u, f, lb, th, ve, pm, hd in product(
        sorted(universes), grid.families, grid.lookbacks, grid.thresholds,
        grid.vol_estimators, grid.position_modes, grid.holdings,
    ):
        sid = f"{u}|{f}|L{lb}|T{th:.2f}|{ve}|{pm}|H{hd}"
        specs.append(ZooStrategySpec(sid, u, f, lb, th, ve, pm, hd))
    if grid.max_strategies is not None and len(specs) > grid.max_strategies:
        rng = np.random.default_rng(grid.seed)
        idx = np.sort(rng.choice(len(specs), size=grid.max_strategies, replace=False))
        specs = [specs[i] for i in idx]
    return specs
```

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(zoo): configurable grid + ZooStrategySpec enumeration"`

---

## Task 4: Six new cited signal families

**Files:** Modify `src/quant_research_stack/strategy_benchmark/signals.py` (append 6 functions + register); Test `tests/strategy_benchmark/zoo/test_new_families.py`.

Each follows the existing contract `(bars, *, lookback, threshold) -> pl.Series` returning a position in roughly [-1,1]. Read the top of `signals.py` for the helpers `_rolling_mean/_rolling_std/_rolling_max/_rolling_min` before editing.

- [ ] **Step 1: Failing test**

```python
# tests/strategy_benchmark/zoo/test_new_families.py
from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.strategy_benchmark.signals import SIGNAL_FAMILIES

_NEW = ["VOLMANAGED_MOMENTUM", "EWMA_CROSS", "ATR_TRAILING_TREND",
        "ROLLING_SHARPE_MOM", "RANGE_OSCILLATOR", "MOM_SKIP"]


def _bars(n: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pl.DataFrame({"date": list(range(n)), "symbol": ["X"] * n,
                         "open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": [1e6] * n})


def test_new_families_registered_and_valid() -> None:
    bars = _bars()
    for name in _NEW:
        assert name in SIGNAL_FAMILIES, f"{name} not registered"
        s = SIGNAL_FAMILIES[name](bars, lookback=20, threshold=1.0)
        assert isinstance(s, pl.Series) and s.len() == bars.height
        finite = s.drop_nulls().drop_nans()
        assert finite.len() > 0
        assert finite.abs().max() <= 5.0  # bounded position
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement — append to `signals.py`** (then add the 6 entries to the `SIGNAL_FAMILIES` dict):

```python
def signal_volmanaged_momentum(bars: pl.DataFrame, *, lookback: int, threshold: float) -> pl.Series:
    """Moreira & Muir (2017): momentum sign scaled by inverse realised variance."""
    df = bars.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("_r")
    )
    df = df.with_columns(
        (pl.col("close").log() - pl.col("close").shift(lookback).log()).alias("_mom"),
        pl.col("_r").rolling_std(window_size=lookback, min_samples=lookback).alias("_vol"),
    )
    df = df.with_columns(
        pl.when((pl.col("_vol") > 0))
        .then(pl.col("_mom").sign() * (0.01 / pl.col("_vol")).clip(0.0, 2.0) * (threshold / 2.5))
        .otherwise(0.0).alias("_s")
    )
    return df["_s"]


def signal_ewma_cross(bars: pl.DataFrame, *, lookback: int, threshold: float) -> pl.Series:
    """EWMA crossover (RiskMetrics lineage): fast vs slow exponential MA."""
    fast = max(2, lookback // 4)
    df = bars.with_columns(
        pl.col("close").ewm_mean(span=fast).alias("_f"),
        pl.col("close").ewm_mean(span=lookback).alias("_sl"),
    )
    df = df.with_columns(
        pl.when(pl.col("_f") > pl.col("_sl")).then(1.0 * threshold / 2.5)
        .when(pl.col("_f") < pl.col("_sl")).then(-1.0 * threshold / 2.5)
        .otherwise(0.0).alias("_s")
    )
    return df["_s"]


def signal_atr_trailing_trend(bars: pl.DataFrame, *, lookback: int, threshold: float) -> pl.Series:
    """Wilder (1978) ATR trend: long when close above close[lookback] by k*ATR."""
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs(),
    )
    df = bars.with_columns(tr.alias("_tr"))
    df = df.with_columns(
        pl.col("_tr").rolling_mean(window_size=lookback, min_samples=lookback).alias("_atr"),
        (pl.col("close") - pl.col("close").shift(lookback)).alias("_chg"),
    )
    df = df.with_columns(
        pl.when((pl.col("_atr") > 0) & (pl.col("_chg") > threshold * pl.col("_atr"))).then(1.0)
        .when((pl.col("_atr") > 0) & (pl.col("_chg") < -threshold * pl.col("_atr"))).then(-1.0)
        .otherwise(0.0).alias("_s")
    )
    return df["_s"]


def signal_rolling_sharpe_mom(bars: pl.DataFrame, *, lookback: int, threshold: float) -> pl.Series:
    """Risk-adjusted momentum: sign of rolling mean/std of returns past a threshold."""
    df = bars.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("_r")
    )
    df = df.with_columns(
        (pl.col("_r").rolling_mean(window_size=lookback, min_samples=lookback)
         / (pl.col("_r").rolling_std(window_size=lookback, min_samples=lookback) + 1e-12)).alias("_rs")
    )
    df = df.with_columns(
        pl.when(pl.col("_rs") > threshold / 5.0).then(1.0)
        .when(pl.col("_rs") < -threshold / 5.0).then(-1.0)
        .otherwise(0.0).alias("_s")
    )
    return df["_s"]


def signal_range_oscillator(bars: pl.DataFrame, *, lookback: int, threshold: float) -> pl.Series:
    """Range trading: position from where close sits in its rolling [min,max] band."""
    df = bars.with_columns(
        pl.col("close").rolling_min(window_size=lookback, min_samples=lookback).alias("_lo"),
        pl.col("close").rolling_max(window_size=lookback, min_samples=lookback).alias("_hi"),
    )
    df = df.with_columns(
        pl.when(pl.col("_hi") > pl.col("_lo"))
        .then(((pl.col("close") - pl.col("_lo")) / (pl.col("_hi") - pl.col("_lo"))) * 2.0 - 1.0)
        .otherwise(0.0).alias("_pos01")
    )
    # mean-revert: fade extremes (negative of position-in-range), scaled by threshold
    df = df.with_columns((-pl.col("_pos01") * (threshold / 2.5)).clip(-1.0, 1.0).alias("_s"))
    return df["_s"]


def signal_mom_skip(bars: pl.DataFrame, *, lookback: int, threshold: float) -> pl.Series:
    """Jegadeesh-Titman echo control: momentum over [t-lookback, t-skip], skip last 5d."""
    skip = 5
    df = bars.with_columns(
        (pl.col("close").shift(skip).log() - pl.col("close").shift(lookback).log()).alias("_m")
    )
    df = df.with_columns(
        pl.when(pl.col("_m") > 0).then(1.0 * threshold / 2.5)
        .when(pl.col("_m") < 0).then(-1.0 * threshold / 2.5)
        .otherwise(0.0).alias("_s")
    )
    return df["_s"]
```

Then extend the registry (find `SIGNAL_FAMILIES = {` and add):

```python
    "VOLMANAGED_MOMENTUM": signal_volmanaged_momentum,
    "EWMA_CROSS": signal_ewma_cross,
    "ATR_TRAILING_TREND": signal_atr_trailing_trend,
    "ROLLING_SHARPE_MOM": signal_rolling_sharpe_mom,
    "RANGE_OSCILLATOR": signal_range_oscillator,
    "MOM_SKIP": signal_mom_skip,
```

- [ ] **Step 4: Run — expect PASS. Also run the existing signals tests for parity:** `PYTHONPATH=src uv run pytest tests/ -k "strategy_benchmark or signals" -q` — all green.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(strategy-benchmark): 6 cited single-asset families (vol-managed/EWMA/ATR/rolling-Sharpe/range/skip)"`

---

## Task 5: Expanded universes

**Files:** Modify `src/quant_research_stack/strategy_benchmark/data.py` (extend `UNIVERSES`); Test `tests/strategy_benchmark/zoo/test_universes.py`.

- [ ] **Step 1: Failing test**

```python
# tests/strategy_benchmark/zoo/test_universes.py
from quant_research_stack.strategy_benchmark.data import UNIVERSES


def test_expanded_universe_set() -> None:
    names = {u.name for u in UNIVERSES}
    for required in {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "EW_BASKET"}:
        assert required in names
    assert len(UNIVERSES) >= 10
    # every universe references at least one ticker, no dup names
    assert all(len(u.tickers) >= 1 for u in UNIVERSES)
    assert len(names) == len(UNIVERSES)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** — read the existing `BenchmarkUniverse` dataclass + `UNIVERSES` tuple in `data.py`, then add entries (keep the existing 5):

```python
    BenchmarkUniverse(name="IWM", tickers=("IWM",), description="Russell 2000 ETF"),
    BenchmarkUniverse(name="DIA", tickers=("DIA",), description="Dow 30 ETF"),
    BenchmarkUniverse(name="XLK", tickers=("XLK",), description="Technology sector SPDR"),
    BenchmarkUniverse(name="XLF", tickers=("XLF",), description="Financials sector SPDR"),
    BenchmarkUniverse(name="XLE", tickers=("XLE",), description="Energy sector SPDR"),
    BenchmarkUniverse(name="EW_SECTORS", tickers=("XLK", "XLF", "XLE"), description="Equal-weight sector basket"),
```

(With the existing ES_F, NQ_F, SPY, QQQ, EW_BASKET → 11 universes total.)

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(strategy-benchmark): expand universe set to 11 (IWM/DIA/sector SPDRs + EW sectors)"`

---

## Task 6: Zoo runner (backtest grid → matrix → PBO → purged OOS)

**Files:** Create `…/zoo/runner.py`; Test `tests/strategy_benchmark/zoo/test_runner.py`.

- [ ] **Step 1: Failing test** (tiny synthetic universes; tiny grid)

```python
# tests/strategy_benchmark/zoo/test_runner.py
from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.strategy_benchmark.zoo.grid import GridConfig
from quant_research_stack.strategy_benchmark.zoo.runner import ZooResult, run_zoo


def _bars(symbol: str, n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    return pl.DataFrame({"date": list(range(n)), "symbol": [symbol] * n,
                         "open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": [1e6] * n})


def test_run_zoo_shapes_and_split() -> None:
    universes = {"U1": _bars("U1", seed=1), "U2": _bars("U2", seed=2)}
    grid = GridConfig(families=("TS_MOMENTUM", "MA_CROSSOVER"), lookbacks=(10, 20),
                      thresholds=(1.0,), vol_estimators=("close_to_close",),
                      position_modes=("long_short",), holdings=(1,))
    res = run_zoo(universes=universes, grid=grid, oos_fraction=0.3, embargo_days=5)
    assert isinstance(res, ZooResult)
    n = 2 * 2 * 2 * 1 * 1 * 1 * 1  # universes×families×lookbacks×...
    assert res.metrics.height == n
    assert res.is_returns.shape[1] == n and res.oos_returns.shape[1] == n
    assert res.is_returns.shape[0] > res.oos_returns.shape[0]  # IS longer than OOS tail
    assert 0.0 <= res.pbo["pbo_probability"] <= 1.0
    assert {"strategy_id", "is_sharpe"}.issubset(set(res.metrics.columns))
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** (compose existing `run_single_asset_backtest` + transforms + `compute_pbo`; build a `float32` (T,N) matrix; purge+embargo the OOS tail):

```python
# src/quant_research_stack/strategy_benchmark/zoo/runner.py
"""Run the zoo: enumerate grid, backtest each on IS, assemble (T,N) returns, PBO,
purged+embargoed OOS tail. research_only."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.strategy_benchmark.backtest import BacktestCostConfig, run_single_asset_backtest
from quant_research_stack.strategy_benchmark.pbo import compute_pbo
from quant_research_stack.strategy_benchmark.signals import SIGNAL_FAMILIES
from quant_research_stack.strategy_benchmark.zoo.grid import GridConfig, ZooStrategySpec, enumerate_zoo
from quant_research_stack.strategy_benchmark.zoo.transforms import (
    apply_holding, apply_position_mode, apply_vol_target,
)
from quant_research_stack.strategy_benchmark.zoo.vol_estimators import rolling_vol

_TARGET_DAILY_VOL = 0.01


@dataclass(frozen=True)
class ZooResult:
    specs: list[ZooStrategySpec]
    metrics: pl.DataFrame
    is_returns: NDArray[np.float32]
    oos_returns: NDArray[np.float32]
    pbo: dict[str, Any]
    wall_clock_sec: float


def build_signal(bars: pl.DataFrame, spec: ZooStrategySpec) -> pl.Series:
    raw = SIGNAL_FAMILIES[spec.family](bars, lookback=spec.lookback, threshold=spec.threshold)
    pos = apply_position_mode(raw, mode=spec.position_mode)
    vol = rolling_vol(bars, window=spec.lookback, estimator=spec.vol_estimator)
    pos = apply_vol_target(pos, vol=vol, target_daily_vol=_TARGET_DAILY_VOL)
    return apply_holding(pos, holding=spec.holding)


def run_zoo(*, universes: dict[str, pl.DataFrame], grid: GridConfig,
            oos_fraction: float = 0.3, embargo_days: int = 10,
            cost: BacktestCostConfig | None = None, n_partitions: int = 16) -> ZooResult:
    t0 = time.perf_counter()
    cost = cost or BacktestCostConfig()
    specs = enumerate_zoo(universes=tuple(universes.keys()), grid=grid)
    all_dates = sorted({d for u in universes.values() for d in u["date"].to_list()})
    T = len(all_dates)
    idx = {d: i for i, d in enumerate(all_dates)}
    split = int(T * (1.0 - oos_fraction))
    is_rows = list(range(0, split))
    oos_rows = list(range(min(split + embargo_days, T), T))  # purge+embargo gap
    full = np.zeros((T, len(specs)), dtype=np.float32)
    is_sharpe = np.zeros(len(specs)); oos_sharpe = np.zeros(len(specs)); turn = np.zeros(len(specs))
    for j, spec in enumerate(specs):
        bars = universes[spec.universe]
        sig = build_signal(bars, spec)
        res = run_single_asset_backtest(bars=bars, signals=sig, cost=cost)
        for k, d in enumerate(bars["date"].to_list()):
            full[idx[d], j] = np.float32(res.daily_net_return[k])
        col_is = full[is_rows, j]; col_oos = full[oos_rows, j]
        is_sharpe[j] = _ann_sharpe(col_is); oos_sharpe[j] = _ann_sharpe(col_oos)
        turn[j] = res.annual_turnover
    metrics = pl.DataFrame({
        "strategy_id": [s.strategy_id for s in specs],
        "universe": [s.universe for s in specs], "family": [s.family for s in specs],
        "lookback": [s.lookback for s in specs], "threshold": [s.threshold for s in specs],
        "vol_estimator": [s.vol_estimator for s in specs], "position_mode": [s.position_mode for s in specs],
        "holding": [s.holding for s in specs], "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe,
        "annual_turnover": turn,
    })
    is_mat = full[is_rows]; oos_mat = full[oos_rows]
    pbo = _pbo_dict(compute_pbo(returns=is_mat.astype(np.float64), n_partitions=n_partitions))
    return ZooResult(specs, metrics, is_mat, oos_mat, pbo, time.perf_counter() - t0)


def _ann_sharpe(r: NDArray[np.float64]) -> float:
    r = r[np.isfinite(r)]
    if r.size < 2 or np.std(r, ddof=1) == 0.0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252.0))


def _pbo_dict(p: Any) -> dict[str, Any]:
    return {"pbo_probability": float(p.pbo_probability), "median_logit": float(p.median_logit),
            "n_strategies": int(p.n_strategies), "failure_rate": float(p.failure_rate)}
```

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(zoo): run_zoo — grid backtest, float32 matrix, PBO, purged OOS split"`

---

## Task 7: Multiple-testing analysis (DSR deflation + tiers + OOS decay)

**Files:** Create `…/zoo/analysis.py`; Test `tests/strategy_benchmark/zoo/test_analysis.py`.

- [ ] **Step 1: Failing test**

```python
# tests/strategy_benchmark/zoo/test_analysis.py
from __future__ import annotations

import numpy as np

from quant_research_stack.strategy_benchmark.zoo.analysis import (
    deflate_best, expected_vs_empirical,
)


def test_expected_vs_empirical_rises_with_n() -> None:
    rng = np.random.default_rng(0)
    sharpes = rng.normal(0, 1, 100_000)  # zero-skill pool
    out = expected_vs_empirical(sharpe_estimates=sharpes, tiers=(1_000, 10_000, 100_000))
    # empirical max grows with N and tracks the theoretical curve (within tolerance)
    e = [r["empirical_max"] for r in out]
    assert e[0] < e[1] < e[2]
    for r in out:
        assert abs(r["empirical_max"] - r["theoretical_max"]) < 0.6


def test_deflate_best_rejects_lucky_winner() -> None:
    rng = np.random.default_rng(1)
    T, N = 500, 5_000
    mat = rng.normal(0, 0.01, (T, N))  # all zero-skill
    res = deflate_best(is_returns=mat)
    assert res["dsr"] < 0.95  # the best of 5000 nulls does NOT survive deflation
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** (reuse `dsr.expected_max_sharpe` + `dsr.compute_dsr`):

```python
# src/quant_research_stack/strategy_benchmark/zoo/analysis.py
"""Multiple-testing analysis: theoretical-vs-empirical best Sharpe across tiers,
and Deflated-Sharpe of the in-sample winner."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_research_stack.strategy_benchmark.dsr import compute_dsr, expected_max_sharpe


def expected_vs_empirical(*, sharpe_estimates: NDArray[np.float64],
                          tiers: tuple[int, ...]) -> list[dict[str, Any]]:
    """For each tier N, compare empirical max of the first N annualised Sharpes to the
    theoretical E[max] under an i.i.d.-normal null with the pool's Sharpe variance."""
    var = float(np.var(sharpe_estimates, ddof=1))
    out: list[dict[str, Any]] = []
    for n in tiers:
        n = min(n, sharpe_estimates.size)
        emp = float(np.max(sharpe_estimates[:n]))
        theo = expected_max_sharpe(n_trials=n, sharpe_variance=var)
        out.append({"n_trials": n, "empirical_max": emp, "theoretical_max": theo})
    return out


def deflate_best(*, is_returns: NDArray[np.float64]) -> dict[str, Any]:
    """Annualised-Sharpe of each column, pick the best, deflate it for N trials."""
    r = is_returns.astype(np.float64)
    mu = np.mean(r, axis=0); sd = np.std(r, axis=0, ddof=1)
    sd[sd == 0.0] = np.nan
    sr = np.nan_to_num(mu / sd * np.sqrt(252.0), nan=0.0, posinf=0.0, neginf=0.0)
    best = int(np.argmax(sr))
    dsr_res = compute_dsr(returns=r[:, best], sharpe_estimates=sr, selected_idx=best)
    return {"selected_idx": best, "observed_sharpe": float(sr[best]),
            "expected_max_under_null": float(dsr_res.expected_max_sharpe_under_null),
            "psr_zero": float(dsr_res.psr_zero), "dsr": float(dsr_res.dsr),
            "n_trials": int(sr.size)}
```

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(zoo): multiple-testing analysis (expected-vs-empirical max, deflated best)"`

---

## Task 8: Permutation null control

**Files:** Create `…/zoo/permutation.py`; Test `tests/strategy_benchmark/zoo/test_permutation.py`.

- [ ] **Step 1: Failing test**

```python
# tests/strategy_benchmark/zoo/test_permutation.py
import numpy as np
from quant_research_stack.strategy_benchmark.zoo.permutation import permutation_control


def test_permutation_best_matches_real_for_zero_skill_pool() -> None:
    rng = np.random.default_rng(0)
    mat = rng.normal(0, 0.01, (500, 4_000))  # zero-skill
    out = permutation_control(is_returns=mat, seed=7)
    # for a zero-skill pool the real best Sharpe and the permuted best Sharpe are close
    assert abs(out["real_best_sharpe"] - out["permuted_best_sharpe"]) < 0.7
    assert out["seed"] == 7
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
# src/quant_research_stack/strategy_benchmark/zoo/permutation.py
"""Permutation null control: independently time-shuffle each strategy's IS returns and
recompute the best Sharpe. If the real best ≈ the permuted best, the winner is an artifact."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _best_sharpe(mat: NDArray[np.float64]) -> float:
    mu = np.mean(mat, axis=0); sd = np.std(mat, axis=0, ddof=1)
    sd[sd == 0.0] = np.nan
    sr = np.nan_to_num(mu / sd * np.sqrt(252.0), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.max(sr))


def permutation_control(*, is_returns: NDArray[np.float64], seed: int = 42) -> dict[str, Any]:
    r = is_returns.astype(np.float64)
    rng = np.random.default_rng(seed)
    permuted = np.empty_like(r)
    for j in range(r.shape[1]):
        permuted[:, j] = r[rng.permutation(r.shape[0]), j]
    return {"real_best_sharpe": _best_sharpe(r), "permuted_best_sharpe": _best_sharpe(permuted),
            "seed": seed, "n_strategies": int(r.shape[1])}
```

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(zoo): permutation null control"`

---

## Task 9: Figures (F1–F5)

**Files:** Create `scripts/make_strategy_zoo_figures.py`; Test `tests/strategy_benchmark/zoo/test_figures.py` (smoke: functions return matplotlib Figures from synthetic inputs).

- [ ] **Step 1: Failing test**

```python
# tests/strategy_benchmark/zoo/test_figures.py
import importlib.util, pathlib
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "zf", pathlib.Path(__file__).resolve().parents[3] / "scripts" / "make_strategy_zoo_figures.py")
zf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(zf)


def test_sharpe_distribution_figure_builds() -> None:
    rng = np.random.default_rng(0)
    fig = zf.fig_sharpe_distribution(is_sharpe=rng.normal(0, 1, 10_000),
                                     permuted_sharpe=rng.normal(0, 1, 10_000))
    assert fig is not None
    assert len(fig.axes) >= 1
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** — a matplotlib module with `fig_sharpe_distribution`, `fig_expected_vs_empirical`, `fig_is_oos_decay`, `fig_overfitting_panel`, `fig_family_heatmap`, and a `main()` that loads artifacts from `reports/signal_research/strategy_zoo_overfitting_v1/` and writes `figures/F1..F5.png`. Use `matplotlib.use("Agg")`. Mirror the styling of `scripts/make_landscape_figures.py` (read it for the rcParams + savefig pattern). Each `fig_*` takes plain arrays/DataFrames and returns a `Figure` (so it is unit-testable without files).

- [ ] **Step 4: Run — expect PASS. Visually check one figure render after the real run (Task 10).**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(zoo): F1-F5 figure builders"`

---

## Task 10: CLI runner + data audit + tiered run

**Files:** Create `scripts/run_strategy_zoo_overfitting.py`; Create `reports/signal_research/strategy_zoo_overfitting_v1/data_audit.md`. No unit test (smoke).

- [ ] **Step 1:** Write the CLI. Flags: `--start`, `--end`, `--max-strategies` (int, the tier), `--oos-fraction` (0.3), `--embargo-days` (10), `--out`. It must: fetch the 11 universes via `strategy_benchmark.data.fetch_benchmark_panel` + `build_universe_returns` (reuse exactly; cache under `data/processed/strategy_zoo_overfitting_v1/`), assemble the `universes: dict[str, bars_df]`, call `run_zoo`, then `analysis.expected_vs_empirical` (tiers up to the actual N) + `analysis.deflate_best` + `permutation_control`, write `metrics.parquet`, `tiers.json`, `deflated_best.json`, `permutation_control.json`, `oos_decay.parquet` (is_sharpe vs oos_sharpe of top-K), and `summary.json`. Read `scripts/run_triple_barrier_av_lee_backtest.py` for the fetch/cache pattern.
- [ ] **Step 2:** Write `data_audit.md` (checklist: no missing/dup bars, corporate-action consistency via adj-close, the IS/OOS purge+embargo gap, PIT caveat for ETFs). Status PENDING until the checks run against the real fetched panel; **STOP and write a negative note if a check fails.**
- [ ] **Step 3:** Verify imports + `--help`; ruff clean. Then run tiers **only if data available**: `--max-strategies 1000`, then `10000`, then `100000`. If network/compute unavailable, commit the script + audit and report the run DEFERRED (do not fabricate).
- [ ] **Step 4:** Commit — `git add scripts/run_strategy_zoo_overfitting.py reports/signal_research/strategy_zoo_overfitting_v1/ && git commit -m "feat(zoo): CLI runner + data audit + tiered run"`

---

## Task 11: README section + report + verdict

**Files:** Create `reports/signal_research/strategy_zoo_overfitting_v1/report.md`, `VERDICT.md`; Modify `README.md`.

- [ ] **Step 1:** Generate figures: `PYTHONPATH=src uv run python scripts/make_strategy_zoo_figures.py` → `reports/signal_research/strategy_zoo_overfitting_v1/figures/F1..F5.png`. Copy/symlink the 5 PNGs into repo `figures/` as `zoo_*.png` for README embedding.
- [ ] **Step 2:** Write `report.md` (setup, N actually run, the four measured claims with numbers, wall-clock) and `VERDICT.md` (apply spec §8: expected = demonstration PASS confirming the thesis; if a survivor appears, escalate to its own intake — do not celebrate). Mirror the tone of `reports/strategy_benchmark_sp_nasdaq_2yr.md`.
- [ ] **Step 3:** Add a README.md section **"Empirical proof: ~100k strategies, ≈0 survivors"** after §6, embedding F1–F5 (`figures/zoo_*.png`) with 1–2 sentences each, linking to the report + the competitive-landscape report. Validate links + math: `PYTHONPATH=src uv run python scripts/check_readme_links.py && PYTHONPATH=src uv run python scripts/check_readme_math.py`.
- [ ] **Step 4:** Commit — `git add -A && git commit -m "docs(zoo): report + verdict + README empirical-overfitting section with figures"`

---

## Self-review (against the spec)

- **§1 four claims** → Task 6 (PBO + IS/OOS), Task 7 (expected-vs-empirical + deflated best = claims 1&3), Task 8 (permutation = claim 4), Task 11 (OOS decay = claim 2). ✔
- **§3 grid to ~100k** → Tasks 1-3 (vol-est×3, modes×2, holdings×3 axes) + Task 5 (11 universes) + Task 4 (families) → 11×21×8×8×3×2×3 ≫ 100k; `--max-strategies` tiers in Task 10. ✔
- **§4 cited families** → Task 4 (6 new, cited) atop existing 15. ✔
- **§6 pipeline** → Tasks 6-8 + Task 10 (data audit, tiers). ✔
- **§7 figures F1-F5** → Task 9 + Task 11 (README embed). ✔
- **§8 gates/verdict** → Task 7 (DSR), Task 6 (PBO), Task 11 (verdict + escalation rule). ✔
- **§9 caution** → Task 6 (float32, chunk-by-universe loop), Task 10 (tiers). ✔
- **Type consistency:** `ZooStrategySpec` fields → `metrics` columns → figures; `rolling_vol(estimator=...)` strings match `GridConfig.vol_estimators`; `build_signal` composes Task1-4 outputs. ✔
- **Placeholder scan:** none; every code step is complete. Task 9/10 reference reading two existing scripts for the styling/fetch pattern (named, not vague). ✔
- **Scope:** single-asset only; cross-sectional deferred to v2 (spec §2). ✔
```
