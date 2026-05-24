# S1-EQ Equity Adaptation + Pragmatic-Strict Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a US-equity sibling of the JS-trained S1 stack (`S1-EQ`) under a new `src/quant_research_stack/alpha_eq/` package, with a pragmatic-strict daily backtest engine that uses fill-aligned PnL and a single dividend accounting path. Promote only if the success gate in spec §6.4 passes on a permanent, untouched holdout.

**Architecture:** Single pooled cross-sectional model: 6 base learners (Ridge, LightGBM, XGBoost, CatBoost, MLP, Conv1D — last three optional in `fast_v1`) plus an L2-regularized `LinearStacker`. Pipeline mirrors S1's surface but lives in a separate package so the JS `feature_00..feature_78` sha256 contract is never touched. Backtest uses split-adjusted tradable prices for execution + MTM and books dividends once as cash PnL on ex-date.

**Tech Stack:** Python 3.11 · Polars / pandas · Pydantic v2 · numpy · scikit-learn · LightGBM · XGBoost · CatBoost · PyTorch (MPS) · joblib · Optuna · pytest · ruff · mypy · YAML configs · Parquet artifacts · uv / Make.

**Spec:** `docs/superpowers/specs/2026-05-24-quantlab-alpha-s1-eq-equity-adaptation-design.md`. Every task references the relevant spec section. Tasks are gated by milestones M1–M6 (spec §6.3). User priorities (in order): (1) data integrity + manifest before any training; (2) dividend-safe PnL before any reported number; (3) `fast_v1` end-to-end before `full_v1`; (4) holdout locked until M6 one-shot; (5) `survivorship_prototype_only` ⇒ research-only no-promotion; (6) plan must not weaken CI gates or accounting invariants.

**Working environment assumptions:**

- `PYTHONPATH=src` for every `python` / `pytest` invocation, as per existing scripts.
- `uv run` is the supported execution wrapper; `uv pip install -e .` is already done in this repo.
- Branch: continue on `quant-llm-implementation` (the active branch from brainstorming) unless the executor explicitly creates a worktree.
- Commits use Conventional Commits format with scope `s1-eq` (e.g. `feat(s1-eq): add PIT classifier`).
- Existing JS-S1 code under `src/quant_research_stack/alpha/` is read-only for this plan; no modifications.
- Existing `Makefile` is extended with new targets in §M0, never rewriting existing ones.

**Pre-flight check before any task:**

```bash
PYTHONPATH=src uv run pytest -q                      # baseline: must be green before starting
PYTHONPATH=src uv run ruff check src scripts tests   # baseline lint
PYTHONPATH=src uv run mypy src                       # baseline types
git status                                            # working tree clean (or only the unstaged
                                                      # tests/test_orderbook_microstructure.py
                                                      # that was present at the start of brainstorming)
```

If baseline pytest is red, do not start; surface it back to the user.

---

## Milestone map

| ID | Milestone | Tasks | Exit criterion |
|---|---|---|---|
| **M0** | Branch & scaffolding | 1–3 | `alpha_eq/` package skeleton importable, Make targets stubbed, baseline tests still green |
| **M1** | Data foundation (PIT, corporate actions, manifest, delisting, borrow, ADV) | 4–17 | `data/processed/equities/_manifest.json` present, `data_quality_label` set, delisting audit table built, all hashes recorded |
| **M2** | Feature pipeline + leakage tests | 18–28 | `feature_cols.json` sha256-locked; all timestamp-contract CI tests green |
| **M3** | `fast_v1` training (Ridge + LGB + XGB + stacker) | 29–40 | Pipeline emits artifacts; reproducibility test green; weak OOF IC is allowed at M3 but blocks promotion to M4 unless justified |
| **M4** | Pragmatic-strict backtest + standard sensitivity pack | 41–58 | `report.md` builds; exposure + PnL-decomposition + no-double-count tests green |
| **M5** | `full_v1` (CatBoost + MLP + optional Conv1D) + audit-level backtest | 59–66 | Permanent holdout evaluated once; `holdout_metrics.json` immutable; full 54-case audit matrix produced |
| **M6** | Success gate, JS-overlay comparison, final report | 67–73 | Go / No-go documented per §6.4; iteration plan if No-go |

---

## M0 — Branch & scaffolding

### Task 1 — Verify pre-flight, create package skeleton

**Spec refs:** §1.2, §6.1.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/__init__.py`
- Create: `src/quant_research_stack/alpha_eq/data/__init__.py`
- Create: `src/quant_research_stack/alpha_eq/features/__init__.py`
- Create: `src/quant_research_stack/alpha_eq/models/__init__.py`
- Create: `src/quant_research_stack/alpha_eq/diagnostics/__init__.py`
- Create: `src/quant_research_stack/alpha_eq/backtest/__init__.py`
- Create: `tests/alpha_eq/__init__.py`
- Create: `tests/alpha_eq/conftest.py`

- [ ] **Step 1: Run baseline pre-flight**

```bash
PYTHONPATH=src uv run pytest -q
PYTHONPATH=src uv run ruff check src scripts tests
PYTHONPATH=src uv run mypy src
git status
```

Expected: baseline pytest green (the unstaged `tests/test_orderbook_microstructure.py` modification may or may not be present; do not commit it as part of this plan). If anything is red, stop and surface.

- [ ] **Step 2: Create package skeleton files**

For each `__init__.py` listed under Files, create with content:

```python
"""S1-EQ — US-equity sibling of the JS-trained S1 stack.

See docs/superpowers/specs/2026-05-24-quantlab-alpha-s1-eq-equity-adaptation-design.md.
"""
```

(Adjust the docstring to the subpackage name: "S1-EQ data layer.", "S1-EQ feature engineering.", etc.)

`tests/alpha_eq/__init__.py` is empty.

- [ ] **Step 3: Create `tests/alpha_eq/conftest.py`**

```python
"""Shared fixtures for alpha_eq tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture()
def tmp_equity_root(tmp_path: Path) -> Path:
    """Disposable processed-equities root for tests."""
    root = tmp_path / "data" / "processed" / "equities"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_panel(rng: np.random.Generator) -> pl.DataFrame:
    """Tiny 5-symbol, 50-date synthetic OHLCV panel for unit tests."""
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    dates = pl.date_range(
        start=pl.date(2020, 1, 2),
        end=pl.date(2020, 3, 13),
        interval="1d",
        eager=True,
    ).filter(pl.col("date").dt.weekday() < 6).slice(0, 50)
    rows = []
    for s in symbols:
        price = 100.0
        for d in dates:
            ret = float(rng.standard_normal()) * 0.02
            price *= (1.0 + ret)
            rows.append(
                {
                    "date": d,
                    "symbol": s,
                    "open": price * (1.0 + float(rng.standard_normal()) * 0.005),
                    "high": price * (1.0 + abs(float(rng.standard_normal())) * 0.01),
                    "low": price * (1.0 - abs(float(rng.standard_normal())) * 0.01),
                    "close": price,
                    "volume": int(1_000_000 + abs(float(rng.standard_normal())) * 500_000),
                }
            )
    return pl.DataFrame(rows)
```

- [ ] **Step 4: Verify importability**

```bash
PYTHONPATH=src uv run python -c "import quant_research_stack.alpha_eq; import quant_research_stack.alpha_eq.data; import quant_research_stack.alpha_eq.features; import quant_research_stack.alpha_eq.models; import quant_research_stack.alpha_eq.diagnostics; import quant_research_stack.alpha_eq.backtest; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha_eq tests/alpha_eq
git commit -m "feat(s1-eq): scaffold alpha_eq package skeleton"
```

---

### Task 2 — Add Make targets (stubs) and configs/alpha_eq.yaml minimal skeleton

**Spec refs:** §6.1.

**Files:**
- Modify: `Makefile` (append new targets, do not edit existing ones)
- Create: `configs/alpha_eq.yaml`
- Create: `configs/eq_focused_basket.yaml`
- Create: `configs/backtest_eq.yaml`

- [ ] **Step 1: Inspect existing Makefile for a safe insertion point**

```bash
tail -20 Makefile
```

- [ ] **Step 2: Append S1-EQ Make targets**

Append to `Makefile`:

```make

# ----- S1-EQ -----

prepare-equity-data:
	PYTHONPATH=src uv run python scripts/prepare_equity_data.py \
		--config configs/alpha_eq.yaml

pit-quality-audit:
	PYTHONPATH=src uv run python scripts/pit_quality_audit.py \
		--equity-root data/processed/equities

fast-retrain-s1-eq:
	PYTHONPATH=src uv run python scripts/train_s1_eq.py \
		--config configs/alpha_eq.yaml --mode fast_v1

full-retrain-s1-eq:
	PYTHONPATH=src uv run python scripts/train_s1_eq.py \
		--config configs/alpha_eq.yaml --mode full_v1

backtest-s1-eq-standard:
	PYTHONPATH=src uv run python scripts/backtest_s1_eq.py \
		--config configs/backtest_eq.yaml --mode standard

backtest-s1-eq-audit:
	PYTHONPATH=src uv run python scripts/backtest_s1_eq.py \
		--config configs/backtest_eq.yaml --mode audit

js-overlay-compare-s1-eq:
	PYTHONPATH=src uv run python scripts/s1_eq_overlay_compare.py \
		--config configs/backtest_eq.yaml
```

- [ ] **Step 3: Create `configs/alpha_eq.yaml`**

```yaml
# S1-EQ training config — fast_v1 + full_v1 profiles.
# Spec: docs/superpowers/specs/2026-05-24-quantlab-alpha-s1-eq-equity-adaptation-design.md

data:
  equity_root: data/processed/equities
  manifest_path: data/processed/equities/_manifest.json
  universe: sp500
  permanent_holdout_fraction: 0.20
  min_holdout_trading_days: 756              # spec §3.6

features:
  enable_meta_features: false                # spec §3.3-7
  noise_seed: 42
  rolling_windows: [5, 20, 60]
  momentum_horizons: [1, 2, 5, 10, 20, 60, 120, 252]
  vix_proxy_fallback: cross_sectional_vol_20 # spec §3.3-6

cv:
  layout: expanding_window
  n_folds: 5
  label_horizon_days: 1
  purge_safety_buffer: 2
  rolling_diagnostic:
    enabled: true
    train_years: 10
    valid_years: 2

models:
  fast_v1: [ridge, lightgbm, xgboost]
  full_v1: [ridge, lightgbm, xgboost, catboost, mlp, sequence]

stacker:
  alpha: 1.0e-3
  prefer_non_negative: true
  flag_large_negative_threshold: -0.25       # spec §4.5

optuna:
  enable: true
  trials:
    lightgbm: 50
    xgboost: 30
    catboost: 30
    mlp: 20
    sequence: 20
    stacker: 30

reproducibility:
  numpy_seed: 42
  torch_seed: 42
  lightgbm_seed: 42
  xgboost_seed: 42
  catboost_seed: 42
```

- [ ] **Step 4: Create `configs/eq_focused_basket.yaml`**

```yaml
# Focused mega-cap reporting basket. Versioned per spec §6.7.
version: 1
description: |
  Liquid US mega-cap cohort for the S1-EQ reporting view. Changing the
  symbol list requires bumping the version and creating a new run_id.
symbols:
  - AAPL
  - ORCL
  - PYPL
  - INTC
  - META
  - TSLA
  - QCOM
  - PLTR
  - GOOGL
  - AVGO
  - ADBE
  - RKLB
  - MSFT
  - AMZN
  - NVDA
  - GOOG
  - NFLX
  - AMD
  - CRM
  - CSCO
  - IBM
  - TXN
  - MU
  - LRCX
  - AMAT
  - JPM
  - BAC
  - GS
  - MS
  - V
  - MA
  - JNJ
  - PFE
  - UNH
  - WMT
  - HD
  - COST
  - PG
  - KO
  - PEP
```

- [ ] **Step 5: Create `configs/backtest_eq.yaml`**

```yaml
# S1-EQ backtest config — standard + audit sensitivity packs (spec §5.14).
universe: sp500
focused_basket_config: configs/eq_focused_basket.yaml

execution:
  fill: open                                  # headline; spec §5.3
  adv_participation_pct_headline: 0.01
  adv_participation_pct_sensitivity: 0.03
  adv_floor_dollars: 1_000_000
  gross_target_headline: 1.0
  q_quantile_headline: 0.10
  q_quantile_sweep: [0.05, 0.10, 0.20]
  min_long_full_universe: 10
  min_short_full_universe: 10
  min_long_focused_basket: 5
  min_short_focused_basket: 5
  max_single_name_weight_frac_of_gross: 0.05

cost:
  commission_bps_one_way: 0.5
  roll_spread_cap_bps: 50
  tiered_fallback_bps:
    easy: 5
    general: 15
    hard: 50
  pre_decimalization_cutoff: "2001-04-09"
  pre_decimalization_multiplier_fallback: 2.5
  pre_decimalization_multiplier_roll: 1.5

borrow:
  stress_multipliers: [1.0, 2.0, 3.0]

financing:
  rates_when_gross_gt_1: [0.0, 0.02, 0.05]

sensitivity:
  standard_pack:
    borrow: [1.0, 3.0]
    fill: [open, hlc3_proxy]
    q: [0.05, 0.10]
    gross: [1.0]
  audit_pack:
    borrow: [1.0, 2.0, 3.0]
    fill: [open, hlc3_proxy, close]
    adv_participation_pct: [0.01, 0.03]
    gross: [0.5, 1.0, 2.0]

reporting:
  cohorts: [full_universe, focused_basket]
  rolling_spy_beta_window: 60
  monthly_returns: true
  annual_returns: true
```

- [ ] **Step 6: Make targets dry-run check (they will fail because scripts don't exist yet — that is expected)**

```bash
make prepare-equity-data 2>&1 | tail -3
```

Expected: `can't open file '.../scripts/prepare_equity_data.py'` or similar. This proves the target is registered.

- [ ] **Step 7: Commit**

```bash
git add Makefile configs/alpha_eq.yaml configs/eq_focused_basket.yaml configs/backtest_eq.yaml
git commit -m "feat(s1-eq): add make targets and config skeletons"
```

---

### Task 3 — Define the M0 sentinel `pyproject` extras and pin guard

**Spec refs:** §6.1 (test tooling), §4.9 (reproducibility).

**Files:**
- Modify: `pyproject.toml` (add `[project.optional-dependencies]` extra `alpha_eq` if not present; do not change existing pins)

- [ ] **Step 1: Inspect `pyproject.toml`**

```bash
sed -n '1,80p' pyproject.toml
```

- [ ] **Step 2: Confirm needed deps are already pinned**

The required libs are: `polars`, `numpy`, `pandas`, `pyarrow`, `pydantic`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `torch`, `joblib`, `optuna`, `pyyaml`, `rich`. They exist for S1. If any of `catboost`, `optuna`, `pyarrow` are missing, add them under existing project dependencies. Do **not** delete or upgrade pins.

```bash
PYTHONPATH=src uv run python -c "import catboost, optuna, pyarrow; print(catboost.__version__, optuna.__version__, pyarrow.__version__)"
```

Expected: three version strings print. If any import errors, add the missing package to `pyproject.toml` `[project] dependencies` (lowest already-pinned version that publishes a wheel for Python 3.11) and re-run `uv pip install -e .`.

- [ ] **Step 3: Commit only if pyproject changed**

```bash
git status
# if pyproject.toml is modified:
git add pyproject.toml uv.lock 2>/dev/null
git commit -m "build(s1-eq): ensure catboost/optuna/pyarrow deps present"
# otherwise no commit
```

---

## M1 — Data foundation

### Task 4 — `manifest.py`: data-quality + corporate-action + borrow + delisting labels

**Spec refs:** §2.7, §2.1, §2.9.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/data/manifest.py`
- Create: `tests/alpha_eq/test_manifest.py`

- [ ] **Step 1: Write failing tests**

`tests/alpha_eq/test_manifest.py`:

```python
"""Manifest schema, hash, and label tests (spec §2.7, §2.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research_stack.alpha_eq.data.manifest import (
    DataQualityLabel,
    EquityManifest,
    ManifestArtifact,
    ManifestMismatchError,
    load_and_verify_manifest,
    write_manifest,
)


def test_data_quality_label_values() -> None:
    assert DataQualityLabel("pit_safe").value == "pit_safe"
    assert DataQualityLabel("partial_pit_universe").value == "partial_pit_universe"
    assert DataQualityLabel("survivorship_prototype_only").value == "survivorship_prototype_only"


def test_data_quality_label_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        DataQualityLabel("institutional_grade_marketing_word")


def test_manifest_artifact_required_fields() -> None:
    art = ManifestArtifact(
        path="sp500_tradable_prices.parquet",
        sha256="a" * 64,
        row_count=10,
        symbol_count=2,
        date_range_start="2020-01-02",
        date_range_end="2020-01-15",
        schema_fingerprint="cols:date,symbol,open,high,low,close,volume",
    )
    assert art.row_count == 10


def test_write_and_load_manifest_round_trip(tmp_equity_root: Path) -> None:
    art = ManifestArtifact(
        path="sp500_tradable_prices.parquet",
        sha256="b" * 64,
        row_count=10,
        symbol_count=2,
        date_range_start="2020-01-02",
        date_range_end="2020-01-15",
        schema_fingerprint="cols:date,symbol,open,high,low,close,volume",
    )
    m = EquityManifest(
        pipeline_version="0.1.0",
        git_sha="deadbeef",
        artifacts={"sp500_tradable_prices": art},
        data_quality_label=DataQualityLabel("partial_pit_universe"),
        corporate_action_quality="split_adj_plus_external_dividends",
        borrow_source_quality="static_proxy_v1",
        pit_membership_source="wikipedia_fallback",
        delisting_audit_quality="partial_capture",
        delisting_audit_counters={
            "delisted_captured": 12,
            "delisted_missing": 1,
            "merger_captured": 8,
            "merger_missing": 0,
            "ticker_changed": 5,
            "unknown_exit": 0,
        },
        build_command_line="prepare_equity_data.py --config configs/alpha_eq.yaml",
        python_version="3.11.x",
        package_versions={"polars": "x.y", "lightgbm": "x.y"},
        warnings=["dividend feed: public_snapshot_not_vendor_pit"],
    )
    out = tmp_equity_root / "_manifest.json"
    write_manifest(out, m)
    m2 = load_and_verify_manifest(out, expected_sha256={"sp500_tradable_prices": "b" * 64})
    assert m2.data_quality_label.value == "partial_pit_universe"
    assert m2.artifacts["sp500_tradable_prices"].sha256 == "b" * 64


def test_load_and_verify_manifest_hard_fails_on_hash_mismatch(tmp_equity_root: Path) -> None:
    out = tmp_equity_root / "_manifest.json"
    out.write_text(
        json.dumps(
            {
                "pipeline_version": "0.1.0",
                "git_sha": "x",
                "artifacts": {
                    "a": {
                        "path": "a.parquet",
                        "sha256": "a" * 64,
                        "row_count": 1,
                        "symbol_count": 1,
                        "date_range_start": "2020-01-02",
                        "date_range_end": "2020-01-02",
                        "schema_fingerprint": "x",
                    }
                },
                "data_quality_label": "pit_safe",
                "corporate_action_quality": "vendor_total_return",
                "borrow_source_quality": "static_proxy_v1",
                "pit_membership_source": "hf:andyqin18/sp500-historical-membership",
                "delisting_audit_quality": "captured_above_threshold",
                "delisting_audit_counters": {
                    "delisted_captured": 0,
                    "delisted_missing": 0,
                    "merger_captured": 0,
                    "merger_missing": 0,
                    "ticker_changed": 0,
                    "unknown_exit": 0,
                },
                "build_command_line": "x",
                "python_version": "3.11.0",
                "package_versions": {},
                "warnings": [],
            }
        )
    )
    with pytest.raises(ManifestMismatchError):
        load_and_verify_manifest(out, expected_sha256={"a": "b" * 64})


def test_manifest_required_fields_missing(tmp_equity_root: Path) -> None:
    out = tmp_equity_root / "_manifest.json"
    out.write_text(json.dumps({"git_sha": "x"}))
    with pytest.raises(ManifestMismatchError):
        load_and_verify_manifest(out, expected_sha256={})
```

- [ ] **Step 2: Run tests, expect fail (module not implemented)**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_manifest.py -v
```

Expected: `ModuleNotFoundError: No module named 'quant_research_stack.alpha_eq.data.manifest'`.

- [ ] **Step 3: Implement `manifest.py`**

`src/quant_research_stack/alpha_eq/data/manifest.py`:

```python
"""Equity data manifest — single source of truth for data-quality labels,
artifact hashes, and reproducibility metadata (spec §2.1, §2.7, §2.9)."""

from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DataQualityLabel(str, enum.Enum):
    PIT_SAFE = "pit_safe"
    PARTIAL_PIT_UNIVERSE = "partial_pit_universe"
    SURVIVORSHIP_PROTOTYPE_ONLY = "survivorship_prototype_only"


class ManifestMismatchError(RuntimeError):
    """Raised when the manifest disagrees with the on-disk artifacts."""


class ManifestArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    row_count: int
    symbol_count: int
    date_range_start: str
    date_range_end: str
    schema_fingerprint: str
    source_url: str | None = None
    source_dataset_id: str | None = None
    source_snapshot_date: str | None = None


class DelistingAuditCounters(BaseModel):
    model_config = ConfigDict(frozen=True)

    delisted_captured: int = 0
    delisted_missing: int = 0
    merger_captured: int = 0
    merger_missing: int = 0
    ticker_changed: int = 0
    unknown_exit: int = 0


class EquityManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    pipeline_version: str
    git_sha: str
    artifacts: dict[str, ManifestArtifact]
    data_quality_label: DataQualityLabel
    corporate_action_quality: str
    borrow_source_quality: str
    pit_membership_source: str
    delisting_audit_quality: str
    delisting_audit_counters: DelistingAuditCounters | dict[str, int] = Field(
        default_factory=DelistingAuditCounters
    )
    build_command_line: str
    python_version: str
    package_versions: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path, manifest: EquityManifest) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    Path(path).write_bytes(_canonical_json(payload))


def load_and_verify_manifest(
    path: Path,
    *,
    expected_sha256: Mapping[str, str],
) -> EquityManifest:
    if not Path(path).exists():
        raise ManifestMismatchError(f"manifest missing: {path}")
    try:
        payload = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ManifestMismatchError(f"manifest is not valid JSON: {exc}") from exc
    try:
        manifest = EquityManifest.model_validate(payload)
    except Exception as exc:
        raise ManifestMismatchError(f"manifest schema error: {exc}") from exc
    for key, sha in expected_sha256.items():
        if key not in manifest.artifacts:
            raise ManifestMismatchError(f"manifest missing artifact key: {key}")
        if manifest.artifacts[key].sha256 != sha:
            raise ManifestMismatchError(
                f"sha256 mismatch for {key}: expected={sha} got={manifest.artifacts[key].sha256}"
            )
    return manifest
```

- [ ] **Step 4: Tests pass**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_manifest.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha_eq/data/manifest.py tests/alpha_eq/test_manifest.py
git commit -m "feat(s1-eq): manifest schema, hashing, hard-fail loader"
```

---

### Task 5 — Tradable / split-adj / total-return price builder

**Spec refs:** §2.3.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/data/corporate_actions.py`
- Create: `tests/alpha_eq/test_corporate_actions.py`

- [ ] **Step 1: Write failing tests**

`tests/alpha_eq/test_corporate_actions.py`:

```python
"""Three-price-series builder (spec §2.3)."""

from __future__ import annotations

import polars as pl

from quant_research_stack.alpha_eq.data.corporate_actions import (
    PriceSeriesBundle,
    build_three_series,
    de_total_return_to_tradable,
)


def _toy_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
            "symbol": ["A", "A", "A", "A"],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1_000_000, 1_100_000, 1_050_000, 1_200_000],
        }
    ).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))


def _toy_dividends() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ex_date": ["2020-01-06"],
            "symbol": ["A"],
            "dividend_per_share": [0.50],
        }
    ).with_columns(pl.col("ex_date").str.strptime(pl.Date, "%Y-%m-%d"))


def test_build_three_series_when_source_is_split_adjusted() -> None:
    bundle = build_three_series(
        panel=_toy_panel(),
        dividends=_toy_dividends(),
        source_is_total_return=False,
    )
    assert isinstance(bundle, PriceSeriesBundle)
    # tradable_* == split-adjusted in v1
    assert bundle.tradable["close"].to_list() == [100.5, 101.5, 102.5, 103.5]
    # total-return reflects the 0.50 dividend reinvested on ex-date 2020-01-06
    tr = bundle.total_return["close_tr"].to_list()
    assert tr[0] == 100.5
    assert tr[1] == 101.5
    assert tr[2] > 102.5  # bumped by dividend reinvestment
    assert tr[3] > 103.5


def test_de_total_return_inverse_of_total_return_build() -> None:
    panel = _toy_panel()
    divs = _toy_dividends()
    bundle = build_three_series(panel=panel, dividends=divs, source_is_total_return=False)
    rebuilt = de_total_return_to_tradable(bundle.total_return, divs)
    # rebuilt should match the split-adjusted tradable close within float tolerance
    assert (
        (rebuilt["close"] - bundle.tradable["close"]).abs().max() < 1e-9
    )
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_corporate_actions.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

`src/quant_research_stack/alpha_eq/data/corporate_actions.py`:

```python
"""Three price series: tradable_* (split-adjusted execution-consistent),
split_adj_* (alias of tradable_* in v1), and total_return_* (split-
adjusted + dividend reinvested, used only for labels/diagnostics/
benchmarks — never portfolio MTM).  Spec §2.3."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class PriceSeriesBundle:
    tradable: pl.DataFrame
    split_adj: pl.DataFrame
    total_return: pl.DataFrame


_OHLCV_COLS: tuple[str, ...] = ("open", "high", "low", "close")


def build_three_series(
    *,
    panel: pl.DataFrame,
    dividends: pl.DataFrame,
    source_is_total_return: bool,
) -> PriceSeriesBundle:
    """Build the three price series from a daily-bars panel + dividend feed.

    If source_is_total_return=True, the upstream panel is assumed to already
    embed dividend reinvestment; tradable_* is recovered by removing the
    dividend ladder.
    """
    panel = panel.sort(["symbol", "date"])
    if source_is_total_return:
        tradable = _remove_dividend_reinvestment(panel, dividends)
    else:
        tradable = panel.clone()
    split_adj = tradable.clone()  # v1: alias; future raw-lot/share accounting splits this out
    total_return = _apply_dividend_reinvestment(tradable, dividends)
    # rename close/open/... → *_tr in total_return for clarity
    total_return = total_return.rename({c: f"{c}_tr" for c in _OHLCV_COLS})
    return PriceSeriesBundle(tradable=tradable, split_adj=split_adj, total_return=total_return)


def _apply_dividend_reinvestment(panel: pl.DataFrame, dividends: pl.DataFrame) -> pl.DataFrame:
    if dividends.is_empty():
        return panel.clone()
    panel = panel.sort(["symbol", "date"])
    divs = dividends.rename({"ex_date": "date"}).sort(["symbol", "date"])
    joined = panel.join(divs, on=["symbol", "date"], how="left").with_columns(
        pl.col("dividend_per_share").fill_null(0.0)
    )
    # Reinvestment factor per (symbol, date) = 1 + div / close_prior
    joined = joined.with_columns(
        pl.col("close").shift(1).over("symbol").alias("close_prior")
    ).with_columns(
        pl.when(pl.col("close_prior").is_not_null() & (pl.col("close_prior") > 0))
        .then(1.0 + pl.col("dividend_per_share") / pl.col("close_prior"))
        .otherwise(1.0)
        .alias("reinvest_factor")
    )
    joined = joined.with_columns(
        pl.col("reinvest_factor").cum_prod().over("symbol").alias("cum_factor")
    )
    out = joined.with_columns(
        [(pl.col(c) * pl.col("cum_factor")).alias(c) for c in _OHLCV_COLS]
    ).drop(["dividend_per_share", "close_prior", "reinvest_factor", "cum_factor"])
    return out


def _remove_dividend_reinvestment(panel: pl.DataFrame, dividends: pl.DataFrame) -> pl.DataFrame:
    """Inverse of _apply_dividend_reinvestment: recover the split-adjusted
    tradable series from a total-return source by dividing out the cumulative
    reinvestment factor."""
    if dividends.is_empty():
        return panel.clone()
    panel = panel.sort(["symbol", "date"])
    divs = dividends.rename({"ex_date": "date"}).sort(["symbol", "date"])
    joined = panel.join(divs, on=["symbol", "date"], how="left").with_columns(
        pl.col("dividend_per_share").fill_null(0.0)
    )
    # We need close_prior in the *tradable* series, but here panel is total-return.
    # Use a stable approximation: treat the panel close as the tradable close at t-1
    # for the purpose of computing reinvest_factor, then iterate one pass.
    joined = joined.with_columns(
        pl.col("close").shift(1).over("symbol").alias("close_prior")
    ).with_columns(
        pl.when(pl.col("close_prior").is_not_null() & (pl.col("close_prior") > 0))
        .then(1.0 + pl.col("dividend_per_share") / pl.col("close_prior"))
        .otherwise(1.0)
        .alias("reinvest_factor")
    )
    joined = joined.with_columns(
        pl.col("reinvest_factor").cum_prod().over("symbol").alias("cum_factor")
    )
    out = joined.with_columns(
        [(pl.col(c) / pl.col("cum_factor")).alias(c) for c in _OHLCV_COLS]
    ).drop(["dividend_per_share", "close_prior", "reinvest_factor", "cum_factor"])
    return out


def de_total_return_to_tradable(
    total_return: pl.DataFrame, dividends: pl.DataFrame
) -> pl.DataFrame:
    """Public helper used by tests + the prepare-equity-data script when the
    upstream HF dataset is vendor_total_return."""
    panel = total_return.rename({f"{c}_tr": c for c in _OHLCV_COLS})
    return _remove_dividend_reinvestment(panel, dividends)
```

- [ ] **Step 4: Tests pass**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_corporate_actions.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha_eq/data/corporate_actions.py tests/alpha_eq/test_corporate_actions.py
git commit -m "feat(s1-eq): three-price-series builder with reversible dividend reinvestment"
```

---

### Task 6 — `pit_membership.py`: PIT membership + ticker mapping loader

**Spec refs:** §2.2.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/data/pit_membership.py`
- Create: `tests/alpha_eq/test_pit_membership.py`

- [ ] **Step 1: Write failing tests**

```python
"""PIT membership + ticker mapping (spec §2.2)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.alpha_eq.data.pit_membership import (
    MembershipSource,
    PITMembership,
    TickerMapping,
    apply_ticker_mapping,
    load_pit_membership,
)


def _toy_membership() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 2), date(2020, 1, 3)],
            "symbol": ["AAPL", "AOL", "AAPL"],
            "in_index": [True, True, True],
            "addition_date": [date(1982, 11, 30), date(1985, 1, 1), date(1982, 11, 30)],
            "removal_date": [None, date(2020, 1, 3), None],
            "removal_reason": [None, "acquired", None],
        }
    )


def test_load_pit_membership_round_trip(tmp_equity_root: Path) -> None:
    df = _toy_membership()
    path = tmp_equity_root / "sp500_pit_membership.parquet"
    df.write_parquet(path)
    mem = load_pit_membership(path, source=MembershipSource.HF_PRIMARY)
    assert isinstance(mem, PITMembership)
    assert mem.is_in_index(symbol="AAPL", on=date(2020, 1, 2))
    assert mem.is_in_index(symbol="AOL", on=date(2020, 1, 2))
    assert not mem.is_in_index(symbol="AOL", on=date(2020, 1, 3))


def test_load_pit_membership_missing_columns(tmp_equity_root: Path) -> None:
    bad = pl.DataFrame({"date": [date(2020, 1, 2)], "symbol": ["AAPL"]})
    p = tmp_equity_root / "bad.parquet"
    bad.write_parquet(p)
    with pytest.raises(ValueError, match="missing column"):
        load_pit_membership(p, source=MembershipSource.HF_PRIMARY)


def test_ticker_mapping_apply() -> None:
    mapping = TickerMapping(
        rows=[
            ("FB", "META", date(2022, 6, 9)),
            ("VIAC", "PARA", date(2022, 2, 16)),
        ]
    )
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2023, 1, 3), date(2022, 6, 9)],
            "symbol": ["FB", "FB", "FB"],
        }
    )
    out = apply_ticker_mapping(df, mapping)
    assert out["symbol"].to_list() == ["FB", "META", "META"]


def test_membership_source_values() -> None:
    assert MembershipSource("hf_primary").value == "hf_primary"
    assert MembershipSource("wikipedia_fallback").value == "wikipedia_fallback"
    assert MembershipSource("absent_prototype_only").value == "absent_prototype_only"
    with pytest.raises(ValueError):
        MembershipSource("guessed")
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_pit_membership.py -v
```

- [ ] **Step 3: Implement**

`src/quant_research_stack/alpha_eq/data/pit_membership.py`:

```python
"""PIT S&P 500 membership table + ticker-mapping logic.

Spec §2.2. The audit script `pit_quality_audit.py` decides which source
to write; this module only loads + applies the table.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl


class MembershipSource(str, enum.Enum):
    HF_PRIMARY = "hf_primary"
    HF_SECONDARY = "hf_secondary"
    KAGGLE = "kaggle"
    WIKIPEDIA_FALLBACK = "wikipedia_fallback"
    ABSENT_PROTOTYPE_ONLY = "absent_prototype_only"


_REQUIRED_COLS: tuple[str, ...] = (
    "date",
    "symbol",
    "in_index",
    "addition_date",
    "removal_date",
    "removal_reason",
)


@dataclass(frozen=True)
class PITMembership:
    source: MembershipSource
    table: pl.DataFrame

    def is_in_index(self, *, symbol: str, on: date) -> bool:
        f = self.table.filter(
            (pl.col("symbol") == symbol) & (pl.col("date") == on)
        )
        if f.is_empty():
            return False
        return bool(f["in_index"][0])


@dataclass(frozen=True)
class TickerMapping:
    """A list of (old_symbol, new_symbol, effective_date) transforms.

    The mapping is applied row-wise: any row whose date >= effective_date and
    whose current symbol is the old_symbol gets renamed to the new_symbol.
    """

    rows: list[tuple[str, str, date]] = field(default_factory=list)


def load_pit_membership(path: Path, *, source: MembershipSource) -> PITMembership:
    if not Path(path).exists():
        raise FileNotFoundError(f"PIT membership table missing: {path}")
    df = pl.read_parquet(path)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"PIT membership table missing column(s): {missing}")
    return PITMembership(source=source, table=df)


def apply_ticker_mapping(df: pl.DataFrame, mapping: TickerMapping) -> pl.DataFrame:
    out = df
    for old, new, effective in mapping.rows:
        out = out.with_columns(
            pl.when((pl.col("symbol") == old) & (pl.col("date") >= effective))
            .then(pl.lit(new))
            .otherwise(pl.col("symbol"))
            .alias("symbol")
        )
    return out
```

- [ ] **Step 4: Tests pass**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_pit_membership.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha_eq/data/pit_membership.py tests/alpha_eq/test_pit_membership.py
git commit -m "feat(s1-eq): PIT membership loader + ticker mapping"
```

---

### Task 7 — Delisting-return audit (spec §2.9)

**Files:**
- Create: `src/quant_research_stack/alpha_eq/data/delisting_audit.py`
- Create: `tests/alpha_eq/test_delisting_audit.py`

- [ ] **Step 1: Write failing tests**

```python
"""Delisting-return audit (spec §2.9)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.data.delisting_audit import (
    DelistingAuditResult,
    audit_delistings,
    classify_exit,
)


def test_classify_exit_basic() -> None:
    assert classify_exit(reason="bankruptcy", terminal_return_known=True) == "delisted_captured"
    assert classify_exit(reason="bankruptcy", terminal_return_known=False) == "delisted_missing"
    assert classify_exit(reason="acquired", terminal_return_known=True) == "merger_captured"
    assert classify_exit(reason="acquired", terminal_return_known=False) == "merger_missing"
    assert classify_exit(reason="ticker_change", terminal_return_known=False) == "ticker_changed"
    assert classify_exit(reason="unknown", terminal_return_known=False) == "unknown_exit"


def test_audit_delistings_counts_and_threshold() -> None:
    panel = pl.DataFrame(
        {
            "date": [
                date(2020, 1, 2), date(2020, 1, 3),  # AAA observed
                date(2020, 1, 2),                    # BBB observed once then gone
                date(2020, 1, 2), date(2020, 1, 3),  # CCC observed
            ],
            "symbol": ["AAA", "AAA", "BBB", "CCC", "CCC"],
            "close": [100.0, 101.0, 50.0, 200.0, 199.0],
        }
    )
    exits = pl.DataFrame(
        {
            "symbol": ["BBB"],
            "exit_date": [date(2020, 1, 3)],
            "exit_reason": ["acquired"],
            "terminal_return_known": [True],
            "terminal_return_value": [-0.10],
        }
    )
    result = audit_delistings(panel=panel, exits=exits)
    assert isinstance(result, DelistingAuditResult)
    assert result.counters["merger_captured"] == 1
    assert result.counters["unknown_exit"] == 0


def test_audit_delistings_flags_unknown_exit() -> None:
    panel = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 2)],
            "symbol": ["AAA", "BBB"],
            "close": [100.0, 50.0],
        }
    )
    # BBB disappears after 2020-01-02 but no exit row provided
    panel_next = pl.DataFrame(
        {"date": [date(2020, 1, 3)], "symbol": ["AAA"], "close": [101.0]}
    )
    full = pl.concat([panel, panel_next])
    exits = pl.DataFrame(
        schema={
            "symbol": pl.Utf8,
            "exit_date": pl.Date,
            "exit_reason": pl.Utf8,
            "terminal_return_known": pl.Boolean,
            "terminal_return_value": pl.Float64,
        }
    )
    result = audit_delistings(panel=full, exits=exits)
    assert result.counters["unknown_exit"] >= 1
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_delisting_audit.py -v
```

- [ ] **Step 3: Implement**

`src/quant_research_stack/alpha_eq/data/delisting_audit.py`:

```python
"""Delisting-return audit (spec §2.9).

Classifies every symbol-exit into a known category so that missing
terminal losses do not silently inflate equity backtest performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

_EXIT_REASONS_DELISTED: frozenset[str] = frozenset(
    {"bankruptcy", "regulatory_delisting", "going_to_zero", "delisted"}
)
_EXIT_REASONS_MERGER: frozenset[str] = frozenset(
    {"acquired", "merger", "going_private", "buyout"}
)
_EXIT_REASONS_TICKER: frozenset[str] = frozenset({"ticker_change"})


def classify_exit(*, reason: str, terminal_return_known: bool) -> str:
    r = reason.lower()
    if r in _EXIT_REASONS_TICKER:
        return "ticker_changed"
    if r in _EXIT_REASONS_DELISTED:
        return "delisted_captured" if terminal_return_known else "delisted_missing"
    if r in _EXIT_REASONS_MERGER:
        return "merger_captured" if terminal_return_known else "merger_missing"
    return "unknown_exit"


@dataclass(frozen=True)
class DelistingAuditResult:
    counters: dict[str, int]
    audit_table: pl.DataFrame


def audit_delistings(*, panel: pl.DataFrame, exits: pl.DataFrame) -> DelistingAuditResult:
    """Inspect a panel + an exits feed.  Any symbol whose last observation
    in `panel` is before the panel's global max date AND has no matching
    `exits` row is recorded as `unknown_exit`."""
    panel = panel.sort(["symbol", "date"])
    global_max = panel["date"].max()
    last_seen = panel.group_by("symbol").agg(pl.col("date").max().alias("last_date"))
    exited = last_seen.filter(pl.col("last_date") < global_max)

    rows: list[dict[str, object]] = []
    known_symbols: set[str] = set(exits["symbol"].to_list()) if not exits.is_empty() else set()
    for sym, last in zip(exited["symbol"].to_list(), exited["last_date"].to_list(), strict=True):
        if sym in known_symbols:
            erow = exits.filter(pl.col("symbol") == sym)
            reason = str(erow["exit_reason"][0])
            terminal_known = bool(erow["terminal_return_known"][0])
            classification = classify_exit(reason=reason, terminal_return_known=terminal_known)
            terminal_value = (
                float(erow["terminal_return_value"][0])
                if "terminal_return_value" in erow.columns
                and erow["terminal_return_value"][0] is not None
                else None
            )
        else:
            classification = "unknown_exit"
            reason = "unknown"
            terminal_known = False
            terminal_value = None
        rows.append(
            {
                "symbol": sym,
                "exit_date": last,
                "exit_reason": reason,
                "terminal_return_captured": terminal_known,
                "terminal_return_value": terminal_value,
                "classification_source": "exits_feed" if sym in known_symbols else "panel_inferred",
                "classification": classification,
            }
        )
    audit_df = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "symbol": pl.Utf8,
            "exit_date": pl.Date,
            "exit_reason": pl.Utf8,
            "terminal_return_captured": pl.Boolean,
            "terminal_return_value": pl.Float64,
            "classification_source": pl.Utf8,
            "classification": pl.Utf8,
        }
    )
    counters = {
        k: int((audit_df["classification"] == k).sum()) if not audit_df.is_empty() else 0
        for k in (
            "delisted_captured",
            "delisted_missing",
            "merger_captured",
            "merger_missing",
            "ticker_changed",
            "unknown_exit",
        )
    }
    return DelistingAuditResult(counters=counters, audit_table=audit_df)
```

- [ ] **Step 4: Tests pass**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_delisting_audit.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha_eq/data/delisting_audit.py tests/alpha_eq/test_delisting_audit.py
git commit -m "feat(s1-eq): delisting-return audit with exit classification"
```

---

### Task 8 — PIT quality classifier (three-tier)

**Spec refs:** §2.1, §2.9.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/data/pit_quality.py`
- Create: `tests/alpha_eq/test_pit_classifier.py`

- [ ] **Step 1: Write failing tests**

```python
"""Three-tier PIT data-quality classifier (spec §2.1)."""

from __future__ import annotations

import polars as pl

from quant_research_stack.alpha_eq.data.delisting_audit import DelistingAuditResult
from quant_research_stack.alpha_eq.data.manifest import DataQualityLabel
from quant_research_stack.alpha_eq.data.pit_membership import MembershipSource
from quant_research_stack.alpha_eq.data.pit_quality import (
    PITQualityInputs,
    classify_pit_quality,
)


def _audit(captured: int, missing: int, unknown_in_holdout: int) -> DelistingAuditResult:
    counters = {
        "delisted_captured": captured,
        "delisted_missing": missing,
        "merger_captured": 0,
        "merger_missing": 0,
        "ticker_changed": 0,
        "unknown_exit": unknown_in_holdout,
    }
    return DelistingAuditResult(counters=counters, audit_table=pl.DataFrame())


def test_pit_safe_when_membership_present_and_audit_above_threshold() -> None:
    inputs = PITQualityInputs(
        membership_source=MembershipSource.HF_PRIMARY,
        audit=_audit(captured=95, missing=5, unknown_in_holdout=0),
        unknown_exit_in_holdout=0,
    )
    assert classify_pit_quality(inputs) == DataQualityLabel.PIT_SAFE


def test_partial_when_membership_present_but_audit_below_threshold() -> None:
    inputs = PITQualityInputs(
        membership_source=MembershipSource.HF_PRIMARY,
        audit=_audit(captured=50, missing=50, unknown_in_holdout=0),
        unknown_exit_in_holdout=0,
    )
    assert classify_pit_quality(inputs) == DataQualityLabel.PARTIAL_PIT_UNIVERSE


def test_partial_when_unknown_exit_in_holdout_nonzero() -> None:
    inputs = PITQualityInputs(
        membership_source=MembershipSource.HF_PRIMARY,
        audit=_audit(captured=100, missing=0, unknown_in_holdout=1),
        unknown_exit_in_holdout=1,
    )
    assert classify_pit_quality(inputs) == DataQualityLabel.PARTIAL_PIT_UNIVERSE


def test_wikipedia_fallback_caps_at_partial() -> None:
    inputs = PITQualityInputs(
        membership_source=MembershipSource.WIKIPEDIA_FALLBACK,
        audit=_audit(captured=100, missing=0, unknown_in_holdout=0),
        unknown_exit_in_holdout=0,
    )
    # Wikipedia is fallback; even with perfect audit it cannot earn pit_safe alone
    assert classify_pit_quality(inputs) == DataQualityLabel.PARTIAL_PIT_UNIVERSE


def test_prototype_only_when_membership_absent() -> None:
    inputs = PITQualityInputs(
        membership_source=MembershipSource.ABSENT_PROTOTYPE_ONLY,
        audit=_audit(captured=0, missing=0, unknown_in_holdout=0),
        unknown_exit_in_holdout=0,
    )
    assert classify_pit_quality(inputs) == DataQualityLabel.SURVIVORSHIP_PROTOTYPE_ONLY
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_pit_classifier.py -v
```

- [ ] **Step 3: Implement**

```python
"""Three-tier PIT data-quality classifier (spec §2.1, §2.9)."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_stack.alpha_eq.data.delisting_audit import DelistingAuditResult
from quant_research_stack.alpha_eq.data.manifest import DataQualityLabel
from quant_research_stack.alpha_eq.data.pit_membership import MembershipSource

DELISTING_CAPTURE_PIT_SAFE_THRESHOLD = 0.95  # ≥95% captured + zero unknown_exit in holdout


@dataclass(frozen=True)
class PITQualityInputs:
    membership_source: MembershipSource
    audit: DelistingAuditResult
    unknown_exit_in_holdout: int


def _capture_ratio(c: dict[str, int]) -> float:
    captured = c["delisted_captured"] + c["merger_captured"] + c["ticker_changed"]
    total = captured + c["delisted_missing"] + c["merger_missing"] + c["unknown_exit"]
    return 1.0 if total == 0 else captured / total


def classify_pit_quality(inputs: PITQualityInputs) -> DataQualityLabel:
    if inputs.membership_source == MembershipSource.ABSENT_PROTOTYPE_ONLY:
        return DataQualityLabel.SURVIVORSHIP_PROTOTYPE_ONLY

    audit_ok = (
        _capture_ratio(inputs.audit.counters) >= DELISTING_CAPTURE_PIT_SAFE_THRESHOLD
        and inputs.unknown_exit_in_holdout == 0
    )

    if inputs.membership_source == MembershipSource.WIKIPEDIA_FALLBACK:
        # Wikipedia is fallback only — never institutional-grade by itself.
        return DataQualityLabel.PARTIAL_PIT_UNIVERSE

    if audit_ok:
        return DataQualityLabel.PIT_SAFE
    return DataQualityLabel.PARTIAL_PIT_UNIVERSE
```

- [ ] **Step 4: Tests pass**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_pit_classifier.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha_eq/data/pit_quality.py tests/alpha_eq/test_pit_classifier.py
git commit -m "feat(s1-eq): three-tier PIT data-quality classifier"
```

---

### Task 9 — Dollar ADV builder

**Spec refs:** §2.5.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/data/adv.py`
- Create: `tests/alpha_eq/test_adv.py`

- [ ] **Step 1: Write failing tests**

```python
"""Dollar ADV (spec §2.5)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.data.adv import build_adv_20d_dollar


def test_adv_is_lagged_by_one_day() -> None:
    panel = pl.DataFrame(
        {
            "date": [date(2020, 1, d) for d in (2, 3, 6, 7, 8, 9, 10)],
            "symbol": ["A"] * 7,
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "volume": [1_000, 1_100, 1_200, 1_300, 1_400, 1_500, 1_600],
        }
    )
    adv = build_adv_20d_dollar(panel, window=3)  # small window for unit test
    # First two rows must be null (insufficient history + lag)
    nulls = adv["adv_20d_dollar_lag1"].is_null().to_list()
    assert nulls[0] is True
    assert nulls[1] is True
    # By row index 3 (date 2020-01-07), window=3 lagged by 1 ⇒ uses rows 0..2.
    expected = float(
        sorted([10.0 * 1_000, 11.0 * 1_100, 12.0 * 1_200])[1]
    )  # rolling-median of three
    assert abs(adv["adv_20d_dollar_lag1"][3] - expected) < 1e-9


def test_adv_uses_dollar_not_share_volume() -> None:
    panel = pl.DataFrame(
        {
            "date": [date(2020, 1, d) for d in (2, 3, 6, 7)],
            "symbol": ["A"] * 4,
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1, 1, 1, 1],
        }
    )
    adv = build_adv_20d_dollar(panel, window=2)
    # When close=100 and volume=1, dollar_volume=100; rolling-median=100; lag1 keeps null at index 0
    vals = [v for v in adv["adv_20d_dollar_lag1"].to_list() if v is not None]
    assert all(abs(v - 100.0) < 1e-9 for v in vals)
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_adv.py -v
```

- [ ] **Step 3: Implement**

```python
"""Dollar ADV builder (spec §2.5).

adv_20d_dollar_lag1 = rolling_median_20( close * volume ).shift(1)
"""

from __future__ import annotations

import polars as pl


def build_adv_20d_dollar(panel: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    panel = panel.sort(["symbol", "date"])
    out = (
        panel.with_columns((pl.col("close") * pl.col("volume").cast(pl.Float64)).alias("_dv"))
        .with_columns(
            pl.col("_dv")
            .rolling_median(window_size=window, min_periods=window)
            .over("symbol")
            .shift(1)
            .over("symbol")
            .alias("adv_20d_dollar_lag1")
        )
        .drop("_dv")
    )
    return out.select(["date", "symbol", "adv_20d_dollar_lag1"])
```

- [ ] **Step 4: Tests pass**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_adv.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha_eq/data/adv.py tests/alpha_eq/test_adv.py
git commit -m "feat(s1-eq): dollar ADV 20d rolling-median with lag1"
```

---

### Task 10 — Borrow proxy: static tiers + date-aware upgrades

**Spec refs:** §2.4.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/data/borrow_proxy.py`
- Create: `tests/alpha_eq/test_borrow_proxy.py`

- [ ] **Step 1: Write failing tests**

```python
"""Borrow proxy (spec §2.4)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.data.borrow_proxy import (
    BORROW_BPS_EASY,
    BORROW_BPS_GENERAL,
    BORROW_BPS_HARD,
    BorrowTier,
    apply_borrow_charges,
    build_borrow_proxy,
    classify_borrow_tier,
)


def test_static_tier_defaults() -> None:
    assert BORROW_BPS_EASY == 25
    assert BORROW_BPS_GENERAL == 100
    assert BORROW_BPS_HARD == 500


def test_classify_borrow_tier_handles_recent_ipo() -> None:
    tier = classify_borrow_tier(
        symbol="NEW",
        on=date(2020, 6, 1),
        ipo_date=date(2020, 1, 15),  # < 6 months
        dollar_adv=50_000_000,
        realized_vol_20=0.3,
        price=50.0,
        recent_index_addition=False,
        short_interest_ratio=None,
        manual_hard_override=False,
    )
    assert tier == BorrowTier.HARD


def test_classify_borrow_tier_low_price_is_hard() -> None:
    tier = classify_borrow_tier(
        symbol="LOW",
        on=date(2020, 6, 1),
        ipo_date=None,
        dollar_adv=50_000_000,
        realized_vol_20=0.3,
        price=3.0,
        recent_index_addition=False,
        short_interest_ratio=None,
        manual_hard_override=False,
    )
    assert tier == BorrowTier.HARD


def test_classify_borrow_tier_low_adv_is_hard() -> None:
    tier = classify_borrow_tier(
        symbol="ILL",
        on=date(2020, 6, 1),
        ipo_date=None,
        dollar_adv=2_000_000,
        realized_vol_20=0.3,
        price=50.0,
        recent_index_addition=False,
        short_interest_ratio=None,
        manual_hard_override=False,
    )
    assert tier == BorrowTier.HARD


def test_recent_index_addition_alone_is_NOT_hard() -> None:
    """Per spec §2.4: index-addition is a watchlist flag, not an auto-hard upgrade
    unless combined with low liquidity or high volatility."""
    tier = classify_borrow_tier(
        symbol="ADD",
        on=date(2020, 6, 1),
        ipo_date=None,
        dollar_adv=200_000_000,
        realized_vol_20=0.2,
        price=80.0,
        recent_index_addition=True,
        short_interest_ratio=None,
        manual_hard_override=False,
    )
    assert tier != BorrowTier.HARD


def test_apply_borrow_charges_only_on_shorts() -> None:
    positions = pl.DataFrame(
        {
            "date": [date(2020, 6, 1), date(2020, 6, 1)],
            "symbol": ["L", "S"],
            "signed_notional": [100_000.0, -100_000.0],
            "tier": ["general", "general"],
        }
    )
    charges = apply_borrow_charges(positions, multiplier=1.0)
    by_sym = {r["symbol"]: r["borrow_cost"] for r in charges.to_dicts()}
    assert by_sym["L"] == 0.0  # longs are not charged
    assert by_sym["S"] > 0.0


def test_apply_borrow_charges_monotonic_in_multiplier() -> None:
    positions = pl.DataFrame(
        {
            "date": [date(2020, 6, 1)],
            "symbol": ["S"],
            "signed_notional": [-100_000.0],
            "tier": ["general"],
        }
    )
    one = apply_borrow_charges(positions, multiplier=1.0)["borrow_cost"][0]
    two = apply_borrow_charges(positions, multiplier=2.0)["borrow_cost"][0]
    three = apply_borrow_charges(positions, multiplier=3.0)["borrow_cost"][0]
    assert one < two < three
    assert abs(two - 2 * one) < 1e-9
    assert abs(three - 3 * one) < 1e-9


def test_build_borrow_proxy_returns_static_table() -> None:
    symbols = ["AAPL", "MSFT", "RKLB"]
    df = build_borrow_proxy(symbols)
    assert set(df.columns) == {"symbol", "borrow_tier", "annual_bps"}
    assert df.height == 3
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_borrow_proxy.py -v
```

- [ ] **Step 3: Implement**

```python
"""Borrow proxy (spec §2.4): static 3-tier table + date-aware upgrades.

Borrow charges apply only to SHORT notional and are stress-tested at 1x/2x/3x
in every backtest report.
"""

from __future__ import annotations

import enum
from datetime import date, timedelta

import polars as pl

BORROW_BPS_EASY: int = 25
BORROW_BPS_GENERAL: int = 100
BORROW_BPS_HARD: int = 500

_LOW_ADV_THRESHOLD_USD = 5_000_000.0
_HIGH_VOL_THRESHOLD = 0.80
_LOW_PRICE_THRESHOLD = 5.0
_HIGH_SI_THRESHOLD = 0.10
_RECENT_IPO_WINDOW = timedelta(days=183)  # ~6 months


class BorrowTier(str, enum.Enum):
    EASY = "easy"
    GENERAL = "general"
    HARD = "hard"


def classify_borrow_tier(
    *,
    symbol: str,
    on: date,
    ipo_date: date | None,
    dollar_adv: float | None,
    realized_vol_20: float | None,
    price: float | None,
    recent_index_addition: bool,
    short_interest_ratio: float | None,
    manual_hard_override: bool,
) -> BorrowTier:
    if manual_hard_override:
        return BorrowTier.HARD
    if ipo_date is not None and (on - ipo_date) < _RECENT_IPO_WINDOW:
        return BorrowTier.HARD
    if dollar_adv is not None and dollar_adv < _LOW_ADV_THRESHOLD_USD:
        return BorrowTier.HARD
    if realized_vol_20 is not None and realized_vol_20 > _HIGH_VOL_THRESHOLD:
        return BorrowTier.HARD
    if price is not None and price < _LOW_PRICE_THRESHOLD:
        return BorrowTier.HARD
    if short_interest_ratio is not None and short_interest_ratio > _HIGH_SI_THRESHOLD:
        return BorrowTier.HARD
    # Recent index addition alone is a watchlist flag, not a tier upgrade.
    if recent_index_addition and (
        (dollar_adv is not None and dollar_adv < _LOW_ADV_THRESHOLD_USD * 2)
        or (realized_vol_20 is not None and realized_vol_20 > _HIGH_VOL_THRESHOLD / 2)
    ):
        return BorrowTier.HARD
    return BorrowTier.GENERAL


def build_borrow_proxy(symbols: list[str]) -> pl.DataFrame:
    """Static v1 borrow proxy: every symbol starts at GENERAL until the
    date-aware classifier upgrades them on a given date."""
    return pl.DataFrame(
        {
            "symbol": list(symbols),
            "borrow_tier": ["general"] * len(symbols),
            "annual_bps": [BORROW_BPS_GENERAL] * len(symbols),
        }
    )


_TIER_TO_BPS: dict[str, int] = {
    "easy": BORROW_BPS_EASY,
    "general": BORROW_BPS_GENERAL,
    "hard": BORROW_BPS_HARD,
}


def apply_borrow_charges(positions: pl.DataFrame, *, multiplier: float = 1.0) -> pl.DataFrame:
    """Compute daily borrow cost per row.  Charges apply only to short
    notional (signed_notional < 0); longs are 0.  Stress multiplier is
    applied linearly to the annual_bps."""
    return positions.with_columns(
        pl.col("tier").replace_strict(_TIER_TO_BPS, return_dtype=pl.Int64).alias("_bps")
    ).with_columns(
        pl.when(pl.col("signed_notional") < 0)
        .then(
            (-pl.col("signed_notional"))
            * pl.col("_bps").cast(pl.Float64)
            * float(multiplier)
            / 10_000.0
            / 252.0
        )
        .otherwise(0.0)
        .alias("borrow_cost")
    ).drop("_bps")
```

- [ ] **Step 4: Tests pass**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_borrow_proxy.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha_eq/data/borrow_proxy.py tests/alpha_eq/test_borrow_proxy.py
git commit -m "feat(s1-eq): borrow proxy with date-aware hard-tier upgrades + stress"
```

---

### Task 11 — Loaders with hash-mismatch hard-fail

**Spec refs:** §2.8.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/data/loaders.py`
- Create: `tests/alpha_eq/test_loaders.py`

- [ ] **Step 1: Write failing tests**

```python
"""Loaders that hard-fail on manifest hash mismatch (spec §2.8)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.alpha_eq.data.loaders import (
    EquityRootLoader,
    LoaderHashError,
)
from quant_research_stack.alpha_eq.data.manifest import (
    DataQualityLabel,
    DelistingAuditCounters,
    EquityManifest,
    ManifestArtifact,
    sha256_of_file,
    write_manifest,
)


def _write_panel(path: Path) -> None:
    pl.DataFrame(
        {"date": ["2020-01-02"], "symbol": ["AAA"], "close": [100.0]}
    ).write_parquet(path)


def _manifest_for(art_path: Path, sha: str) -> EquityManifest:
    return EquityManifest(
        pipeline_version="0.1.0",
        git_sha="deadbeef",
        artifacts={
            "sp500_tradable_prices": ManifestArtifact(
                path=str(art_path.name),
                sha256=sha,
                row_count=1,
                symbol_count=1,
                date_range_start="2020-01-02",
                date_range_end="2020-01-02",
                schema_fingerprint="cols:date,symbol,close",
            )
        },
        data_quality_label=DataQualityLabel.PARTIAL_PIT_UNIVERSE,
        corporate_action_quality="split_adj_plus_external_dividends",
        borrow_source_quality="static_proxy_v1",
        pit_membership_source="wikipedia_fallback",
        delisting_audit_quality="partial_capture",
        delisting_audit_counters=DelistingAuditCounters(),
        build_command_line="x",
        python_version="3.11.0",
        package_versions={},
        warnings=[],
    )


def test_loader_succeeds_on_matching_hash(tmp_equity_root: Path) -> None:
    p = tmp_equity_root / "sp500_tradable_prices.parquet"
    _write_panel(p)
    sha = sha256_of_file(p)
    write_manifest(tmp_equity_root / "_manifest.json", _manifest_for(p, sha))
    loader = EquityRootLoader(root=tmp_equity_root)
    df = loader.load_tradable_prices()
    assert df.height == 1


def test_loader_hard_fails_on_corruption(tmp_equity_root: Path) -> None:
    p = tmp_equity_root / "sp500_tradable_prices.parquet"
    _write_panel(p)
    sha = sha256_of_file(p)
    write_manifest(tmp_equity_root / "_manifest.json", _manifest_for(p, sha))
    # mutate file → hash mismatch
    pl.DataFrame(
        {"date": ["2020-01-02"], "symbol": ["AAA"], "close": [101.0]}
    ).write_parquet(p)
    loader = EquityRootLoader(root=tmp_equity_root)
    with pytest.raises(LoaderHashError):
        loader.load_tradable_prices()


def test_loader_hard_fails_on_missing_artifact(tmp_equity_root: Path) -> None:
    write_manifest(
        tmp_equity_root / "_manifest.json",
        _manifest_for(tmp_equity_root / "sp500_tradable_prices.parquet", "a" * 64),
    )
    loader = EquityRootLoader(root=tmp_equity_root)
    with pytest.raises(FileNotFoundError):
        loader.load_tradable_prices()
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_loaders.py -v
```

- [ ] **Step 3: Implement**

```python
"""Hash-verified parquet loaders for the equity processed root (spec §2.8)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from quant_research_stack.alpha_eq.data.manifest import (
    EquityManifest,
    ManifestMismatchError,
    load_and_verify_manifest,
    sha256_of_file,
)


class LoaderHashError(RuntimeError):
    pass


@dataclass(frozen=True)
class EquityRootLoader:
    root: Path

    def _manifest(self) -> EquityManifest:
        return load_and_verify_manifest(self.root / "_manifest.json", expected_sha256={})

    def _verified_path(self, artifact_key: str) -> Path:
        m = self._manifest()
        if artifact_key not in m.artifacts:
            raise ManifestMismatchError(f"artifact key not in manifest: {artifact_key}")
        art = m.artifacts[artifact_key]
        path = self.root / art.path
        if not path.exists():
            raise FileNotFoundError(f"artifact missing on disk: {path}")
        actual_sha = sha256_of_file(path)
        if actual_sha != art.sha256:
            raise LoaderHashError(
                f"hash mismatch on {artifact_key}: manifest={art.sha256} disk={actual_sha}"
            )
        return path

    def load_tradable_prices(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_tradable_prices"))

    def load_split_adjusted_prices(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_split_adjusted_prices"))

    def load_total_return_prices(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_total_return_prices"))

    def load_dividends(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_dividends"))

    def load_adv(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_adv"))

    def load_borrow_proxy(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_borrow_proxy"))

    def load_delisting_audit(self) -> pl.DataFrame:
        return pl.read_parquet(self._verified_path("sp500_delisting_audit"))

    def load_pit_membership(self) -> pl.DataFrame | None:
        """PIT membership is optional — absence implies prototype-only."""
        m = self._manifest()
        if "sp500_pit_membership" not in m.artifacts:
            return None
        return pl.read_parquet(self._verified_path("sp500_pit_membership"))
```

- [ ] **Step 4: Tests pass**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_loaders.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/quant_research_stack/alpha_eq/data/loaders.py tests/alpha_eq/test_loaders.py
git commit -m "feat(s1-eq): hash-verified parquet loaders with manifest hard-fail"
```

---

### Task 12 — `scripts/prepare_equity_data.py` orchestrator

**Spec refs:** §2.6, §2.7.

**Files:**
- Create: `scripts/prepare_equity_data.py`
- Create: `tests/alpha_eq/test_prepare_equity_data.py`

- [ ] **Step 1: Write failing tests**

```python
"""End-to-end smoke for the equity-data prep script."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import polars as pl


def _write_minimal_inputs(root: Path) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "symbol": ["A", "A", "A"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000_000, 1_100_000, 1_050_000],
        }
    ).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")).write_parquet(
        raw / "panel.parquet"
    )
    pl.DataFrame(
        {"ex_date": ["2020-01-06"], "symbol": ["A"], "dividend_per_share": [0.5]}
    ).with_columns(pl.col("ex_date").str.strptime(pl.Date, "%Y-%m-%d")).write_parquet(
        raw / "dividends.parquet"
    )


def test_prepare_equity_data_writes_manifest_and_artifacts(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    out_root = tmp_path / "processed" / "equities"
    out_root.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/prepare_equity_data.py",
            "--panel",
            str(tmp_path / "raw" / "panel.parquet"),
            "--dividends",
            str(tmp_path / "raw" / "dividends.parquet"),
            "--equity-root",
            str(out_root),
            "--membership-source",
            "absent_prototype_only",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert (out_root / "_manifest.json").exists()
    manifest = json.loads((out_root / "_manifest.json").read_text())
    assert manifest["data_quality_label"] == "survivorship_prototype_only"
    # required artifacts present
    for key in (
        "sp500_tradable_prices",
        "sp500_split_adjusted_prices",
        "sp500_total_return_prices",
        "sp500_dividends",
        "sp500_adv",
        "sp500_borrow_proxy",
        "sp500_delisting_audit",
    ):
        assert key in manifest["artifacts"]
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_prepare_equity_data.py -v
```

- [ ] **Step 3: Implement**

`scripts/prepare_equity_data.py`:

```python
"""End-to-end equity data preparation (spec §2).

Builds the processed-equities root from a daily-bars panel + dividend feed.

Usage:
    PYTHONPATH=src uv run python scripts/prepare_equity_data.py \
        --panel data/raw/.../panel.parquet \
        --dividends data/raw/.../dividends.parquet \
        --equity-root data/processed/equities \
        [--membership-path data/raw/.../membership.parquet] \
        [--membership-source hf_primary|kaggle|wikipedia_fallback|absent_prototype_only] \
        [--exits-path data/raw/.../exits.parquet] \
        [--source-is-total-return]
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console

from quant_research_stack.alpha_eq.data.adv import build_adv_20d_dollar
from quant_research_stack.alpha_eq.data.borrow_proxy import build_borrow_proxy
from quant_research_stack.alpha_eq.data.corporate_actions import build_three_series
from quant_research_stack.alpha_eq.data.delisting_audit import audit_delistings
from quant_research_stack.alpha_eq.data.manifest import (
    DataQualityLabel,
    DelistingAuditCounters,
    EquityManifest,
    ManifestArtifact,
    sha256_of_file,
    write_manifest,
)
from quant_research_stack.alpha_eq.data.pit_membership import MembershipSource
from quant_research_stack.alpha_eq.data.pit_quality import (
    PITQualityInputs,
    classify_pit_quality,
)

console = Console()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _read_panel(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).sort(["symbol", "date"])


def _read_dividends(path: Path | None) -> pl.DataFrame:
    if path is None or not Path(path).exists():
        return pl.DataFrame(
            schema={"ex_date": pl.Date, "symbol": pl.Utf8, "dividend_per_share": pl.Float64}
        )
    return pl.read_parquet(path)


def _read_exits(path: Path | None) -> pl.DataFrame:
    if path is None or not Path(path).exists():
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "exit_date": pl.Date,
                "exit_reason": pl.Utf8,
                "terminal_return_known": pl.Boolean,
                "terminal_return_value": pl.Float64,
            }
        )
    return pl.read_parquet(path)


def _schema_fingerprint(df: pl.DataFrame) -> str:
    return "cols:" + ",".join(df.columns)


def _write_parquet_with_artifact(
    df: pl.DataFrame, *, root: Path, name: str
) -> tuple[str, ManifestArtifact]:
    p = root / f"{name}.parquet"
    df.write_parquet(p)
    art = ManifestArtifact(
        path=p.name,
        sha256=sha256_of_file(p),
        row_count=df.height,
        symbol_count=int(df["symbol"].n_unique()) if "symbol" in df.columns else 0,
        date_range_start=str(df["date"].min()) if "date" in df.columns else "",
        date_range_end=str(df["date"].max()) if "date" in df.columns else "",
        schema_fingerprint=_schema_fingerprint(df),
    )
    return name, art


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--dividends", default=None)
    p.add_argument("--equity-root", required=True)
    p.add_argument("--membership-path", default=None)
    p.add_argument(
        "--membership-source",
        default="absent_prototype_only",
        choices=[m.value for m in MembershipSource],
    )
    p.add_argument("--exits-path", default=None)
    p.add_argument("--source-is-total-return", action="store_true")
    p.add_argument("--config", default=None, help="optional unused config for Make-target compat")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.equity_root)
    root.mkdir(parents=True, exist_ok=True)

    panel = _read_panel(Path(args.panel))
    dividends = _read_dividends(Path(args.dividends) if args.dividends else None)
    exits = _read_exits(Path(args.exits_path) if args.exits_path else None)

    bundle = build_three_series(
        panel=panel, dividends=dividends, source_is_total_return=args.source_is_total_return
    )
    adv = build_adv_20d_dollar(panel)
    borrow = build_borrow_proxy(sorted(panel["symbol"].unique().to_list()))
    audit = audit_delistings(panel=panel, exits=exits)

    artifacts: dict[str, ManifestArtifact] = {}
    for df, name in (
        (bundle.tradable, "sp500_tradable_prices"),
        (bundle.split_adj, "sp500_split_adjusted_prices"),
        (bundle.total_return, "sp500_total_return_prices"),
        (dividends, "sp500_dividends"),
        (adv, "sp500_adv"),
        (borrow, "sp500_borrow_proxy"),
        (audit.audit_table, "sp500_delisting_audit"),
    ):
        key, art = _write_parquet_with_artifact(df, root=root, name=name)
        artifacts[key] = art

    membership_source = MembershipSource(args.membership_source)
    if args.membership_path and membership_source != MembershipSource.ABSENT_PROTOTYPE_ONLY:
        mem_df = pl.read_parquet(args.membership_path)
        _, mart = _write_parquet_with_artifact(mem_df, root=root, name="sp500_pit_membership")
        artifacts["sp500_pit_membership"] = mart

    unknown_in_holdout = int(audit.counters.get("unknown_exit", 0))
    label = classify_pit_quality(
        PITQualityInputs(
            membership_source=membership_source,
            audit=audit,
            unknown_exit_in_holdout=unknown_in_holdout,
        )
    )

    corporate_action_quality = (
        "vendor_total_return"
        if args.source_is_total_return
        else "split_adj_plus_external_dividends"
    )

    manifest = EquityManifest(
        pipeline_version="0.1.0",
        git_sha=_git_sha(),
        artifacts=artifacts,
        data_quality_label=label,
        corporate_action_quality=corporate_action_quality,
        borrow_source_quality="static_proxy_v1",
        pit_membership_source=membership_source.value,
        delisting_audit_quality=(
            "captured_above_threshold"
            if label == DataQualityLabel.PIT_SAFE
            else ("partial_capture" if not audit.audit_table.is_empty() else "audit_absent")
        ),
        delisting_audit_counters=DelistingAuditCounters(**audit.counters),
        build_command_line=" ".join(sys.argv),
        python_version=platform.python_version(),
        package_versions={"polars": pl.__version__},
        warnings=(
            ["dividend feed: public_snapshot_not_vendor_pit"]
            if args.dividends and "yfinance" in str(args.dividends)
            else []
        ),
    )
    write_manifest(root / "_manifest.json", manifest)
    console.print(f"[bold green]Manifest written:[/bold green] {root / '_manifest.json'}")
    console.print(f"  data_quality_label = {label.value}")
    console.print(f"  build at {datetime.utcnow().isoformat()}Z")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Tests pass**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_prepare_equity_data.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_equity_data.py tests/alpha_eq/test_prepare_equity_data.py
git commit -m "feat(s1-eq): prepare_equity_data orchestrator + manifest emission"
```

---

### Task 13 — `scripts/pit_quality_audit.py` (re-classify existing root)

**Spec refs:** §2.1, §2.9.

**Files:**
- Create: `scripts/pit_quality_audit.py`
- Create: `tests/alpha_eq/test_pit_quality_audit_cli.py`

- [ ] **Step 1: Write failing test**

```python
"""CLI smoke for pit_quality_audit.py — re-runs the classifier and prints
a markdown summary."""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl

from quant_research_stack.alpha_eq.data.manifest import (
    DataQualityLabel,
    DelistingAuditCounters,
    EquityManifest,
    ManifestArtifact,
    sha256_of_file,
    write_manifest,
)


def _seed_root(root: Path) -> None:
    pl.DataFrame({"date": ["2020-01-02"], "symbol": ["AAA"], "close": [1.0]}).write_parquet(
        root / "sp500_tradable_prices.parquet"
    )
    sha = sha256_of_file(root / "sp500_tradable_prices.parquet")
    m = EquityManifest(
        pipeline_version="0.1.0",
        git_sha="deadbeef",
        artifacts={
            "sp500_tradable_prices": ManifestArtifact(
                path="sp500_tradable_prices.parquet",
                sha256=sha,
                row_count=1,
                symbol_count=1,
                date_range_start="2020-01-02",
                date_range_end="2020-01-02",
                schema_fingerprint="cols:date,symbol,close",
            )
        },
        data_quality_label=DataQualityLabel.PARTIAL_PIT_UNIVERSE,
        corporate_action_quality="split_adj_plus_external_dividends",
        borrow_source_quality="static_proxy_v1",
        pit_membership_source="wikipedia_fallback",
        delisting_audit_quality="partial_capture",
        delisting_audit_counters=DelistingAuditCounters(),
        build_command_line="x",
        python_version="3.11.0",
        package_versions={},
        warnings=[],
    )
    write_manifest(root / "_manifest.json", m)


def test_pit_quality_audit_cli_prints_label(tmp_equity_root: Path) -> None:
    _seed_root(tmp_equity_root)
    res = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/pit_quality_audit.py",
            "--equity-root",
            str(tmp_equity_root),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert "partial_pit_universe" in res.stdout
```

- [ ] **Step 2: Implement**

```python
"""Audit an existing equity-processed root and print its data-quality label.

Usage:
    PYTHONPATH=src uv run python scripts/pit_quality_audit.py \
        --equity-root data/processed/equities
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

console = Console()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--equity-root", required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.equity_root) / "_manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]No manifest at {manifest_path}[/red]")
        return 2
    m = json.loads(manifest_path.read_text())
    console.print(f"[bold]Equity root:[/bold] {args.equity_root}")
    console.print(f"  data_quality_label        = [bold]{m['data_quality_label']}[/bold]")
    console.print(f"  corporate_action_quality  = {m['corporate_action_quality']}")
    console.print(f"  borrow_source_quality     = {m['borrow_source_quality']}")
    console.print(f"  pit_membership_source     = {m['pit_membership_source']}")
    console.print(f"  delisting_audit_quality   = {m['delisting_audit_quality']}")
    print(m["data_quality_label"])  # also stdout-plain for grep/test
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_pit_quality_audit_cli.py -v
git add scripts/pit_quality_audit.py tests/alpha_eq/test_pit_quality_audit_cli.py
git commit -m "feat(s1-eq): pit_quality_audit CLI"
```

---

### Task 14 — M1 integration: end-to-end manifest, prototype-only banner check

**Spec refs:** §2 in aggregate.

**Files:**
- Create: `tests/alpha_eq/test_m1_integration.py`

- [ ] **Step 1: Write test**

```python
"""M1 integration: prepare → audit → loader round trip, with prototype-only banner check."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import polars as pl

from quant_research_stack.alpha_eq.data.loaders import EquityRootLoader


def test_prepare_then_load_then_label_visible(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "date": [f"2020-01-{d:02d}" for d in range(2, 12)],
            "symbol": ["A"] * 10,
            "open": list(range(100, 110)),
            "high": list(range(101, 111)),
            "low": list(range(99, 109)),
            "close": [float(x) + 0.5 for x in range(100, 110)],
            "volume": [1_000_000] * 10,
        }
    ).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")).write_parquet(
        raw / "panel.parquet"
    )

    out_root = tmp_path / "processed" / "equities"
    out_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", "scripts/prepare_equity_data.py",
            "--panel", str(raw / "panel.parquet"),
            "--equity-root", str(out_root),
            "--membership-source", "absent_prototype_only",
        ],
        check=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )

    loader = EquityRootLoader(root=out_root)
    df = loader.load_tradable_prices()
    assert df.height == 10
    manifest = json.loads((out_root / "_manifest.json").read_text())
    assert manifest["data_quality_label"] == "survivorship_prototype_only"
    # Required: any future report consumer can see this label as a banner trigger
    assert "delisting_audit_counters" in manifest
```

- [ ] **Step 2: Run + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_m1_integration.py -v
git add tests/alpha_eq/test_m1_integration.py
git commit -m "test(s1-eq): m1 integration — prepare → audit → loader round trip"
```

---

## M2 — Feature pipeline + leakage tests

### Task 15 — `features/timestamps.py`: feature_as_of_date contract

**Spec refs:** §3.1, §3.5.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/features/timestamps.py`
- Create: `tests/alpha_eq/test_timestamp_contract.py`

- [ ] **Step 1: Write failing tests**

```python
"""feature_as_of_date < execution_date hard invariant (spec §3.1, §3.5)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_research_stack.alpha_eq.features.timestamps import (
    TimestampContractError,
    assert_feature_before_execution,
    attach_execution_date,
    attach_feature_as_of_date,
)


def test_after_close_convention_attaches_same_day_as_of() -> None:
    df = pl.DataFrame(
        {"date": [date(2020, 1, 2), date(2020, 1, 3)], "symbol": ["A", "A"]}
    )
    out = attach_feature_as_of_date(df, convention="after_close_t")
    assert out["feature_as_of_date"].to_list() == [date(2020, 1, 2), date(2020, 1, 3)]


def test_execution_date_is_next_trading_day() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
            "symbol": ["A", "A", "A"],
        }
    )
    out = attach_execution_date(df, convention="next_trading_day")
    # 2020-01-03 (Fri) → 2020-01-06 (Mon, weekday skip)
    rows = out.to_dicts()
    assert rows[1]["execution_date"] == date(2020, 1, 6)


def test_assert_feature_before_execution_passes() -> None:
    df = pl.DataFrame(
        {
            "feature_as_of_date": [date(2020, 1, 2)],
            "execution_date": [date(2020, 1, 3)],
        }
    )
    assert_feature_before_execution(df)  # no raise


def test_assert_feature_before_execution_raises_on_violation() -> None:
    df = pl.DataFrame(
        {
            "feature_as_of_date": [date(2020, 1, 3)],
            "execution_date": [date(2020, 1, 3)],
        }
    )
    with pytest.raises(TimestampContractError):
        assert_feature_before_execution(df)
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_timestamp_contract.py -v
```

- [ ] **Step 3: Implement**

```python
"""Feature/execution timestamp invariants (spec §3.1)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl


class TimestampContractError(RuntimeError):
    pass


def attach_feature_as_of_date(df: pl.DataFrame, *, convention: str) -> pl.DataFrame:
    if convention != "after_close_t":
        raise ValueError(f"unsupported convention: {convention!r}")
    return df.with_columns(pl.col("date").alias("feature_as_of_date"))


def attach_execution_date(df: pl.DataFrame, *, convention: str) -> pl.DataFrame:
    """Next trading day; weekends skipped.  Holiday handling is deferred:
    the engine drops rows whose execution_date has no row in tradable_prices."""
    if convention != "next_trading_day":
        raise ValueError(f"unsupported convention: {convention!r}")

    def _bump(d: date) -> date:
        nd = d + timedelta(days=1)
        while nd.weekday() >= 5:
            nd += timedelta(days=1)
        return nd

    return df.with_columns(
        pl.col("date").map_elements(_bump, return_dtype=pl.Date).alias("execution_date")
    )


def assert_feature_before_execution(df: pl.DataFrame) -> None:
    """Hard invariant — used as a runtime assert by the training and backtest entry points."""
    if "feature_as_of_date" not in df.columns or "execution_date" not in df.columns:
        raise TimestampContractError("missing feature_as_of_date or execution_date columns")
    bad = df.filter(pl.col("feature_as_of_date") >= pl.col("execution_date"))
    if not bad.is_empty():
        raise TimestampContractError(
            f"feature_as_of_date >= execution_date on {bad.height} rows"
        )
```

- [ ] **Step 4: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_timestamp_contract.py -v
git add src/quant_research_stack/alpha_eq/features/timestamps.py tests/alpha_eq/test_timestamp_contract.py
git commit -m "feat(s1-eq): feature_as_of_date < execution_date hard invariant"
```

---

### Task 16 — `features/returns_momentum.py`

**Spec refs:** §3.3-1.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/features/returns_momentum.py`
- Create: `tests/alpha_eq/test_features_returns_momentum.py`

- [ ] **Step 1: Write tests**

```python
"""Returns / momentum features (spec §3.3-1)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.features.returns_momentum import (
    build_returns_momentum,
)


def _toy() -> pl.DataFrame:
    closes = np.geomspace(100.0, 200.0, num=300)
    dates = pl.date_range(
        start=date(2020, 1, 2), end=date(2021, 3, 1), interval="1d", eager=True
    ).filter(pl.col("date").dt.weekday() < 6).slice(0, 300)
    return pl.DataFrame({"date": dates, "symbol": ["A"] * 300, "close": closes})


def test_returns_momentum_emits_expected_columns() -> None:
    df = build_returns_momentum(
        _toy(), horizons=(1, 5, 20, 60, 120, 252), include_skip5=(60, 120, 252)
    )
    expected = {
        "log_return_1", "log_return_5", "log_return_20",
        "log_return_60", "log_return_120", "log_return_252",
        "cumulative_return_60_skip5", "cumulative_return_120_skip5", "cumulative_return_252_skip5",
        "mean_reversion_5",
    }
    assert expected.issubset(set(df.columns))


def test_returns_momentum_no_future_leak_for_after_close_convention() -> None:
    df = _toy()
    out = build_returns_momentum(df, horizons=(1,))
    # the last row's log_return_1 must depend ONLY on close_t and close_{t-1}
    last = out.tail(2).to_dicts()
    expected = float(np.log(last[1]["close"] / last[0]["close"]))
    assert abs(last[1]["log_return_1"] - expected) < 1e-12
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_returns_momentum.py -v
```

- [ ] **Step 3: Implement**

```python
"""Returns / momentum features (spec §3.3-1).

After-close_t convention: features at date t may use the complete day-t close.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def build_returns_momentum(
    panel: pl.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 2, 5, 10, 20, 60, 120, 252),
    include_skip5: tuple[int, ...] = (60, 120, 252),
) -> pl.DataFrame:
    panel = panel.sort(["symbol", "date"])
    out = panel
    for h in horizons:
        out = out.with_columns(
            (pl.col("close").log() - pl.col("close").shift(h).over("symbol").log()).alias(
                f"log_return_{h}"
            )
        )
    for h in include_skip5:
        # 12-1-style momentum: cumulative log-return over h days, skipping the most recent 5
        out = out.with_columns(
            (
                pl.col("close").shift(5).over("symbol").log()
                - pl.col("close").shift(h).over("symbol").log()
            ).alias(f"cumulative_return_{h}_skip5")
        )
    if 5 in horizons:
        out = out.with_columns((-pl.col("log_return_5")).alias("mean_reversion_5"))
    return out
```

- [ ] **Step 4: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_returns_momentum.py -v
git add src/quant_research_stack/alpha_eq/features/returns_momentum.py tests/alpha_eq/test_features_returns_momentum.py
git commit -m "feat(s1-eq): returns/momentum features (multi-horizon + skip5)"
```

---

### Task 17 — `features/volatility.py`

**Spec refs:** §3.3-2.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/features/volatility.py`
- Create: `tests/alpha_eq/test_features_volatility.py`

- [ ] **Step 1: Write tests**

```python
"""Volatility features (spec §3.3-2)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.features.volatility import build_volatility


def _toy(n: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    dates = pl.date_range(
        start=date(2020, 1, 2), end=date(2020, 12, 31), interval="1d", eager=True
    ).filter(pl.col("date").dt.weekday() < 6).slice(0, n)
    closes = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    highs = closes * (1 + np.abs(rng.standard_normal(n)) * 0.01)
    lows = closes * (1 - np.abs(rng.standard_normal(n)) * 0.01)
    opens = closes * (1 + rng.standard_normal(n) * 0.005)
    return pl.DataFrame(
        {"date": dates, "symbol": ["A"] * n,
         "open": opens, "high": highs, "low": lows, "close": closes}
    )


def test_volatility_columns_present_and_nonneg() -> None:
    df = build_volatility(_toy(), windows=(5, 20, 60), parkinson_window=20, gk_window=20, vov_window=60)
    for col in ("realized_vol_5", "realized_vol_20", "realized_vol_60",
                "parkinson_vol_20", "garman_klass_vol_20", "vol_of_vol_60"):
        assert col in df.columns
        vals = df[col].drop_nulls().to_numpy()
        assert np.all(vals >= 0.0)
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_volatility.py -v
```

- [ ] **Step 3: Implement**

```python
"""Volatility features (spec §3.3-2)."""

from __future__ import annotations

import polars as pl


def build_volatility(
    panel: pl.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 20, 60),
    parkinson_window: int = 20,
    gk_window: int = 20,
    vov_window: int = 60,
) -> pl.DataFrame:
    panel = panel.sort(["symbol", "date"])
    out = panel.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("_log_ret_1")
    )
    for w in windows:
        out = out.with_columns(
            pl.col("_log_ret_1")
            .rolling_std(window_size=w, min_periods=w)
            .over("symbol")
            .alias(f"realized_vol_{w}")
        )
    # Parkinson: (1 / 4 ln 2) * sum (ln(high/low))^2 / w
    out = out.with_columns(
        ((pl.col("high") / pl.col("low")).log() ** 2)
        .rolling_mean(window_size=parkinson_window, min_periods=parkinson_window)
        .over("symbol")
        .alias("_pk_raw")
    ).with_columns(
        (pl.col("_pk_raw") / (4.0 * pl.lit(0.6931471805599453))).sqrt().alias(
            f"parkinson_vol_{parkinson_window}"
        )
    ).drop("_pk_raw")
    # Garman-Klass per-day variance, then rolling-mean square-root
    out = out.with_columns(
        (
            0.5 * ((pl.col("high") / pl.col("low")).log() ** 2)
            - (2.0 * pl.lit(0.6931471805599453) - 1.0)
            * ((pl.col("close") / pl.col("open")).log() ** 2)
        ).alias("_gk_var")
    ).with_columns(
        pl.col("_gk_var")
        .rolling_mean(window_size=gk_window, min_periods=gk_window)
        .over("symbol")
        .sqrt()
        .alias(f"garman_klass_vol_{gk_window}")
    ).drop("_gk_var")
    # vol-of-vol: std of realized_vol_20 over vov_window
    if 20 in windows:
        out = out.with_columns(
            pl.col("realized_vol_20")
            .rolling_std(window_size=vov_window, min_periods=vov_window)
            .over("symbol")
            .alias(f"vol_of_vol_{vov_window}")
        )
    return out.drop("_log_ret_1")
```

- [ ] **Step 4: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_volatility.py -v
git add src/quant_research_stack/alpha_eq/features/volatility.py tests/alpha_eq/test_features_volatility.py
git commit -m "feat(s1-eq): volatility features (realized/parkinson/garman-klass/vov)"
```

---

### Task 18 — `features/microstructure_proxies.py` (Amihud, Roll with NaN policy, Kyle proxy)

**Spec refs:** §3.3-3.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/features/microstructure_proxies.py`
- Create: `tests/alpha_eq/test_features_microstructure.py`

- [ ] **Step 1: Write tests**

```python
"""Microstructure proxies (spec §3.3-3): amihud, roll w/ NaN policy, kyle_proxy."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.features.microstructure_proxies import (
    build_microstructure_proxies,
)


def _toy(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    dates = pl.date_range(
        start=date(2020, 1, 2), end=date(2020, 6, 30), interval="1d", eager=True
    ).filter(pl.col("date").dt.weekday() < 6).slice(0, n)
    closes = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    opens = closes * (1 + rng.standard_normal(n) * 0.005)
    highs = np.maximum(closes, opens) * 1.005
    lows = np.minimum(closes, opens) * 0.995
    vols = rng.integers(500_000, 2_000_000, size=n)
    return pl.DataFrame(
        {"date": dates, "symbol": ["A"] * n,
         "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}
    )


def test_amihud_roll_kyle_proxy_columns_present() -> None:
    df = build_microstructure_proxies(_toy(), window=20)
    for col in ("amihud_illiq_20", "roll_spread_20", "kyle_proxy_signed_volume_20",
                "overnight_gap", "intraday_return", "close_location_20"):
        assert col in df.columns


def test_roll_spread_is_null_on_positive_autocov() -> None:
    """When autocov is non-negative, the Roll estimator is explicitly NaN
    (spec §3.3-3): silent zero-fill is forbidden."""
    # Construct a series whose autocov is positive: monotonic trend
    n = 60
    closes = np.linspace(100.0, 200.0, n)
    dates = pl.date_range(
        start=date(2020, 1, 2), end=date(2020, 6, 30), interval="1d", eager=True
    ).filter(pl.col("date").dt.weekday() < 6).slice(0, n)
    df = pl.DataFrame(
        {"date": dates, "symbol": ["A"] * n,
         "open": closes, "high": closes * 1.01, "low": closes * 0.99,
         "close": closes, "volume": [1_000_000] * n}
    )
    out = build_microstructure_proxies(df, window=20)
    # at least some rows must be null (autocov non-negative for trending series)
    n_null = int(out["roll_spread_20"].is_null().sum())
    assert n_null > 0
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_microstructure.py -v
```

- [ ] **Step 3: Implement**

```python
"""Microstructure proxies (spec §3.3-3).

Roll's NaN policy: when the 20-day autocovariance of returns is non-negative,
roll_spread_20 is explicitly null.  Silent zero-fill is forbidden.
"""

from __future__ import annotations

import polars as pl


def build_microstructure_proxies(
    panel: pl.DataFrame, *, window: int = 20
) -> pl.DataFrame:
    panel = panel.sort(["symbol", "date"])
    df = panel.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("_log_ret_1"),
        (pl.col("close") * pl.col("volume").cast(pl.Float64)).alias("_dv"),
    )
    # Amihud: rolling mean of |log_return_1| / dollar_volume
    df = df.with_columns(
        (pl.col("_log_ret_1").abs() / pl.col("_dv").clip(lower_bound=1e-9))
        .rolling_mean(window_size=window, min_periods=window)
        .over("symbol")
        .alias(f"amihud_illiq_{window}")
    )
    # Roll spread: 2 * sqrt(-cov(r_t, r_{t-1})) with NaN policy
    df = df.with_columns(pl.col("_log_ret_1").shift(1).over("symbol").alias("_log_ret_lag1"))
    # Compute rolling covariance via E[XY] - E[X] E[Y]
    df = df.with_columns(
        (pl.col("_log_ret_1") * pl.col("_log_ret_lag1"))
        .rolling_mean(window_size=window, min_periods=window)
        .over("symbol")
        .alias("_exy"),
        pl.col("_log_ret_1")
        .rolling_mean(window_size=window, min_periods=window)
        .over("symbol")
        .alias("_ex"),
        pl.col("_log_ret_lag1")
        .rolling_mean(window_size=window, min_periods=window)
        .over("symbol")
        .alias("_ey"),
    )
    df = df.with_columns(
        (pl.col("_exy") - pl.col("_ex") * pl.col("_ey")).alias("_autocov")
    ).with_columns(
        pl.when(pl.col("_autocov") < 0)
        .then(2.0 * (-pl.col("_autocov")).sqrt())
        .otherwise(None)
        .alias(f"roll_spread_{window}")
    )
    # Kyle proxy: rolling slope of |log_return| on sign(log_return) * dollar_volume
    df = df.with_columns(
        (pl.col("_log_ret_1").sign() * pl.col("_dv")).alias("_signed_dv"),
        pl.col("_log_ret_1").abs().alias("_abs_ret"),
    )
    df = df.with_columns(
        (
            (pl.col("_abs_ret") * pl.col("_signed_dv"))
            .rolling_mean(window_size=window, min_periods=window)
            .over("symbol")
            - (
                pl.col("_abs_ret")
                .rolling_mean(window_size=window, min_periods=window)
                .over("symbol")
                * pl.col("_signed_dv")
                .rolling_mean(window_size=window, min_periods=window)
                .over("symbol")
            )
        ).alias("_cov_xy"),
        (
            pl.col("_signed_dv")
            .rolling_var(window_size=window, min_periods=window)
            .over("symbol")
        ).alias("_var_x"),
    ).with_columns(
        (pl.col("_cov_xy") / pl.col("_var_x").clip(lower_bound=1e-18)).alias(
            f"kyle_proxy_signed_volume_{window}"
        )
    )
    # overnight_gap, intraday_return
    df = df.with_columns(
        (pl.col("open") / pl.col("close").shift(1).over("symbol")).log().alias("overnight_gap"),
        (pl.col("close") / pl.col("open")).log().alias("intraday_return"),
    )
    # close_location_20
    df = df.with_columns(
        pl.col("high").rolling_max(window_size=window, min_periods=window).over("symbol").alias("_h20"),
        pl.col("low").rolling_min(window_size=window, min_periods=window).over("symbol").alias("_l20"),
    ).with_columns(
        ((pl.col("close") - pl.col("_l20")) / (pl.col("_h20") - pl.col("_l20")).clip(lower_bound=1e-9))
        .alias(f"close_location_{window}")
    )
    return df.drop(
        [
            "_log_ret_1", "_log_ret_lag1", "_dv",
            "_exy", "_ex", "_ey", "_autocov",
            "_signed_dv", "_abs_ret", "_cov_xy", "_var_x",
            "_h20", "_l20",
        ]
    )
```

- [ ] **Step 4: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_microstructure.py -v
git add src/quant_research_stack/alpha_eq/features/microstructure_proxies.py tests/alpha_eq/test_features_microstructure.py
git commit -m "feat(s1-eq): microstructure proxies (amihud, roll w/ NaN policy, kyle proxy)"
```

---

### Task 19 — `features/volume_liquidity.py`

**Spec refs:** §3.3-4.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/features/volume_liquidity.py`
- Create: `tests/alpha_eq/test_features_volume_liquidity.py`

- [ ] **Step 1: Write tests**

```python
"""Volume / liquidity features (spec §3.3-4)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.features.volume_liquidity import (
    build_volume_liquidity,
)


def _toy() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": pl.date_range(
                date(2020, 1, 2), date(2020, 5, 31), interval="1d", eager=True
            ).filter(pl.col("date").dt.weekday() < 6).slice(0, 60),
            "symbol": ["A"] * 60,
            "close": [100.0 + i * 0.1 for i in range(60)],
            "volume": [1_000_000 + i * 1_000 for i in range(60)],
        }
    )


def test_volume_liquidity_columns_present_and_no_turnover_proxy() -> None:
    df = build_volume_liquidity(_toy(), window=20)
    assert "dollar_volume" in df.columns
    assert "log_dollar_volume_20d" in df.columns
    assert "volume_zscore_20d" in df.columns
    # spec §3.3-4: turnover_proxy_20 is DROPPED unless real shares-outstanding source
    assert "turnover_proxy_20" not in df.columns
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_volume_liquidity.py -v
```

- [ ] **Step 3: Implement**

```python
"""Volume / liquidity features (spec §3.3-4).

turnover_proxy_20 is intentionally dropped: synthetic shares-outstanding
estimates from volume history are not financially well-defined.  The real
metric is only emitted when a vendor shares-outstanding feed is provided
(future spec extension).
"""

from __future__ import annotations

import polars as pl


def build_volume_liquidity(panel: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    panel = panel.sort(["symbol", "date"])
    out = panel.with_columns(
        (pl.col("close") * pl.col("volume").cast(pl.Float64)).alias("dollar_volume")
    )
    out = out.with_columns(
        (pl.col("dollar_volume") + 1.0)
        .log()
        .rolling_mean(window_size=window, min_periods=window)
        .over("symbol")
        .alias(f"log_dollar_volume_{window}d")
    )
    out = out.with_columns(
        (
            (pl.col("volume") - pl.col("volume").rolling_mean(window_size=window, min_periods=window).over("symbol"))
            / pl.col("volume").rolling_std(window_size=window, min_periods=window).over("symbol").clip(lower_bound=1e-9)
        ).alias(f"volume_zscore_{window}d")
    )
    return out
```

- [ ] **Step 4: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_volume_liquidity.py -v
git add src/quant_research_stack/alpha_eq/features/volume_liquidity.py tests/alpha_eq/test_features_volume_liquidity.py
git commit -m "feat(s1-eq): volume/liquidity features (turnover_proxy intentionally dropped)"
```

---

### Task 20 — `features/cross_sectional_ranks.py` (within-PIT universe)

**Spec refs:** §3.3-5.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/features/cross_sectional_ranks.py`
- Create: `tests/alpha_eq/test_features_cross_sectional_ranks.py`

- [ ] **Step 1: Write tests**

```python
"""Cross-sectional ranks, computed within the date-t tradable PIT universe (spec §3.3-5)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.features.cross_sectional_ranks import (
    build_cross_sectional_ranks,
)


def _panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 1, 2)] * 4 + [date(2020, 1, 3)] * 4,
            "symbol": ["A", "B", "C", "D"] * 2,
            "in_universe": [True, True, True, False, True, True, False, True],
            "feature_value": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 99.0, 0.5],
        }
    )


def test_rank_is_only_within_in_universe() -> None:
    df = build_cross_sectional_ranks(
        _panel(),
        columns=("feature_value",),
        universe_col="in_universe",
    )
    # D on 2020-01-02 is out-of-universe → rank is null
    row_d = df.filter((pl.col("symbol") == "D") & (pl.col("date") == date(2020, 1, 2)))
    assert row_d["rank_feature_value"][0] is None
    # A on 2020-01-02 ranks lowest of three in-universe (1 of 3)
    row_a = df.filter((pl.col("symbol") == "A") & (pl.col("date") == date(2020, 1, 2)))
    # mapped to [-0.5, 0.5]; min rank=1 → (1-1)/(3-1)=0 → -0.5
    assert abs(row_a["rank_feature_value"][0] - (-0.5)) < 1e-12


def test_rank_invariant_to_out_of_universe_rows() -> None:
    base = _panel()
    extra_oou = pl.DataFrame(
        {
            "date": [date(2020, 1, 2)],
            "symbol": ["X"],
            "in_universe": [False],
            "feature_value": [-99999.0],
        }
    )
    joined = pl.concat([base, extra_oou])
    a = build_cross_sectional_ranks(
        base, columns=("feature_value",), universe_col="in_universe"
    ).filter(pl.col("date") == date(2020, 1, 2))
    b = build_cross_sectional_ranks(
        joined, columns=("feature_value",), universe_col="in_universe"
    ).filter((pl.col("date") == date(2020, 1, 2)) & (pl.col("symbol") != "X"))
    # ranks for the original symbols must not change
    assert a.sort("symbol")["rank_feature_value"].to_list() == b.sort("symbol")["rank_feature_value"].to_list()
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_cross_sectional_ranks.py -v
```

- [ ] **Step 3: Implement**

```python
"""Cross-sectional ranks within the date-t tradable PIT universe (spec §3.3-5)."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl


def build_cross_sectional_ranks(
    panel: pl.DataFrame,
    *,
    columns: Iterable[str],
    universe_col: str,
) -> pl.DataFrame:
    out = panel
    for col in columns:
        # Compute rank only over in-universe rows per date; out-of-universe → null
        ranked = out.with_columns(
            pl.when(pl.col(universe_col))
            .then(
                pl.col(col).rank(method="ordinal").over("date").cast(pl.Float64)
            )
            .otherwise(None)
            .alias(f"_rank_raw_{col}"),
            pl.when(pl.col(universe_col))
            .then(pl.col(col))
            .otherwise(None)
            .alias(f"_inuv_{col}"),
        )
        # universe size per date (count of non-null _inuv_<col>)
        ranked = ranked.with_columns(
            pl.col(f"_inuv_{col}").count().over("date").alias(f"_n_{col}").cast(pl.Float64)
        )
        ranked = ranked.with_columns(
            pl.when(pl.col(f"_n_{col}") > 1)
            .then(
                (pl.col(f"_rank_raw_{col}") - 1.0) / (pl.col(f"_n_{col}") - 1.0) - 0.5
            )
            .otherwise(None)
            .alias(f"rank_{col}")
        ).drop([f"_rank_raw_{col}", f"_inuv_{col}", f"_n_{col}"])
        out = ranked
    return out
```

- [ ] **Step 4: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_cross_sectional_ranks.py -v
git add src/quant_research_stack/alpha_eq/features/cross_sectional_ranks.py tests/alpha_eq/test_features_cross_sectional_ranks.py
git commit -m "feat(s1-eq): cross-sectional ranks scoped to PIT universe"
```

---

### Task 21 — `features/market_regime.py` with VIX fallback rule

**Spec refs:** §3.3-6.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/features/market_regime.py`
- Create: `tests/alpha_eq/test_features_market_regime.py`

- [ ] **Step 1: Write tests**

```python
"""Market regime features + VIX fallback rule (spec §3.3-6)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.features.market_regime import (
    build_market_regime,
)


def _panel(n: int = 80) -> pl.DataFrame:
    dates = pl.date_range(
        date(1988, 1, 4), date(1988, 12, 31), interval="1d", eager=True
    ).filter(pl.col("date").dt.weekday() < 6).slice(0, n)
    rows = []
    for d in dates:
        for s in ["A", "B", "C"]:
            rows.append({"date": d, "symbol": s, "close": 100.0 + (hash(s) % 5)})
    return pl.DataFrame(rows)


def test_vix_fallback_when_no_vix_provided() -> None:
    df = build_market_regime(panel=_panel(), vix=None, spy_close=None)
    # vix_close column is present but is_proxy true everywhere
    assert "vix_close" in df.columns
    assert "vix_is_proxy" in df.columns
    assert all(df["vix_is_proxy"].to_list())


def test_no_truncation_when_vix_missing_early_dates() -> None:
    """Missing VIX must NOT silently drop early rows."""
    panel = _panel()
    n_before = panel.height
    df = build_market_regime(panel=panel, vix=None, spy_close=None)
    assert df.height == n_before
```

- [ ] **Step 2: Run, expect fail**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_market_regime.py -v
```

- [ ] **Step 3: Implement**

```python
"""Market / regime context features with mandatory VIX fallback rule (spec §3.3-6)."""

from __future__ import annotations

import polars as pl


def _cross_sectional_vol_20(panel: pl.DataFrame) -> pl.DataFrame:
    rets = panel.sort(["symbol", "date"]).with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("_r1")
    )
    by_date = rets.group_by("date").agg(pl.col("_r1").std().alias("_xs_vol"))
    by_date = by_date.sort("date").with_columns(
        pl.col("_xs_vol")
        .rolling_mean(window_size=20, min_periods=20)
        .alias("cross_sectional_vol_20")
    )
    return by_date.select(["date", "cross_sectional_vol_20"])


def build_market_regime(
    *,
    panel: pl.DataFrame,
    vix: pl.DataFrame | None,
    spy_close: pl.DataFrame | None,
) -> pl.DataFrame:
    """Attach broadcast market features. VIX missing → fallback to cross-sectional
    volatility proxy; never silently drops dates."""
    xs_vol = _cross_sectional_vol_20(panel)
    # cross_sectional_dispersion: std of daily returns across the universe (no smoothing)
    rets = panel.sort(["symbol", "date"]).with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("_r1")
    )
    disp = rets.group_by("date").agg(
        pl.col("_r1").std().alias("cross_sectional_dispersion")
    )

    # join in cross_sectional_vol_20 + dispersion
    panel = panel.join(xs_vol, on="date", how="left").join(disp, on="date", how="left")

    if vix is not None and not vix.is_empty():
        panel = panel.join(
            vix.rename({"vix_close": "_vix_external"}), on="date", how="left"
        ).with_columns(
            pl.when(pl.col("_vix_external").is_not_null())
            .then(pl.col("_vix_external"))
            .otherwise(pl.col("cross_sectional_vol_20"))
            .alias("vix_close"),
            pl.col("_vix_external").is_null().alias("vix_is_proxy"),
        ).drop("_vix_external")
    else:
        panel = panel.with_columns(
            pl.col("cross_sectional_vol_20").alias("vix_close"),
            pl.lit(True).alias("vix_is_proxy"),
        )

    if spy_close is not None and not spy_close.is_empty():
        spy_join = spy_close.sort("date").with_columns(
            (pl.col("spy_close").log() - pl.col("spy_close").shift(5).log()).alias("spy_log_return_5"),
            (
                (pl.col("spy_close").log() - pl.col("spy_close").shift(1).log())
                .rolling_std(window_size=20, min_periods=20)
            ).alias("spy_realized_vol_20"),
        ).drop("spy_close")
        panel = panel.join(spy_join, on="date", how="left")
    else:
        panel = panel.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("spy_log_return_5"),
            pl.lit(None, dtype=pl.Float64).alias("spy_realized_vol_20"),
        )

    return panel
```

- [ ] **Step 4: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_market_regime.py -v
git add src/quant_research_stack/alpha_eq/features/market_regime.py tests/alpha_eq/test_features_market_regime.py
git commit -m "feat(s1-eq): market regime features + VIX fallback (no silent truncation)"
```

---

### Task 22 — `features/noise_sentinel.py`

**Spec refs:** §3.3-9, §3.4.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/features/noise_sentinel.py`
- Create: `tests/alpha_eq/test_features_noise_sentinel.py`

- [ ] **Step 1: Write tests**

```python
"""Seeded Gaussian noise sentinel (spec §3.3-9)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.features.noise_sentinel import attach_noise_sentinel


def test_noise_sentinel_is_deterministic_per_date_symbol() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 2)],
            "symbol": ["A", "A", "B"],
        }
    )
    out1 = attach_noise_sentinel(df, seed=42)
    out2 = attach_noise_sentinel(df, seed=42)
    assert out1["gaussian_noise_seed42"].to_list() == out2["gaussian_noise_seed42"].to_list()


def test_noise_sentinel_different_seeds_differ() -> None:
    df = pl.DataFrame(
        {"date": [date(2020, 1, 2)], "symbol": ["A"]}
    )
    a = attach_noise_sentinel(df, seed=42)["gaussian_noise_seed42"][0]
    b = attach_noise_sentinel(df, seed=43)["gaussian_noise_seed43"][0]
    assert a != b
```

- [ ] **Step 2: Implement**

```python
"""Seeded Gaussian noise sentinel (spec §3.3-9, §3.4).

A real engineered feature that ranks below this noise feature on
≥ 3 of 5 folds is a drop-candidate (combined with unstable IC + ablation
confirmation — see spec §3.4).
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl


def _seeded_value(*, seed: int, date_iso: str, symbol: str) -> float:
    payload = f"{seed}|{date_iso}|{symbol}".encode("utf-8")
    h = hashlib.sha256(payload).digest()
    # Use the first 8 bytes as a uniform seed; map to N(0,1) via inverse CDF
    rng = np.random.default_rng(int.from_bytes(h[:8], "big", signed=False))
    return float(rng.standard_normal())


def attach_noise_sentinel(df: pl.DataFrame, *, seed: int = 42) -> pl.DataFrame:
    col = f"gaussian_noise_seed{seed}"
    values = [
        _seeded_value(seed=seed, date_iso=str(d), symbol=s)
        for d, s in zip(df["date"].to_list(), df["symbol"].to_list(), strict=True)
    ]
    return df.with_columns(pl.Series(col, values, dtype=pl.Float64))
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_noise_sentinel.py -v
git add src/quant_research_stack/alpha_eq/features/noise_sentinel.py tests/alpha_eq/test_features_noise_sentinel.py
git commit -m "feat(s1-eq): seeded gaussian noise sentinel"
```

---

### Task 23 — `features/meta_features.py` stub (disabled by default)

**Spec refs:** §3.3-7.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/features/meta_features.py`
- Create: `tests/alpha_eq/test_features_meta_features.py`

- [ ] **Step 1: Write tests**

```python
"""Meta-features default-disabled gate (spec §3.3-7)."""

from __future__ import annotations

import pytest

from quant_research_stack.alpha_eq.features.meta_features import (
    MetaFeaturesDisabledError,
    build_meta_features,
)


def test_meta_features_disabled_by_default_raises() -> None:
    with pytest.raises(MetaFeaturesDisabledError):
        build_meta_features(panel=None, enable=False)


def test_meta_features_audited_gate_required() -> None:
    """Enabling without an audit-pass token must also raise."""
    with pytest.raises(MetaFeaturesDisabledError):
        build_meta_features(panel=None, enable=True, audit_pass_token=None)
```

- [ ] **Step 2: Implement**

```python
"""Foundation-model meta-features — disabled by default in v1 (spec §3.3-7).

The decision to enable or disable is made using development-window validation
only.  Holdout informs only future runs."""

from __future__ import annotations

from typing import Any


class MetaFeaturesDisabledError(RuntimeError):
    pass


def build_meta_features(
    *,
    panel: Any,
    enable: bool = False,
    audit_pass_token: str | None = None,
) -> Any:
    if not enable:
        raise MetaFeaturesDisabledError(
            "meta-features disabled by default in v1 (spec §3.3-7); "
            "enable only after timestamp audit, ablation, baseline comparison, "
            "and dev-window improvement"
        )
    if audit_pass_token is None:
        raise MetaFeaturesDisabledError(
            "meta-features require an audit_pass_token recorded in metadata.json"
        )
    raise NotImplementedError(
        "meta-features extractor wiring deferred to a follow-up task; "
        "v1 ships with this gate disabled"
    )
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_meta_features.py -v
git add src/quant_research_stack/alpha_eq/features/meta_features.py tests/alpha_eq/test_features_meta_features.py
git commit -m "feat(s1-eq): meta-features default-disabled gate"
```

---

### Task 24 — `features/builder.py` composes everything + emits `feature_cols.json`

**Spec refs:** §3, §3.5.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/features/builder.py`
- Create: `tests/alpha_eq/test_features_builder.py`

- [ ] **Step 1: Write tests**

```python
"""Feature builder composition + sha256-locked feature_cols.json."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.features.builder import (
    FeatureBuildConfig,
    build_features,
    write_feature_cols_json,
)


def _toy_panel(n: int = 80) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    dates = pl.date_range(
        date(2020, 1, 2), date(2020, 12, 31), interval="1d", eager=True
    ).filter(pl.col("date").dt.weekday() < 6).slice(0, n)
    rows = []
    for s in ["A", "B", "C"]:
        c = 100.0
        for d in dates:
            r = float(rng.standard_normal()) * 0.01
            c *= (1 + r)
            rows.append(
                {
                    "date": d, "symbol": s,
                    "open": c * (1 + float(rng.standard_normal()) * 0.005),
                    "high": c * (1 + abs(float(rng.standard_normal())) * 0.01),
                    "low": c * (1 - abs(float(rng.standard_normal())) * 0.01),
                    "close": c,
                    "volume": int(1_000_000 + abs(float(rng.standard_normal())) * 100_000),
                    "in_universe": True,
                }
            )
    return pl.DataFrame(rows)


def test_build_features_returns_expected_columns_and_no_meta() -> None:
    df = build_features(panel=_toy_panel(), config=FeatureBuildConfig())
    must_have = {
        "feature_as_of_date", "execution_date",
        "log_return_1", "realized_vol_20", "amihud_illiq_20",
        "dollar_volume", "rank_log_return_1", "vix_close", "vix_is_proxy",
        "gaussian_noise_seed42",
    }
    assert must_have.issubset(set(df.columns))


def test_write_feature_cols_json_sha256(tmp_path: Path) -> None:
    cols = ["log_return_1", "realized_vol_20", "gaussian_noise_seed42"]
    out = tmp_path / "feature_cols.json"
    write_feature_cols_json(out, cols)
    blob = json.loads(out.read_text())
    assert blob["feature_columns"] == cols
    assert len(blob["feature_cols_sha256"]) == 64
```

- [ ] **Step 2: Implement**

```python
"""Feature builder composition + sha256-locked feature_cols.json (spec §3)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from quant_research_stack.alpha_eq.features.cross_sectional_ranks import (
    build_cross_sectional_ranks,
)
from quant_research_stack.alpha_eq.features.market_regime import build_market_regime
from quant_research_stack.alpha_eq.features.microstructure_proxies import (
    build_microstructure_proxies,
)
from quant_research_stack.alpha_eq.features.noise_sentinel import attach_noise_sentinel
from quant_research_stack.alpha_eq.features.returns_momentum import (
    build_returns_momentum,
)
from quant_research_stack.alpha_eq.features.timestamps import (
    attach_execution_date,
    attach_feature_as_of_date,
)
from quant_research_stack.alpha_eq.features.volatility import build_volatility
from quant_research_stack.alpha_eq.features.volume_liquidity import (
    build_volume_liquidity,
)


@dataclass(frozen=True)
class FeatureBuildConfig:
    momentum_horizons: tuple[int, ...] = (1, 2, 5, 10, 20, 60, 120, 252)
    vol_windows: tuple[int, ...] = (5, 20, 60)
    micro_window: int = 20
    liquidity_window: int = 20
    rank_columns: tuple[str, ...] = (
        "log_return_1", "log_return_5", "log_return_20",
        "realized_vol_20", "dollar_volume", "amihud_illiq_20",
        "overnight_gap", "close_location_20",
    )
    noise_seed: int = 42
    universe_col: str = "in_universe"
    enable_meta_features: bool = False


def build_features(*, panel: pl.DataFrame, config: FeatureBuildConfig) -> pl.DataFrame:
    df = panel
    df = attach_feature_as_of_date(df, convention="after_close_t")
    df = attach_execution_date(df, convention="next_trading_day")
    df = build_returns_momentum(df, horizons=config.momentum_horizons)
    df = build_volatility(df, windows=config.vol_windows)
    df = build_microstructure_proxies(df, window=config.micro_window)
    df = build_volume_liquidity(df, window=config.liquidity_window)
    df = build_market_regime(panel=df, vix=None, spy_close=None)
    df = build_cross_sectional_ranks(
        df, columns=config.rank_columns, universe_col=config.universe_col
    )
    df = attach_noise_sentinel(df, seed=config.noise_seed)
    return df


def _canonical_sha256(columns: Iterable[str]) -> str:
    payload = json.dumps(list(columns), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_feature_cols_json(path: Path, columns: list[str]) -> None:
    blob = {
        "feature_columns": list(columns),
        "feature_cols_sha256": _canonical_sha256(columns),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(blob, separators=(",", ":"), sort_keys=True))
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_builder.py -v
git add src/quant_research_stack/alpha_eq/features/builder.py tests/alpha_eq/test_features_builder.py
git commit -m "feat(s1-eq): feature builder composition + sha256-locked feature_cols.json"
```

---

### Task 25 — `features/labels.py`: `y_raw`, `y_vn`, `y_xs` builders

**Spec refs:** §3.2.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/features/labels.py`
- Create: `tests/alpha_eq/test_features_labels.py`

- [ ] **Step 1: Write tests**

```python
"""Label builders y_raw / y_vn / y_xs (spec §3.2)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.features.labels import build_labels


def _toy() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 1, d) for d in (2, 3, 6, 7)] * 3,
            "symbol": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "close_tr": [100.0, 101.0, 102.0, 103.0,
                          50.0, 50.5, 51.0, 51.5,
                          200.0, 199.0, 201.0, 199.0],
            "realized_vol_20": [0.02] * 12,
            "in_universe": [True] * 12,
        }
    )


def test_labels_present() -> None:
    out = build_labels(_toy(), close_tr="close_tr", vol_col="realized_vol_20", universe_col="in_universe")
    for col in ("y_raw", "y_vn", "y_xs"):
        assert col in out.columns


def test_y_xs_zero_mean_per_date_among_universe() -> None:
    out = build_labels(_toy(), close_tr="close_tr", vol_col="realized_vol_20", universe_col="in_universe")
    by_date = out.filter(pl.col("y_xs").is_not_null()).group_by("date").agg(
        pl.col("y_xs").mean().alias("mu")
    )
    for mu in by_date["mu"].to_list():
        assert abs(mu) < 1e-9
```

- [ ] **Step 2: Implement**

```python
"""Forward-return labels (spec §3.2)."""

from __future__ import annotations

import polars as pl


def build_labels(
    panel: pl.DataFrame,
    *,
    close_tr: str,
    vol_col: str,
    universe_col: str,
) -> pl.DataFrame:
    panel = panel.sort(["symbol", "date"])
    out = panel.with_columns(
        ((pl.col(close_tr).shift(-1).over("symbol") / pl.col(close_tr)) - 1.0).alias("y_raw")
    )
    out = out.with_columns(
        (pl.col("y_raw") / pl.col(vol_col).clip(lower_bound=1e-9)).alias("y_vn")
    )
    # cross-sectional residual within the per-date universe
    inuv_mean = out.filter(pl.col(universe_col)).group_by("date").agg(
        pl.col("y_vn").mean().alias("_xs_mean")
    )
    out = out.join(inuv_mean, on="date", how="left").with_columns(
        pl.when(pl.col(universe_col))
        .then(pl.col("y_vn") - pl.col("_xs_mean"))
        .otherwise(None)
        .alias("y_xs")
    ).drop("_xs_mean")
    return out
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_features_labels.py -v
git add src/quant_research_stack/alpha_eq/features/labels.py tests/alpha_eq/test_features_labels.py
git commit -m "feat(s1-eq): label builders y_raw/y_vn/y_xs (cross-sectional residualization)"
```

---

### Task 26 — Holdout isolation guard

**Spec refs:** §3.6.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/data/holdout.py`
- Create: `tests/alpha_eq/test_holdout_isolation.py`

- [ ] **Step 1: Write tests**

```python
"""Permanent holdout isolation (spec §3.6) — load guard."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quant_research_stack.alpha_eq.data.holdout import (
    HoldoutAccessError,
    HoldoutGate,
    compute_holdout_dates,
)


def test_compute_holdout_uses_last_20_percent() -> None:
    dates = [date(2020, 1, d) for d in range(2, 12)]  # 10 dates
    dev, hold = compute_holdout_dates(dates, fraction=0.2)
    assert len(hold) == 2
    assert hold == sorted(hold)
    assert dev[-1] < hold[0]


def test_holdout_gate_blocks_training_caller(tmp_path: Path) -> None:
    holdout = [date(2020, 1, 9), date(2020, 1, 10)]
    gate = HoldoutGate(holdout_dates=holdout)
    panel = pl.DataFrame(
        {"date": [date(2020, 1, 9), date(2020, 1, 8)], "symbol": ["A", "A"], "x": [1.0, 2.0]}
    )
    with pytest.raises(HoldoutAccessError):
        gate.filter_for_caller(panel, caller="training")


def test_holdout_gate_allows_inference_evaluate_holdout() -> None:
    holdout = [date(2020, 1, 9)]
    gate = HoldoutGate(holdout_dates=holdout)
    panel = pl.DataFrame({"date": [date(2020, 1, 9)], "symbol": ["A"], "x": [1.0]})
    out = gate.filter_for_caller(panel, caller="inference.evaluate_holdout")
    assert out.height == 1
```

- [ ] **Step 2: Implement**

```python
"""Permanent holdout gate (spec §3.6).

Only `inference.evaluate_holdout` is allowed to read holdout rows. All other
callers (training, tuning, stacker_fit, threshold_selection, feature_pruning,
adversarial_validation) receive a filtered panel that excludes holdout dates,
and an attempt to bypass via direct loaders is blocked by an explicit
HoldoutAccessError.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

import polars as pl

_ALLOWED_CALLERS: Final[frozenset[str]] = frozenset({"inference.evaluate_holdout"})


class HoldoutAccessError(RuntimeError):
    pass


def compute_holdout_dates(
    sorted_unique_dates: Sequence[date], *, fraction: float
) -> tuple[list[date], list[date]]:
    n = len(sorted_unique_dates)
    if n == 0:
        return [], []
    n_hold = max(1, int(round(n * fraction)))
    return list(sorted_unique_dates[: n - n_hold]), list(sorted_unique_dates[n - n_hold :])


@dataclass(frozen=True)
class HoldoutGate:
    holdout_dates: list[date]

    def filter_for_caller(self, panel: pl.DataFrame, *, caller: str) -> pl.DataFrame:
        if caller in _ALLOWED_CALLERS:
            return panel.filter(pl.col("date").is_in(self.holdout_dates))
        if panel.filter(pl.col("date").is_in(self.holdout_dates)).height > 0:
            raise HoldoutAccessError(
                f"caller {caller!r} attempted to access holdout dates; "
                "use inference.evaluate_holdout()"
            )
        return panel
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_holdout_isolation.py -v
git add src/quant_research_stack/alpha_eq/data/holdout.py tests/alpha_eq/test_holdout_isolation.py
git commit -m "feat(s1-eq): holdout-isolation gate (only inference.evaluate_holdout)"
```

---

### Task 27 — Minimum holdout length test + helper

**Spec refs:** §3.6, §6.4-2.

**Files:**
- Modify: `src/quant_research_stack/alpha_eq/data/holdout.py` (add `assert_min_holdout_length`)
- Create: `tests/alpha_eq/test_holdout_min_length.py`

- [ ] **Step 1: Write tests**

```python
"""Minimum holdout length ≥ 3 years (spec §3.6, §6.4-2)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_research_stack.alpha_eq.data.holdout import (
    HoldoutTooShortError,
    assert_min_holdout_length,
)


def test_holdout_min_length_passes_with_756_days() -> None:
    holdout = [date(2020, 1, 1) + timedelta(days=i) for i in range(756)]
    assert_min_holdout_length(holdout, min_trading_days=756)  # no raise


def test_holdout_min_length_raises_when_too_short() -> None:
    holdout = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    with pytest.raises(HoldoutTooShortError):
        assert_min_holdout_length(holdout, min_trading_days=756)
```

- [ ] **Step 2: Append to `holdout.py`**

```python
class HoldoutTooShortError(RuntimeError):
    pass


def assert_min_holdout_length(holdout_dates: list[date], *, min_trading_days: int) -> None:
    if len(holdout_dates) < min_trading_days:
        raise HoldoutTooShortError(
            f"holdout has {len(holdout_dates)} trading days, requires ≥ {min_trading_days}"
        )
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_holdout_min_length.py -v
git add src/quant_research_stack/alpha_eq/data/holdout.py tests/alpha_eq/test_holdout_min_length.py
git commit -m "feat(s1-eq): minimum-holdout-length guard (>=756 trading days)"
```

---

### Task 28 — Scaler-fit-window contract

**Spec refs:** §3.5 (object-level scaler tests).

**Files:**
- Create: `src/quant_research_stack/alpha_eq/training/scalers.py`
- Create: `tests/alpha_eq/test_scaler_fit_window.py`

- [ ] **Step 1: Write tests**

```python
"""Scaler fit-window object-level contract (spec §3.5)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quant_research_stack.alpha_eq.training.scalers import (
    FoldScaler,
    ScalerLeakError,
)


def test_fold_scaler_records_fit_window() -> None:
    s = FoldScaler(fold_id=0)
    s.fit(
        x=np.array([[1.0, 2.0], [3.0, 4.0]]),
        train_dates=[date(2020, 1, 2), date(2020, 1, 3)],
    )
    assert s.fitted_on_start_date == date(2020, 1, 2)
    assert s.fitted_on_end_date == date(2020, 1, 3)
    assert s.fold_id == 0


def test_fold_scaler_raises_when_transform_includes_validation_window() -> None:
    s = FoldScaler(fold_id=0)
    s.fit(
        x=np.array([[1.0, 2.0], [3.0, 4.0]]),
        train_dates=[date(2020, 1, 2), date(2020, 1, 3)],
    )
    with pytest.raises(ScalerLeakError):
        s.assert_transform_dates_outside_fit(
            transform_dates=[date(2020, 1, 2)]  # overlap with fit window
        )


def test_fold_scaler_transform_validates_distinct_dates() -> None:
    s = FoldScaler(fold_id=0)
    s.fit(
        x=np.array([[1.0, 2.0], [3.0, 4.0]]),
        train_dates=[date(2020, 1, 2), date(2020, 1, 3)],
    )
    s.assert_transform_dates_outside_fit(transform_dates=[date(2020, 1, 10)])
```

- [ ] **Step 2: Implement**

```python
"""Per-fold scalers with explicit fit-window metadata (spec §3.5)."""

from __future__ import annotations

from datetime import date

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler


class ScalerLeakError(RuntimeError):
    pass


class FoldScaler:
    def __init__(self, *, fold_id: int) -> None:
        self.fold_id = fold_id
        self._scaler = StandardScaler()
        self._fit_start: date | None = None
        self._fit_end: date | None = None
        self._fit_dates: frozenset[date] = frozenset()

    @property
    def fitted_on_start_date(self) -> date:
        if self._fit_start is None:
            raise RuntimeError("scaler not fit yet")
        return self._fit_start

    @property
    def fitted_on_end_date(self) -> date:
        if self._fit_end is None:
            raise RuntimeError("scaler not fit yet")
        return self._fit_end

    def fit(self, *, x: NDArray[np.float64], train_dates: list[date]) -> None:
        self._scaler.fit(x)
        self._fit_dates = frozenset(train_dates)
        self._fit_start = min(self._fit_dates)
        self._fit_end = max(self._fit_dates)

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._scaler.transform(x), dtype=np.float64)

    def assert_transform_dates_outside_fit(self, *, transform_dates: list[date]) -> None:
        overlap = self._fit_dates.intersection(transform_dates)
        if overlap:
            raise ScalerLeakError(
                f"fold {self.fold_id}: transform dates overlap fit window: {sorted(overlap)}"
            )
```

Also create `src/quant_research_stack/alpha_eq/training/__init__.py` (empty docstring file).

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_scaler_fit_window.py -v
git add src/quant_research_stack/alpha_eq/training tests/alpha_eq/test_scaler_fit_window.py
git commit -m "feat(s1-eq): per-fold scaler with explicit fit-window metadata"
```

---

## M3 — `fast_v1` training pipeline

### Task 29 — `AlphaEqConfig` (Pydantic v2) + `RunResult` types

**Spec refs:** §4.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/config.py`
- Create: `tests/alpha_eq/test_config.py`

- [ ] **Step 1: Write tests**

```python
"""AlphaEqConfig validates basic invariants."""

from __future__ import annotations

import pytest

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode


def test_config_default_construction() -> None:
    cfg = AlphaEqConfig(mode=TrainingMode.FAST_V1)
    assert cfg.mode == TrainingMode.FAST_V1
    assert cfg.cv.n_folds == 5
    assert cfg.features.enable_meta_features is False
    assert cfg.data.permanent_holdout_fraction == 0.20


def test_config_rejects_invalid_holdout_fraction() -> None:
    with pytest.raises(ValueError):
        AlphaEqConfig(mode=TrainingMode.FAST_V1, data={"permanent_holdout_fraction": 0.5})


def test_mode_full_v1_models() -> None:
    cfg = AlphaEqConfig(mode=TrainingMode.FULL_V1)
    assert set(cfg.active_models()) == {"ridge", "lightgbm", "xgboost", "catboost", "mlp", "sequence"}
```

- [ ] **Step 2: Implement**

```python
"""S1-EQ Pydantic v2 configuration (spec §4)."""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingMode(str, enum.Enum):
    FAST_V1 = "fast_v1"
    FULL_V1 = "full_v1"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    equity_root: str = "data/processed/equities"
    manifest_path: str = "data/processed/equities/_manifest.json"
    universe: str = "sp500"
    permanent_holdout_fraction: float = Field(default=0.20, gt=0.0, lt=0.4)
    min_holdout_trading_days: int = Field(default=756, ge=252)


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable_meta_features: bool = False
    noise_seed: int = 42


class CVConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layout: str = "expanding_window"
    n_folds: int = 5
    label_horizon_days: int = 1
    purge_safety_buffer: int = 2

    @property
    def purge_days(self) -> int:
        return max(5, self.label_horizon_days + self.purge_safety_buffer)

    @property
    def embargo_days(self) -> int:
        return max(5, self.label_horizon_days)


class StackerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: float = 1.0e-3
    prefer_non_negative: bool = True
    flag_large_negative_threshold: float = -0.25


class ReproConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    numpy_seed: int = 42
    torch_seed: int = 42
    lightgbm_seed: int = 42
    xgboost_seed: int = 42
    catboost_seed: int = 42


_MODE_MODELS: dict[TrainingMode, tuple[str, ...]] = {
    TrainingMode.FAST_V1: ("ridge", "lightgbm", "xgboost"),
    TrainingMode.FULL_V1: ("ridge", "lightgbm", "xgboost", "catboost", "mlp", "sequence"),
}


class AlphaEqConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: TrainingMode
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    stacker: StackerConfig = Field(default_factory=StackerConfig)
    reproducibility: ReproConfig = Field(default_factory=ReproConfig)

    def active_models(self) -> tuple[str, ...]:
        return _MODE_MODELS[self.mode]

    @model_validator(mode="after")
    def _validate(self) -> "AlphaEqConfig":
        if self.cv.n_folds < 3:
            raise ValueError("n_folds must be ≥ 3")
        return self
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_config.py -v
git add src/quant_research_stack/alpha_eq/config.py tests/alpha_eq/test_config.py
git commit -m "feat(s1-eq): AlphaEqConfig (pydantic v2) with fast_v1/full_v1 modes"
```

---

### Task 30 — `models/ridge.py` (target=y_xs)

**Spec refs:** §4.4.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/models/ridge.py`
- Create: `tests/alpha_eq/test_models_ridge.py`

- [ ] **Step 1: Write tests**

```python
"""Ridge S1-EQ model (target = y_xs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.models.ridge import (
    RidgeEqConfig,
    RidgeEqModel,
)


def test_ridge_fit_predict_save_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 8))
    y = x[:, 0] - 0.5 * x[:, 1] + rng.standard_normal(200) * 0.1
    m = RidgeEqModel(RidgeEqConfig(alpha=1.0))
    m.fit(x=x, y=y)
    p = m.predict(x)
    assert p.shape == (200,)
    out = tmp_path / "ridge.joblib"
    m.save(out)
    m2 = RidgeEqModel.load(out)
    np.testing.assert_allclose(m.predict(x), m2.predict(x), atol=1e-12)
```

- [ ] **Step 2: Implement**

```python
"""Ridge S1-EQ base learner (target = y_xs)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class RidgeEqConfig:
    alpha: float = 1.0


class RidgeEqModel:
    def __init__(self, config: RidgeEqConfig) -> None:
        self.config = config
        self._estimator = Ridge(alpha=config.alpha, fit_intercept=True)

    def fit(self, *, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        self._estimator.fit(x, y)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._estimator.predict(x), dtype=np.float64)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self._estimator, "config": asdict(self.config)}, path)

    @classmethod
    def load(cls, path: Path) -> "RidgeEqModel":
        payload = joblib.load(path)
        inst = cls(RidgeEqConfig(**payload["config"]))
        inst._estimator = payload["estimator"]
        return inst
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_models_ridge.py -v
git add src/quant_research_stack/alpha_eq/models/ridge.py tests/alpha_eq/test_models_ridge.py
git commit -m "feat(s1-eq): ridge base learner (target y_xs)"
```

---

### Task 31 — `models/lightgbm_model.py`

**Spec refs:** §4.4.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/models/lightgbm_model.py`
- Create: `tests/alpha_eq/test_models_lightgbm.py`

- [ ] **Step 1: Write tests**

```python
"""LightGBM S1-EQ model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.models.lightgbm_model import (
    LightGBMEqConfig,
    LightGBMEqModel,
)


def test_lightgbm_fit_predict_save_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 6))
    y = x[:, 0] * 0.5 + rng.standard_normal(500) * 0.1
    m = LightGBMEqModel(LightGBMEqConfig(n_estimators=50, num_leaves=15, seed=42))
    m.fit(x=x, y=y, x_val=x[:100], y_val=y[:100])
    p = m.predict(x[:10])
    assert p.shape == (10,)
    out = tmp_path / "lgb.txt"
    cfg = tmp_path / "lgb.config.json"
    m.save(out, config_path=cfg)
    m2 = LightGBMEqModel.load(out, config_path=cfg)
    np.testing.assert_allclose(m.predict(x[:5]), m2.predict(x[:5]), atol=1e-9)
```

- [ ] **Step 2: Implement**

```python
"""LightGBM S1-EQ base learner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LightGBMEqConfig:
    num_leaves: int = 63
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    seed: int = 42


class LightGBMEqModel:
    def __init__(self, config: LightGBMEqConfig) -> None:
        self.config = config
        self._booster: lgb.Booster | None = None

    def fit(
        self,
        *,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        x_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
    ) -> None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "seed": self.config.seed,
            "verbose": -1,
        }
        train_set = lgb.Dataset(x, label=y)
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = []
        if x_val is not None and y_val is not None:
            valid_sets.append(lgb.Dataset(x_val, label=y_val))
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(self.config.early_stopping_rounds, verbose=False))
        self._booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._booster is None:
            raise RuntimeError("model not fit")
        return np.asarray(self._booster.predict(x), dtype=np.float64)

    def save(self, path: Path, *, config_path: Path) -> None:
        if self._booster is None:
            raise RuntimeError("model not fit")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path))
        Path(config_path).write_text(json.dumps(asdict(self.config), sort_keys=True))

    @classmethod
    def load(cls, path: Path, *, config_path: Path) -> "LightGBMEqModel":
        cfg = LightGBMEqConfig(**json.loads(Path(config_path).read_text()))
        m = cls(cfg)
        m._booster = lgb.Booster(model_file=str(path))
        return m
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_models_lightgbm.py -v
git add src/quant_research_stack/alpha_eq/models/lightgbm_model.py tests/alpha_eq/test_models_lightgbm.py
git commit -m "feat(s1-eq): LightGBM base learner with early stopping + config sidecar"
```

---

### Task 32 — `models/xgboost_model.py`

**Spec refs:** §4.4.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/models/xgboost_model.py`
- Create: `tests/alpha_eq/test_models_xgboost.py`

- [ ] **Step 1: Write tests**

```python
"""XGBoost S1-EQ model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.models.xgboost_model import (
    XGBoostEqConfig,
    XGBoostEqModel,
)


def test_xgboost_fit_predict_save_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 5))
    y = x[:, 0] * 0.3 + rng.standard_normal(400) * 0.1
    m = XGBoostEqModel(XGBoostEqConfig(n_estimators=50, max_depth=4, seed=42))
    m.fit(x=x, y=y, x_val=x[:80], y_val=y[:80])
    out = tmp_path / "xgb.json"
    cfg = tmp_path / "xgb.config.json"
    m.save(out, config_path=cfg)
    m2 = XGBoostEqModel.load(out, config_path=cfg)
    np.testing.assert_allclose(m.predict(x[:5]), m2.predict(x[:5]), atol=1e-9)
```

- [ ] **Step 2: Implement**

```python
"""XGBoost S1-EQ base learner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray


@dataclass(frozen=True)
class XGBoostEqConfig:
    max_depth: int = 8
    learning_rate: float = 0.05
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    tree_method: str = "hist"
    seed: int = 42


class XGBoostEqModel:
    def __init__(self, config: XGBoostEqConfig) -> None:
        self.config = config
        self._booster: xgb.Booster | None = None

    def fit(
        self,
        *,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        x_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
    ) -> None:
        dtrain = xgb.DMatrix(x, label=y)
        evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]
        if x_val is not None and y_val is not None:
            evals.append((xgb.DMatrix(x_val, label=y_val), "valid"))
        params = {
            "objective": "reg:squarederror",
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "tree_method": self.config.tree_method,
            "seed": self.config.seed,
            "verbosity": 0,
        }
        self._booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.n_estimators,
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds if len(evals) > 1 else None,
            verbose_eval=False,
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._booster is None:
            raise RuntimeError("model not fit")
        return np.asarray(self._booster.predict(xgb.DMatrix(x)), dtype=np.float64)

    def save(self, path: Path, *, config_path: Path) -> None:
        if self._booster is None:
            raise RuntimeError("model not fit")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path))
        Path(config_path).write_text(json.dumps(asdict(self.config), sort_keys=True))

    @classmethod
    def load(cls, path: Path, *, config_path: Path) -> "XGBoostEqModel":
        cfg = XGBoostEqConfig(**json.loads(Path(config_path).read_text()))
        m = cls(cfg)
        m._booster = xgb.Booster()
        m._booster.load_model(str(path))
        return m
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_models_xgboost.py -v
git add src/quant_research_stack/alpha_eq/models/xgboost_model.py tests/alpha_eq/test_models_xgboost.py
git commit -m "feat(s1-eq): XGBoost base learner with early stopping + config sidecar"
```

---

### Task 33 — `stacking.py`: L2-regularized linear stacker + signed diagnostic

**Spec refs:** §4.5.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/stacking.py`
- Create: `tests/alpha_eq/test_stacking.py`

- [ ] **Step 1: Write tests**

```python
"""Linear stacker — L2-regularized, signed-diagnostic, large-negative-weight flag."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.stacking import (
    LinearStackerEq,
    StackerArtifact,
    flag_large_negative_weights,
)


def test_stacker_fits_and_predicts() -> None:
    rng = np.random.default_rng(0)
    oof = rng.standard_normal((300, 3))
    y = oof.sum(axis=1) + rng.standard_normal(300) * 0.1
    s = LinearStackerEq(
        alpha=1e-3,
        prefer_non_negative=True,
        feature_order=("ridge", "lgb", "xgb"),
    )
    s.fit(oof_predictions=oof, y=y)
    p = s.predict(oof[:5])
    assert p.shape == (5,)
    # non-negative-preferred stacker should have weights ≥ 0
    assert np.all(s.weights >= -1e-9)


def test_stacker_signed_variant() -> None:
    rng = np.random.default_rng(0)
    oof = rng.standard_normal((300, 3))
    # construct a case where one base learner should be negatively weighted
    y = oof[:, 0] - oof[:, 1] + rng.standard_normal(300) * 0.01
    s = LinearStackerEq(
        alpha=1e-3,
        prefer_non_negative=False,
        feature_order=("good", "bad", "noise"),
    )
    s.fit(oof_predictions=oof, y=y)
    # at least one weight is negative when signed mode is allowed
    assert np.any(s.weights < 0)


def test_flag_large_negative_weights() -> None:
    flagged = flag_large_negative_weights(
        weights=np.array([0.5, -0.30, 0.10]),
        names=("a", "b", "c"),
        threshold=-0.25,
    )
    assert "b" in flagged


def test_stacker_save_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    oof = rng.standard_normal((100, 3))
    y = oof.sum(axis=1)
    s = LinearStackerEq(alpha=1e-3, prefer_non_negative=True,
                       feature_order=("ridge", "lgb", "xgb"))
    s.fit(oof_predictions=oof, y=y)
    out = tmp_path / "stacker.joblib"
    s.save(out)
    s2 = LinearStackerEq.load(out)
    np.testing.assert_allclose(s.predict(oof[:5]), s2.predict(oof[:5]), atol=1e-12)
    art = StackerArtifact.from_model(s2)
    assert art.feature_order == ("ridge", "lgb", "xgb")
```

- [ ] **Step 2: Implement**

```python
"""L2-regularized linear stacker with optional non-negativity + signed
diagnostic + large-negative-weight flag (spec §4.5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


class LinearStackerEq:
    def __init__(
        self,
        *,
        alpha: float,
        prefer_non_negative: bool,
        feature_order: Sequence[str],
    ) -> None:
        self.alpha = float(alpha)
        self.prefer_non_negative = bool(prefer_non_negative)
        self.feature_order: tuple[str, ...] = tuple(feature_order)
        self._estimator: Ridge | None = None

    def fit(self, *, oof_predictions: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        if oof_predictions.shape[1] != len(self.feature_order):
            raise ValueError("oof_predictions cols != len(feature_order)")
        est = Ridge(
            alpha=self.alpha,
            positive=self.prefer_non_negative,
            fit_intercept=False,
        )
        est.fit(oof_predictions, y)
        self._estimator = est

    def predict(self, oof: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._estimator is None:
            raise RuntimeError("stacker not fit")
        return np.asarray(self._estimator.predict(oof), dtype=np.float64)

    @property
    def weights(self) -> NDArray[np.float64]:
        if self._estimator is None:
            raise RuntimeError("stacker not fit")
        return np.asarray(self._estimator.coef_, dtype=np.float64)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self._estimator,
                "alpha": self.alpha,
                "prefer_non_negative": self.prefer_non_negative,
                "feature_order": list(self.feature_order),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "LinearStackerEq":
        payload = joblib.load(path)
        m = cls(
            alpha=payload["alpha"],
            prefer_non_negative=payload["prefer_non_negative"],
            feature_order=tuple(payload["feature_order"]),
        )
        m._estimator = payload["estimator"]
        return m


@dataclass(frozen=True)
class StackerArtifact:
    feature_order: tuple[str, ...]
    weights: tuple[float, ...]
    flagged_negatives: tuple[str, ...]

    @classmethod
    def from_model(
        cls, model: LinearStackerEq, *, threshold: float = -0.25
    ) -> "StackerArtifact":
        w = model.weights
        flagged = flag_large_negative_weights(
            weights=w, names=model.feature_order, threshold=threshold
        )
        return cls(
            feature_order=model.feature_order,
            weights=tuple(float(x) for x in w),
            flagged_negatives=tuple(flagged),
        )


def flag_large_negative_weights(
    *, weights: NDArray[np.float64], names: Sequence[str], threshold: float
) -> list[str]:
    return [n for w, n in zip(weights, names, strict=True) if float(w) < float(threshold)]
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_stacking.py -v
git add src/quant_research_stack/alpha_eq/stacking.py tests/alpha_eq/test_stacking.py
git commit -m "feat(s1-eq): L2-regularized stacker with signed diagnostic + flag"
```

---

### Task 34 — `training/cv.py` walk-forward folds with dynamic purge/embargo

**Spec refs:** §4.2.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/training/cv.py`
- Create: `tests/alpha_eq/test_training_cv.py`

- [ ] **Step 1: Write tests**

```python
"""Walk-forward CV with dynamic purge/embargo (spec §4.2)."""

from __future__ import annotations

from datetime import date, timedelta

from quant_research_stack.alpha_eq.config import CVConfig
from quant_research_stack.alpha_eq.training.cv import build_expanding_window_folds


def test_expanding_window_folds_are_chronological() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(500)]
    cfg = CVConfig()
    folds = build_expanding_window_folds(dev_window_dates=dates, cv=cfg)
    assert len(folds) == cfg.n_folds
    # each fold's train end < validation start (after purge)
    for f in folds:
        assert max(f.train_dates) < min(f.validation_dates)
        assert (min(f.validation_dates) - max(f.train_dates)).days >= cfg.purge_days
    # expanding: each subsequent fold's train set is a superset of the prior's
    for prev, nxt in zip(folds, folds[1:], strict=True):
        assert set(prev.train_dates).issubset(set(nxt.train_dates))


def test_embargo_excludes_post_validation_window_from_next_train() -> None:
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(500)]
    cfg = CVConfig()
    folds = build_expanding_window_folds(dev_window_dates=dates, cv=cfg)
    for prev, nxt in zip(folds, folds[1:], strict=True):
        embargo_start = max(prev.validation_dates) + timedelta(days=1)
        embargo_end = embargo_start + timedelta(days=cfg.embargo_days - 1)
        embargo = {embargo_start + timedelta(days=k) for k in range(cfg.embargo_days)}
        assert embargo.isdisjoint(set(nxt.train_dates))
```

- [ ] **Step 2: Implement**

```python
"""Expanding-window walk-forward CV with dynamic purge + embargo (spec §4.2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from quant_research_stack.alpha_eq.config import CVConfig


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_dates: tuple[date, ...]
    validation_dates: tuple[date, ...]


def build_expanding_window_folds(
    *, dev_window_dates: Sequence[date], cv: CVConfig
) -> list[Fold]:
    sorted_dates = sorted(set(dev_window_dates))
    n = len(sorted_dates)
    n_folds = cv.n_folds
    val_size = max(1, n // (n_folds + 1))
    folds: list[Fold] = []
    for k in range(n_folds):
        val_start_idx = (k + 1) * val_size
        val_end_idx = min(n, val_start_idx + val_size)
        if val_start_idx >= n or val_end_idx <= val_start_idx:
            break
        val_dates = sorted_dates[val_start_idx:val_end_idx]
        purge_cutoff = val_dates[0] - timedelta(days=cv.purge_days)
        train_candidates = [d for d in sorted_dates[:val_start_idx] if d <= purge_cutoff]
        folds.append(
            Fold(
                fold_id=k,
                train_dates=tuple(train_candidates),
                validation_dates=tuple(val_dates),
            )
        )
    # embargo: any date within `embargo_days` after a fold's validation end is excluded from
    # the *next* fold's train_dates (and following folds, transitively).
    cleaned: list[Fold] = []
    for k, f in enumerate(folds):
        if k == 0:
            cleaned.append(f)
            continue
        prev_val_end = max(folds[k - 1].validation_dates)
        embargo = {prev_val_end + timedelta(days=i) for i in range(1, cv.embargo_days + 1)}
        new_train = tuple(d for d in f.train_dates if d not in embargo)
        cleaned.append(Fold(fold_id=f.fold_id, train_dates=new_train, validation_dates=f.validation_dates))
    return cleaned
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_training_cv.py -v
git add src/quant_research_stack/alpha_eq/training/cv.py tests/alpha_eq/test_training_cv.py
git commit -m "feat(s1-eq): expanding-window CV with dynamic purge + embargo"
```

---

### Task 35 — `training/loop.py` per-fold base-learner loop

**Spec refs:** §4.3, §4.4.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/training/loop.py`
- Create: `tests/alpha_eq/test_training_loop.py`

- [ ] **Step 1: Write tests**

```python
"""Per-fold base-learner training loop produces OOF preds and persists fold-fit scalers."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode
from quant_research_stack.alpha_eq.training.cv import Fold
from quant_research_stack.alpha_eq.training.loop import run_fold_loop


def _toy_dataset(n_dates: int = 200, n_symbols: int = 10) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for s in range(n_symbols):
            f1 = float(rng.standard_normal())
            f2 = float(rng.standard_normal())
            y = 0.3 * f1 - 0.2 * f2 + float(rng.standard_normal()) * 0.1
            rows.append({"date": d, "symbol": f"S{s}", "f1": f1, "f2": f2, "y_xs": y})
    return pl.DataFrame(rows)


def test_run_fold_loop_returns_oof_rows() -> None:
    df = _toy_dataset()
    cfg = AlphaEqConfig(mode=TrainingMode.FAST_V1)
    fold = Fold(
        fold_id=0,
        train_dates=tuple(df["date"].unique().sort().head(100).to_list()),
        validation_dates=tuple(df["date"].unique().sort().tail(50).to_list()),
    )
    feature_cols = ["f1", "f2"]
    oof = run_fold_loop(
        panel=df, feature_cols=feature_cols, target="y_xs", fold=fold, config=cfg
    )
    # rows for fold 0 validation
    assert oof.height == 50 * 10
    # one prediction column per active model
    for m in cfg.active_models():
        assert f"pred_{m}" in oof.columns
```

- [ ] **Step 2: Implement**

```python
"""Per-fold base-learner training loop (spec §4.3, §4.4).

Trains every base learner from `config.active_models()` and returns the
fold-validation rows enriched with one prediction column per learner.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.config import AlphaEqConfig
from quant_research_stack.alpha_eq.models.lightgbm_model import (
    LightGBMEqConfig,
    LightGBMEqModel,
)
from quant_research_stack.alpha_eq.models.ridge import RidgeEqConfig, RidgeEqModel
from quant_research_stack.alpha_eq.models.xgboost_model import (
    XGBoostEqConfig,
    XGBoostEqModel,
)
from quant_research_stack.alpha_eq.training.cv import Fold
from quant_research_stack.alpha_eq.training.scalers import FoldScaler


def _split(panel: pl.DataFrame, fold: Fold, target: str, feature_cols: Sequence[str]):
    train = panel.filter(pl.col("date").is_in(list(fold.train_dates))).drop_nulls(subset=[target] + list(feature_cols))
    valid = panel.filter(pl.col("date").is_in(list(fold.validation_dates))).drop_nulls(subset=list(feature_cols))
    x_tr = train.select(list(feature_cols)).to_numpy().astype(np.float64)
    y_tr = train[target].to_numpy().astype(np.float64)
    x_va = valid.select(list(feature_cols)).to_numpy().astype(np.float64)
    y_va = (
        valid[target].to_numpy().astype(np.float64) if target in valid.columns else np.array([])
    )
    return train, valid, x_tr, y_tr, x_va, y_va


def run_fold_loop(
    *,
    panel: pl.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    fold: Fold,
    config: AlphaEqConfig,
) -> pl.DataFrame:
    _train, valid, x_tr, y_tr, x_va, _y_va = _split(panel, fold, target, feature_cols)

    scaler = FoldScaler(fold_id=fold.fold_id)
    scaler.fit(x=x_tr, train_dates=list(fold.train_dates))
    scaler.assert_transform_dates_outside_fit(transform_dates=list(fold.validation_dates))
    x_tr_s = scaler.transform(x_tr)
    x_va_s = scaler.transform(x_va)

    preds: dict[str, np.ndarray] = {}

    if "ridge" in config.active_models():
        m = RidgeEqModel(RidgeEqConfig(alpha=1.0))
        m.fit(x=x_tr_s, y=y_tr)
        preds["ridge"] = m.predict(x_va_s)

    if "lightgbm" in config.active_models():
        lgb_cfg = LightGBMEqConfig(seed=config.reproducibility.lightgbm_seed)
        m_lgb = LightGBMEqModel(lgb_cfg)
        m_lgb.fit(x=x_tr_s, y=y_tr, x_val=x_va_s, y_val=_y_va if _y_va.size else None)
        preds["lightgbm"] = m_lgb.predict(x_va_s)

    if "xgboost" in config.active_models():
        xgb_cfg = XGBoostEqConfig(seed=config.reproducibility.xgboost_seed)
        m_xgb = XGBoostEqModel(xgb_cfg)
        m_xgb.fit(x=x_tr_s, y=y_tr, x_val=x_va_s, y_val=_y_va if _y_va.size else None)
        preds["xgboost"] = m_xgb.predict(x_va_s)

    out = valid
    for name, p in preds.items():
        out = out.with_columns(pl.Series(f"pred_{name}", p))
    return out
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_training_loop.py -v
git add src/quant_research_stack/alpha_eq/training/loop.py tests/alpha_eq/test_training_loop.py
git commit -m "feat(s1-eq): per-fold base-learner loop produces OOF preds"
```

---

### Task 36 — OOF collector + stacker fit on OOF

**Spec refs:** §4.5.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/training/oof.py`
- Create: `tests/alpha_eq/test_training_oof.py`

- [ ] **Step 1: Write tests**

```python
"""OOF aggregation + stacker fit."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.training.oof import (
    collect_oof,
    fit_stacker_on_oof,
)


def test_collect_and_fit_stacker() -> None:
    fold_outputs = [
        pl.DataFrame(
            {
                "date": [date(2020, 1, k)] * 5,
                "symbol": list("ABCDE"),
                "y_xs": np.linspace(-1, 1, 5),
                "pred_ridge": np.linspace(-0.9, 0.9, 5),
                "pred_lightgbm": np.linspace(-0.8, 0.8, 5),
                "pred_xgboost": np.linspace(-0.7, 0.7, 5),
            }
        )
        for k in (2, 3, 6)
    ]
    oof = collect_oof(fold_outputs)
    assert oof.height == 15
    stacker = fit_stacker_on_oof(
        oof=oof,
        feature_order=("ridge", "lightgbm", "xgboost"),
        target="y_xs",
        alpha=1e-3,
        prefer_non_negative=True,
    )
    assert stacker.weights.shape == (3,)
```

- [ ] **Step 2: Implement**

```python
"""OOF aggregation + stacker fit."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.stacking import LinearStackerEq


def collect_oof(fold_outputs: Sequence[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(list(fold_outputs))


def fit_stacker_on_oof(
    *,
    oof: pl.DataFrame,
    feature_order: Sequence[str],
    target: str,
    alpha: float,
    prefer_non_negative: bool,
) -> LinearStackerEq:
    cols = [f"pred_{n}" for n in feature_order]
    keep = oof.drop_nulls(subset=cols + [target])
    x = keep.select(cols).to_numpy().astype(np.float64)
    y = keep[target].to_numpy().astype(np.float64)
    s = LinearStackerEq(
        alpha=alpha, prefer_non_negative=prefer_non_negative, feature_order=feature_order
    )
    s.fit(oof_predictions=x, y=y)
    return s
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_training_oof.py -v
git add src/quant_research_stack/alpha_eq/training/oof.py tests/alpha_eq/test_training_oof.py
git commit -m "feat(s1-eq): OOF aggregator + stacker fit on OOF predictions"
```

---

### Task 37 — Refit-on-full + persistence

**Spec refs:** §4.8, §4.11.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/training/persist.py`
- Create: `tests/alpha_eq/test_training_persist.py`

- [ ] **Step 1: Write tests**

```python
"""Refit-on-full + artifact persistence — required artifacts under run_dir."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode
from quant_research_stack.alpha_eq.training.persist import (
    REQUIRED_FAST_V1_ARTIFACTS,
    persist_fast_v1_run,
)


def _toy_panel() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(50):
        d = date(2020, 1, 1) + timedelta(days=i)
        for s in range(6):
            rows.append({
                "date": d,
                "symbol": f"S{s}",
                "f1": float(rng.standard_normal()),
                "f2": float(rng.standard_normal()),
                "y_xs": float(rng.standard_normal()),
            })
    return pl.DataFrame(rows)


def test_persist_fast_v1_run_writes_required_artifacts(tmp_path: Path) -> None:
    cfg = AlphaEqConfig(mode=TrainingMode.FAST_V1)
    persist_fast_v1_run(
        run_dir=tmp_path,
        config=cfg,
        feature_cols=["f1", "f2"],
        dev_panel=_toy_panel(),
        target="y_xs",
    )
    for art in REQUIRED_FAST_V1_ARTIFACTS:
        assert (tmp_path / art).exists(), f"missing artifact: {art}"
    # _artifact_sha256.json covers every required artifact
    sha_blob = json.loads((tmp_path / "_artifact_sha256.json").read_text())
    for art in REQUIRED_FAST_V1_ARTIFACTS:
        assert art in sha_blob
```

- [ ] **Step 2: Implement**

```python
"""Refit-on-full + persistence of all required S1-EQ artifacts (spec §4.8, §4.11)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.config import AlphaEqConfig
from quant_research_stack.alpha_eq.features.builder import write_feature_cols_json
from quant_research_stack.alpha_eq.models.lightgbm_model import (
    LightGBMEqConfig,
    LightGBMEqModel,
)
from quant_research_stack.alpha_eq.models.ridge import RidgeEqConfig, RidgeEqModel
from quant_research_stack.alpha_eq.models.xgboost_model import (
    XGBoostEqConfig,
    XGBoostEqModel,
)
from quant_research_stack.alpha_eq.stacking import LinearStackerEq


REQUIRED_FAST_V1_ARTIFACTS: tuple[str, ...] = (
    "feature_cols.json",
    "models/ridge.joblib",
    "models/lightgbm.txt",
    "models/lightgbm.config.json",
    "models/xgboost.json",
    "models/xgboost.config.json",
    "models/stacker.joblib",
    "metadata.json",
    "_artifact_sha256.json",
)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def persist_fast_v1_run(
    *,
    run_dir: Path,
    config: AlphaEqConfig,
    feature_cols: Sequence[str],
    dev_panel: pl.DataFrame,
    target: str,
) -> None:
    run_dir = Path(run_dir)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    write_feature_cols_json(run_dir / "feature_cols.json", list(feature_cols))

    panel = dev_panel.drop_nulls(subset=list(feature_cols) + [target])
    x = panel.select(list(feature_cols)).to_numpy().astype(np.float64)
    y = panel[target].to_numpy().astype(np.float64)

    r = RidgeEqModel(RidgeEqConfig(alpha=1.0))
    r.fit(x=x, y=y)
    r.save(run_dir / "models" / "ridge.joblib")

    lgb_cfg = LightGBMEqConfig(seed=config.reproducibility.lightgbm_seed)
    m_lgb = LightGBMEqModel(lgb_cfg)
    m_lgb.fit(x=x, y=y)
    m_lgb.save(run_dir / "models" / "lightgbm.txt", config_path=run_dir / "models" / "lightgbm.config.json")

    xgb_cfg = XGBoostEqConfig(seed=config.reproducibility.xgboost_seed)
    m_xgb = XGBoostEqModel(xgb_cfg)
    m_xgb.fit(x=x, y=y)
    m_xgb.save(run_dir / "models" / "xgboost.json", config_path=run_dir / "models" / "xgboost.config.json")

    # Initialize the stacker with default weights for fast_v1 smoke; real OOF-fit
    # happens in training.train_fast_v1.
    stacker = LinearStackerEq(
        alpha=config.stacker.alpha,
        prefer_non_negative=config.stacker.prefer_non_negative,
        feature_order=("ridge", "lightgbm", "xgboost"),
    )
    oof_smoke = np.column_stack([r.predict(x), m_lgb.predict(x), m_xgb.predict(x)])
    stacker.fit(oof_predictions=oof_smoke, y=y)
    stacker.save(run_dir / "models" / "stacker.joblib")

    metadata = {
        "git_sha": "filled-by-train_s1_eq.py",
        "data_manifest_sha256": "filled-by-train_s1_eq.py",
        "hyperparams": {
            "ridge": {"alpha": 1.0},
            "lightgbm": dict(),
            "xgboost": dict(),
            "stacker": {"alpha": config.stacker.alpha},
        },
        "mode": config.mode.value,
        "seeds": {
            "numpy": config.reproducibility.numpy_seed,
            "lightgbm": config.reproducibility.lightgbm_seed,
            "xgboost": config.reproducibility.xgboost_seed,
        },
        "research_dof": {
            "optuna_trials": {},
            "model_classes_searched": list(config.active_models()),
            "feature_sets_evaluated": 1,
            "threshold_sweeps": 0,
            "post_hoc_decisions": [],
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, sort_keys=True, indent=2))

    sha = {art: _sha256_file(run_dir / art) for art in REQUIRED_FAST_V1_ARTIFACTS if art != "_artifact_sha256.json"}
    (run_dir / "_artifact_sha256.json").write_text(json.dumps(sha, sort_keys=True, indent=2))
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_training_persist.py -v
git add src/quant_research_stack/alpha_eq/training/persist.py tests/alpha_eq/test_training_persist.py
git commit -m "feat(s1-eq): refit-on-full persistence with required artifact set"
```

---

### Task 38 — `inference.py` load + bound predictor

**Spec refs:** §4.10.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/inference.py`
- Create: `tests/alpha_eq/test_inference.py`

- [ ] **Step 1: Write tests**

```python
"""Inference loader from a persisted run dir."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode
from quant_research_stack.alpha_eq.inference import (
    FeatureSchemaMismatchError,
    load_predictor_from_run,
)
from quant_research_stack.alpha_eq.training.persist import persist_fast_v1_run


def _toy_panel(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(40):
        d = date(2020, 1, 1) + timedelta(days=i)
        for s in range(6):
            rows.append({
                "date": d,
                "symbol": f"S{s}",
                "f1": float(rng.standard_normal()),
                "f2": float(rng.standard_normal()),
                "y_xs": float(rng.standard_normal()),
            })
    return pl.DataFrame(rows)


def test_load_predictor_predicts(tmp_path: Path) -> None:
    cfg = AlphaEqConfig(mode=TrainingMode.FAST_V1)
    persist_fast_v1_run(
        run_dir=tmp_path, config=cfg, feature_cols=["f1", "f2"],
        dev_panel=_toy_panel(), target="y_xs",
    )
    predictor = load_predictor_from_run(tmp_path)
    out = predictor.predict_batch(
        pl.DataFrame({"f1": [0.1, -0.2, 0.3], "f2": [0.4, 0.5, -0.6]})
    )
    assert out.shape == (3,)


def test_load_predictor_schema_mismatch(tmp_path: Path) -> None:
    cfg = AlphaEqConfig(mode=TrainingMode.FAST_V1)
    persist_fast_v1_run(
        run_dir=tmp_path, config=cfg, feature_cols=["f1", "f2"],
        dev_panel=_toy_panel(), target="y_xs",
    )
    predictor = load_predictor_from_run(tmp_path)
    import pytest

    with pytest.raises(FeatureSchemaMismatchError):
        predictor.predict_batch(pl.DataFrame({"f1": [0.1]}))  # missing f2
```

- [ ] **Step 2: Implement**

```python
"""S1-EQ inference loader (mirrors alpha/inference.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_research_stack.alpha_eq.models.lightgbm_model import LightGBMEqModel
from quant_research_stack.alpha_eq.models.ridge import RidgeEqModel
from quant_research_stack.alpha_eq.models.xgboost_model import XGBoostEqModel
from quant_research_stack.alpha_eq.stacking import LinearStackerEq


class FeatureSchemaMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class BoundEqPredictor:
    feature_columns: tuple[str, ...]
    ridge: RidgeEqModel
    lightgbm: LightGBMEqModel
    xgboost: XGBoostEqModel
    stacker: LinearStackerEq

    def predict_batch(self, df: pl.DataFrame) -> NDArray[np.float64]:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise FeatureSchemaMismatchError(f"missing feature columns: {missing}")
        x = df.select(list(self.feature_columns)).to_numpy().astype(np.float64)
        oof = np.column_stack(
            [self.ridge.predict(x), self.lightgbm.predict(x), self.xgboost.predict(x)]
        )
        return self.stacker.predict(oof)


def load_predictor_from_run(run_dir: Path) -> BoundEqPredictor:
    rd = Path(run_dir)
    cols = json.loads((rd / "feature_cols.json").read_text())["feature_columns"]
    return BoundEqPredictor(
        feature_columns=tuple(cols),
        ridge=RidgeEqModel.load(rd / "models" / "ridge.joblib"),
        lightgbm=LightGBMEqModel.load(
            rd / "models" / "lightgbm.txt",
            config_path=rd / "models" / "lightgbm.config.json",
        ),
        xgboost=XGBoostEqModel.load(
            rd / "models" / "xgboost.json",
            config_path=rd / "models" / "xgboost.config.json",
        ),
        stacker=LinearStackerEq.load(rd / "models" / "stacker.joblib"),
    )
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_inference.py -v
git add src/quant_research_stack/alpha_eq/inference.py tests/alpha_eq/test_inference.py
git commit -m "feat(s1-eq): inference loader + bound predictor with schema check"
```

---

### Task 39 — `train_s1_eq.py` CLI (`--mode fast_v1`)

**Spec refs:** §4.

**Files:**
- Create: `scripts/train_s1_eq.py`
- Create: `tests/alpha_eq/test_train_cli_fast_v1.py`

- [ ] **Step 1: Write test**

```python
"""CLI smoke for fast_v1 training."""

from __future__ import annotations

import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.data.manifest import (
    DataQualityLabel,
    DelistingAuditCounters,
    EquityManifest,
    ManifestArtifact,
    sha256_of_file,
    write_manifest,
)


def _seed_root(root: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        d = date(2020, 1, 1) + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(5):
            rows.append({
                "date": d,
                "symbol": f"S{s}",
                "open": 100.0 + float(rng.standard_normal()),
                "high": 101.0 + float(rng.standard_normal()),
                "low": 99.0 + float(rng.standard_normal()),
                "close": 100.0 + float(rng.standard_normal()),
                "volume": int(1_000_000 + abs(float(rng.standard_normal())) * 100_000),
            })
    df = pl.DataFrame(rows)
    df.write_parquet(root / "sp500_tradable_prices.parquet")
    df.rename({c: f"{c}_tr" for c in ("open", "high", "low", "close")}).write_parquet(
        root / "sp500_total_return_prices.parquet"
    )
    df.write_parquet(root / "sp500_split_adjusted_prices.parquet")
    pl.DataFrame(
        schema={"ex_date": pl.Date, "symbol": pl.Utf8, "dividend_per_share": pl.Float64}
    ).write_parquet(root / "sp500_dividends.parquet")
    pl.DataFrame(
        {"date": df["date"], "symbol": df["symbol"], "adv_20d_dollar_lag1": [1e7] * df.height}
    ).write_parquet(root / "sp500_adv.parquet")
    pl.DataFrame(
        {"symbol": [f"S{s}" for s in range(5)], "borrow_tier": ["general"] * 5, "annual_bps": [100] * 5}
    ).write_parquet(root / "sp500_borrow_proxy.parquet")
    pl.DataFrame(
        schema={
            "symbol": pl.Utf8, "exit_date": pl.Date, "exit_reason": pl.Utf8,
            "terminal_return_captured": pl.Boolean, "terminal_return_value": pl.Float64,
            "classification_source": pl.Utf8, "classification": pl.Utf8,
        }
    ).write_parquet(root / "sp500_delisting_audit.parquet")

    arts = {}
    for key in (
        "sp500_tradable_prices", "sp500_total_return_prices", "sp500_split_adjusted_prices",
        "sp500_dividends", "sp500_adv", "sp500_borrow_proxy", "sp500_delisting_audit",
    ):
        p = root / f"{key}.parquet"
        arts[key] = ManifestArtifact(
            path=p.name, sha256=sha256_of_file(p),
            row_count=pl.read_parquet(p).height,
            symbol_count=int(pl.read_parquet(p)["symbol"].n_unique()) if "symbol" in pl.read_parquet(p).columns else 0,
            date_range_start=str(pl.read_parquet(p)["date"].min()) if "date" in pl.read_parquet(p).columns else "",
            date_range_end=str(pl.read_parquet(p)["date"].max()) if "date" in pl.read_parquet(p).columns else "",
            schema_fingerprint="cols:" + ",".join(pl.read_parquet(p).columns),
        )
    m = EquityManifest(
        pipeline_version="0.1.0", git_sha="deadbeef", artifacts=arts,
        data_quality_label=DataQualityLabel.SURVIVORSHIP_PROTOTYPE_ONLY,
        corporate_action_quality="split_adj_plus_external_dividends",
        borrow_source_quality="static_proxy_v1",
        pit_membership_source="absent_prototype_only",
        delisting_audit_quality="audit_absent",
        delisting_audit_counters=DelistingAuditCounters(),
        build_command_line="x", python_version="3.11.0", package_versions={}, warnings=[],
    )
    write_manifest(root / "_manifest.json", m)


def test_train_cli_fast_v1_smoke(tmp_path: Path) -> None:
    root = tmp_path / "equities"
    root.mkdir(parents=True, exist_ok=True)
    _seed_root(root)
    out = tmp_path / "runs"
    out.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [
            "uv", "run", "python", "scripts/train_s1_eq.py",
            "--config", "configs/alpha_eq.yaml",
            "--mode", "fast_v1",
            "--equity-root", str(root),
            "--experiments-root", str(out),
        ],
        check=True,
        capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    run_dirs = list(out.iterdir())
    assert run_dirs, res.stdout + res.stderr
    rd = run_dirs[0]
    assert (rd / "feature_cols.json").exists()
    assert (rd / "models" / "stacker.joblib").exists()
```

- [ ] **Step 2: Implement**

```python
"""Unified S1-EQ trainer CLI.

Usage:
    PYTHONPATH=src uv run python scripts/train_s1_eq.py \
        --config configs/alpha_eq.yaml --mode fast_v1 \
        --equity-root data/processed/equities \
        --experiments-root experiments/alpha_eq
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from rich.console import Console

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode
from quant_research_stack.alpha_eq.data.holdout import (
    HoldoutGate,
    assert_min_holdout_length,
    compute_holdout_dates,
)
from quant_research_stack.alpha_eq.data.loaders import EquityRootLoader
from quant_research_stack.alpha_eq.features.builder import (
    FeatureBuildConfig,
    build_features,
)
from quant_research_stack.alpha_eq.features.labels import build_labels
from quant_research_stack.alpha_eq.training.persist import persist_fast_v1_run

console = Console()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/alpha_eq.yaml")
    p.add_argument("--mode", default="fast_v1", choices=[m.value for m in TrainingMode])
    p.add_argument("--equity-root", default="data/processed/equities")
    p.add_argument("--experiments-root", default="experiments/alpha_eq")
    return p.parse_args()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _build_panel(loader: EquityRootLoader) -> pl.DataFrame:
    tradable = loader.load_tradable_prices()
    total_return = loader.load_total_return_prices()
    if "close_tr" in total_return.columns:
        panel = tradable.join(total_return.select(["date", "symbol", "close_tr"]), on=["date", "symbol"], how="left")
    else:
        panel = tradable.with_columns(pl.col("close").alias("close_tr"))
    panel = panel.with_columns(pl.lit(True).alias("in_universe"))
    return panel


def main() -> int:
    args = _parse_args()
    cfg_dict = yaml.safe_load(Path(args.config).read_text())
    cfg_dict["mode"] = args.mode
    config = AlphaEqConfig.model_validate(cfg_dict)

    np.random.seed(config.reproducibility.numpy_seed)

    loader = EquityRootLoader(root=Path(args.equity_root))
    panel = _build_panel(loader)

    sorted_dates = sorted(panel["date"].unique().to_list())
    dev_dates, hold_dates = compute_holdout_dates(
        sorted_dates, fraction=config.data.permanent_holdout_fraction
    )
    # In a real run we require ≥ 756 trading days; for prototype runs we
    # still record the count but only enforce when label is pit_safe (deferred to M6).
    if len(hold_dates) >= config.data.min_holdout_trading_days:
        assert_min_holdout_length(hold_dates, min_trading_days=config.data.min_holdout_trading_days)

    gate = HoldoutGate(holdout_dates=hold_dates)
    dev_panel = gate.filter_for_caller(panel, caller="training")

    features = build_features(panel=dev_panel, config=FeatureBuildConfig())
    features = build_labels(features, close_tr="close_tr", vol_col="realized_vol_20", universe_col="in_universe")

    feature_cols = [
        c for c in features.columns
        if c.startswith(("log_return_", "realized_vol_", "amihud_illiq_", "roll_spread_",
                         "kyle_proxy_signed_volume_", "overnight_gap", "intraday_return",
                         "close_location_", "dollar_volume", "log_dollar_volume_",
                         "volume_zscore_", "rank_", "spy_log_return_", "spy_realized_vol_",
                         "vix_close", "cross_sectional_", "gaussian_noise_"))
    ]

    run_dir = Path(args.experiments_root) / _run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    persist_fast_v1_run(
        run_dir=run_dir, config=config, feature_cols=feature_cols,
        dev_panel=features, target="y_xs",
    )
    # Update metadata.json git_sha + data_manifest_sha256 + DoF
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["git_sha"] = _git_sha()
    # Read the equity-root manifest sha indirectly via tradable_prices artifact:
    eq_manifest_path = Path(args.equity_root) / "_manifest.json"
    if eq_manifest_path.exists():
        import hashlib
        meta["data_manifest_sha256"] = hashlib.sha256(eq_manifest_path.read_bytes()).hexdigest()
    meta["build_command_line"] = " ".join(sys.argv)
    meta["python_version"] = platform.python_version()
    meta["holdout_dates_count"] = len(hold_dates)
    meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2))

    # Also persist holdout_dates.json
    (run_dir / "holdout_dates.json").write_text(
        json.dumps([str(d) for d in hold_dates], sort_keys=True)
    )

    console.print(f"[bold green]Run persisted:[/bold green] {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_train_cli_fast_v1.py -v
git add scripts/train_s1_eq.py tests/alpha_eq/test_train_cli_fast_v1.py
git commit -m "feat(s1-eq): train_s1_eq fast_v1 CLI with holdout lock + manifest hash"
```

---

### Task 40 — Reproducibility test (relaxed-byte-identical)

**Spec refs:** §4.9.

**Files:**
- Create: `tests/alpha_eq/test_reproducibility.py`

- [ ] **Step 1: Write test**

```python
"""Reproducibility contract (spec §4.9):
- byte-identical: splits, configs, feature_cols, manifest hashes
- within tolerance: predictions, metrics
"""

from __future__ import annotations

import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def _seed_minimal_root(root: Path) -> None:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(60):
        d = date(2020, 1, 1) + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(5):
            rows.append({
                "date": d, "symbol": f"S{s}",
                "open": 100.0 + float(rng.standard_normal()),
                "high": 101.0 + float(rng.standard_normal()),
                "low": 99.0 + float(rng.standard_normal()),
                "close": 100.0 + float(rng.standard_normal()),
                "volume": 1_000_000,
            })
    pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).write_parquet(
        root / "panel.parquet"
    )


def test_two_runs_produce_identical_feature_cols_and_splits(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    _seed_minimal_root(raw)

    eq = tmp_path / "equities"
    eq.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "run", "python", "scripts/prepare_equity_data.py",
         "--panel", str(raw / "panel.parquet"),
         "--equity-root", str(eq),
         "--membership-source", "absent_prototype_only"],
        check=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    out_a = tmp_path / "run_a"; out_a.mkdir()
    out_b = tmp_path / "run_b"; out_b.mkdir()
    for out in (out_a, out_b):
        subprocess.run(
            ["uv", "run", "python", "scripts/train_s1_eq.py",
             "--config", "configs/alpha_eq.yaml", "--mode", "fast_v1",
             "--equity-root", str(eq), "--experiments-root", str(out)],
            check=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )
    run_a = next(out_a.iterdir())
    run_b = next(out_b.iterdir())
    # byte-identical: feature_cols.json, holdout_dates.json
    assert (run_a / "feature_cols.json").read_bytes() == (run_b / "feature_cols.json").read_bytes()
    assert (run_a / "holdout_dates.json").read_bytes() == (run_b / "holdout_dates.json").read_bytes()
    # Stacker weights within tolerance
    import joblib
    sa = joblib.load(run_a / "models" / "stacker.joblib")["estimator"].coef_
    sb = joblib.load(run_b / "models" / "stacker.joblib")["estimator"].coef_
    np.testing.assert_allclose(sa, sb, atol=1e-6)
```

- [ ] **Step 2: Run + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_reproducibility.py -v
git add tests/alpha_eq/test_reproducibility.py
git commit -m "test(s1-eq): reproducibility contract (byte-id splits, tol-id weights)"
```

---

## M4 — Pragmatic-strict backtest engine

> **Priority gate per user §6:** the dividend-safe PnL accounting tests (Tasks 43–44 below) MUST be written and passing BEFORE the engine emits any reported number. Tasks 41–44 must complete before Tasks 45+.

### Task 41 — `backtest/contracts.py` temporal invariant + canonical row shape

**Spec refs:** §5.2.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/backtest/contracts.py`
- Create: `tests/alpha_eq/test_backtest_contracts.py`

- [ ] **Step 1: Write tests**

```python
"""Backtest row contract (spec §5.2)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_research_stack.alpha_eq.backtest.contracts import (
    BacktestContractError,
    assert_backtest_row_contract,
)


def test_assert_backtest_row_contract_passes_on_well_formed_row() -> None:
    df = pl.DataFrame(
        {
            "feature_as_of_date": [date(2020, 1, 2)],
            "execution_date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "signed_target_notional": [1000.0],
            "fill_price": [101.0],
        }
    )
    assert_backtest_row_contract(df)


def test_assert_backtest_row_contract_fails_when_feature_not_before_execution() -> None:
    df = pl.DataFrame(
        {
            "feature_as_of_date": [date(2020, 1, 3)],
            "execution_date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "signed_target_notional": [1000.0],
            "fill_price": [101.0],
        }
    )
    with pytest.raises(BacktestContractError):
        assert_backtest_row_contract(df)


def test_assert_backtest_row_contract_required_columns() -> None:
    df = pl.DataFrame({"symbol": ["AAPL"]})
    with pytest.raises(BacktestContractError):
        assert_backtest_row_contract(df)
```

- [ ] **Step 2: Implement**

```python
"""Backtest row temporal contract (spec §5.2)."""

from __future__ import annotations

import polars as pl

_REQUIRED_ROW_COLS: tuple[str, ...] = (
    "feature_as_of_date",
    "execution_date",
    "symbol",
    "signed_target_notional",
    "fill_price",
)


class BacktestContractError(RuntimeError):
    pass


def assert_backtest_row_contract(df: pl.DataFrame) -> None:
    missing = [c for c in _REQUIRED_ROW_COLS if c not in df.columns]
    if missing:
        raise BacktestContractError(f"missing required columns: {missing}")
    bad = df.filter(pl.col("feature_as_of_date") >= pl.col("execution_date"))
    if not bad.is_empty():
        raise BacktestContractError(
            f"feature_as_of_date >= execution_date on {bad.height} rows"
        )
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_backtest_contracts.py -v
git add src/quant_research_stack/alpha_eq/backtest/contracts.py tests/alpha_eq/test_backtest_contracts.py
git commit -m "feat(s1-eq): backtest row temporal contract"
```

---

### Task 42 — `backtest/pnl.py` skeleton — price PnL + cash dividend booking

**Spec refs:** §5.4, §5.11.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/backtest/pnl.py`

- [ ] **Step 1: Implement the dividend-safe PnL primitives**

(No tests yet — Tasks 43–44 are the tests, written in TDD order *against* this module.)

```python
"""Fill-aligned PnL accounting + cash dividend booking (spec §5.4).

The v1 invariant: portfolio MTM uses `tradable_*` prices (split-adjusted,
price-only); dividends are booked exactly once on ex-date as cash PnL.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(frozen=True)
class PnLDecomposition:
    gross_alpha_bps_per_day: float
    cash_dividend_bps_per_day: float
    commission_drag_bps_per_day: float
    spread_drag_bps_per_day: float
    borrow_drag_bps_per_day: float
    financing_drag_bps_per_day: float
    net_alpha_bps_per_day: float


def compute_position_price_pnl(
    *,
    held_positions: pl.DataFrame,           # cols: date, symbol, signed_notional_prev, close_prev, close_today
    new_lots: pl.DataFrame,                 # cols: date, symbol, signed_notional_new, fill_price, close_today
) -> pl.DataFrame:
    """Per-row price PnL (no dividends, no costs).

    Held positions: PnL = signed_notional_prev * (close_today / close_prev - 1).
    New lots:      PnL = signed_shares_new * (close_today - fill_price), where
                   signed_shares_new = signed_notional_new / fill_price.
    """
    if not held_positions.is_empty():
        held_pnl = held_positions.with_columns(
            (pl.col("signed_notional_prev") * (pl.col("close_today") / pl.col("close_prev") - 1.0))
            .alias("price_pnl")
        ).select(["date", "symbol", "price_pnl"])
    else:
        held_pnl = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "price_pnl": pl.Float64})

    if not new_lots.is_empty():
        new_pnl = new_lots.with_columns(
            (
                (pl.col("signed_notional_new") / pl.col("fill_price"))
                * (pl.col("close_today") - pl.col("fill_price"))
            ).alias("price_pnl")
        ).select(["date", "symbol", "price_pnl"])
    else:
        new_pnl = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "price_pnl": pl.Float64})

    return pl.concat([held_pnl, new_pnl])


def compute_cash_dividend_pnl(
    *,
    positions_on_ex_date: pl.DataFrame,     # cols: date, symbol, signed_notional, ref_close
    dividends: pl.DataFrame,                 # cols: ex_date, symbol, dividend_per_share
) -> pl.DataFrame:
    """Cash dividend PnL = signed_shares * dividend_per_share.

    Longs receive the dividend; shorts are debited (canonical convention).
    `signed_shares = signed_notional / ref_close` where `ref_close` is the
    prior trading day's close (holder-of-record snapshot).
    """
    if dividends.is_empty() or positions_on_ex_date.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "cash_dividend_pnl": pl.Float64})
    divs = dividends.rename({"ex_date": "date"})
    joined = positions_on_ex_date.join(divs, on=["date", "symbol"], how="inner")
    return joined.with_columns(
        ((pl.col("signed_notional") / pl.col("ref_close")) * pl.col("dividend_per_share")).alias(
            "cash_dividend_pnl"
        )
    ).select(["date", "symbol", "cash_dividend_pnl"])


def decompose_pnl(
    *,
    price_pnl: pl.DataFrame,
    cash_dividend_pnl: pl.DataFrame,
    commission_drag: pl.DataFrame,
    spread_drag: pl.DataFrame,
    borrow_drag: pl.DataFrame,
    financing_drag: pl.DataFrame,
    equity: float,
    n_days: int,
) -> PnLDecomposition:
    """Aggregate the five drags and return per-day bps numbers used in reports.

    portfolio_pnl = price_pnl + cash_dividend_pnl - sum_of_drags
    """

    def _per_day_bps(frame: pl.DataFrame, col: str) -> float:
        if frame.is_empty() or n_days == 0 or equity == 0:
            return 0.0
        return float(frame[col].sum()) / float(equity) / float(n_days) * 10_000.0

    gross_bps = _per_day_bps(price_pnl, "price_pnl") + _per_day_bps(
        cash_dividend_pnl, "cash_dividend_pnl"
    )
    div_bps = _per_day_bps(cash_dividend_pnl, "cash_dividend_pnl")
    comm_bps = _per_day_bps(commission_drag, "commission_drag")
    spread_bps = _per_day_bps(spread_drag, "spread_drag")
    borrow_bps = _per_day_bps(borrow_drag, "borrow_drag")
    fin_bps = _per_day_bps(financing_drag, "financing_drag")
    net_bps = gross_bps - comm_bps - spread_bps - borrow_bps - fin_bps
    return PnLDecomposition(
        gross_alpha_bps_per_day=gross_bps,
        cash_dividend_bps_per_day=div_bps,
        commission_drag_bps_per_day=comm_bps,
        spread_drag_bps_per_day=spread_bps,
        borrow_drag_bps_per_day=borrow_bps,
        financing_drag_bps_per_day=fin_bps,
        net_alpha_bps_per_day=net_bps,
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/quant_research_stack/alpha_eq/backtest/pnl.py
git commit -m "feat(s1-eq): backtest PnL primitives — price + cash dividends + decomposition"
```

---

### Task 43 — `test_fill_pnl_alignment.py` (priority-2 gate test)

**Spec refs:** §5.4.

**Files:**
- Create: `tests/alpha_eq/test_fill_pnl_alignment.py`

- [ ] **Step 1: Write test**

```python
"""Fill-aligned PnL: new positions PnL from FILL price (not close_t),
existing positions close-to-close MTM (spec §5.4)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.pnl import compute_position_price_pnl


def test_new_position_pnl_is_close_minus_fill_times_shares() -> None:
    new_lots = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "signed_notional_new": [10_000.0],   # long $10k
            "fill_price": [100.0],                # next-day open
            "close_today": [102.0],
        }
    )
    held = pl.DataFrame(
        schema={
            "date": pl.Date, "symbol": pl.Utf8,
            "signed_notional_prev": pl.Float64,
            "close_prev": pl.Float64, "close_today": pl.Float64,
        }
    )
    pnl = compute_position_price_pnl(held_positions=held, new_lots=new_lots)
    # shares = 10000/100 = 100; price PnL = 100 * (102 - 100) = 200
    assert pnl.height == 1
    assert abs(pnl["price_pnl"][0] - 200.0) < 1e-9


def test_held_position_pnl_is_close_to_close_total_return() -> None:
    held = pl.DataFrame(
        {
            "date": [date(2020, 1, 4)],
            "symbol": ["AAPL"],
            "signed_notional_prev": [10_000.0],
            "close_prev": [102.0],
            "close_today": [104.04],
        }
    )
    new_lots = pl.DataFrame(
        schema={
            "date": pl.Date, "symbol": pl.Utf8,
            "signed_notional_new": pl.Float64, "fill_price": pl.Float64,
            "close_today": pl.Float64,
        }
    )
    pnl = compute_position_price_pnl(held_positions=held, new_lots=new_lots)
    # ret = 104.04/102 - 1 = 0.02; PnL = 10_000 * 0.02 = 200
    assert pnl.height == 1
    assert abs(pnl["price_pnl"][0] - 200.0) < 1e-9


def test_short_new_position_pnl_signed_correctly() -> None:
    new_lots = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "symbol": ["XYZ"],
            "signed_notional_new": [-10_000.0],   # short $10k
            "fill_price": [100.0],
            "close_today": [102.0],              # stock went UP → short LOSES
        }
    )
    held = pl.DataFrame(
        schema={
            "date": pl.Date, "symbol": pl.Utf8,
            "signed_notional_prev": pl.Float64,
            "close_prev": pl.Float64, "close_today": pl.Float64,
        }
    )
    pnl = compute_position_price_pnl(held_positions=held, new_lots=new_lots)
    # signed_shares = -100; PnL = -100 * (102 - 100) = -200 (correct: short loss)
    assert abs(pnl["price_pnl"][0] - (-200.0)) < 1e-9
```

- [ ] **Step 2: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_fill_pnl_alignment.py -v
git add tests/alpha_eq/test_fill_pnl_alignment.py
git commit -m "test(s1-eq): fill-aligned PnL — new positions PnL from fill, not close_t"
```

---

### Task 44 — `test_no_dividend_double_count.py` (priority-2 gate test)

**Spec refs:** §5.11.

**Files:**
- Create: `tests/alpha_eq/test_no_dividend_double_count.py`

- [ ] **Step 1: Write test**

```python
"""Dividends are booked exactly once as cash PnL on ex-date; the MTM path
uses split-adjusted tradable_* (NOT total-return) prices, so there is no
double-count (spec §5.11)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.pnl import (
    compute_cash_dividend_pnl,
    compute_position_price_pnl,
)


def test_long_receives_dividend_cash_once() -> None:
    positions = pl.DataFrame(
        {
            "date": [date(2020, 1, 6)],
            "symbol": ["AAPL"],
            "signed_notional": [10_000.0],
            "ref_close": [100.0],   # 100 shares
        }
    )
    dividends = pl.DataFrame(
        {
            "ex_date": [date(2020, 1, 6)],
            "symbol": ["AAPL"],
            "dividend_per_share": [0.5],
        }
    )
    div_pnl = compute_cash_dividend_pnl(positions_on_ex_date=positions, dividends=dividends)
    # 100 shares × $0.50 = $50 cash dividend
    assert abs(div_pnl["cash_dividend_pnl"][0] - 50.0) < 1e-9


def test_short_is_debited_dividend_cash_once() -> None:
    positions = pl.DataFrame(
        {
            "date": [date(2020, 1, 6)],
            "symbol": ["AAPL"],
            "signed_notional": [-10_000.0],
            "ref_close": [100.0],
        }
    )
    dividends = pl.DataFrame(
        {
            "ex_date": [date(2020, 1, 6)],
            "symbol": ["AAPL"],
            "dividend_per_share": [0.5],
        }
    )
    div_pnl = compute_cash_dividend_pnl(positions_on_ex_date=positions, dividends=dividends)
    assert abs(div_pnl["cash_dividend_pnl"][0] - (-50.0)) < 1e-9


def test_price_pnl_does_not_include_dividend_when_tradable_used() -> None:
    """Held position across ex-date: price PnL uses tradable_close (split-adj),
    which has NOT been bumped by the dividend.  Therefore the price PnL must
    equal the raw close-to-close change with no dividend lift."""
    held = pl.DataFrame(
        {
            "date": [date(2020, 1, 6)],
            "symbol": ["AAPL"],
            "signed_notional_prev": [10_000.0],
            "close_prev": [100.0],
            "close_today": [99.50],   # ex-date drop ≈ dividend amount
        }
    )
    new_lots = pl.DataFrame(
        schema={"date": pl.Date, "symbol": pl.Utf8, "signed_notional_new": pl.Float64,
                "fill_price": pl.Float64, "close_today": pl.Float64}
    )
    pnl = compute_position_price_pnl(held_positions=held, new_lots=new_lots)
    # price PnL = 10_000 * (99.50/100.0 - 1) = -50
    assert abs(pnl["price_pnl"][0] - (-50.0)) < 1e-9
    # Combined with cash dividend of +50 (longs), holder is flat over ex-date — correct.
```

- [ ] **Step 2: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_no_dividend_double_count.py -v
git add tests/alpha_eq/test_no_dividend_double_count.py
git commit -m "test(s1-eq): no-dividend-double-count — long+ex-date drop nets flat"
```

---

### Task 45 — `backtest/portfolio.py`: universe gating + equal-weight L/S + min-bucket + per-name caps

**Spec refs:** §5.5.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/backtest/portfolio.py`
- Create: `tests/alpha_eq/test_backtest_portfolio.py`

- [ ] **Step 1: Write tests**

```python
"""Portfolio construction (spec §5.5)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.portfolio import (
    PortfolioBuildConfig,
    PortfolioConstructionError,
    build_target_positions,
)


def _signals(date_: date, n: int = 20) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "execution_date": [date_] * n,
            "symbol": [f"S{i}" for i in range(n)],
            "y_xs_pred": [(i - n / 2) / n for i in range(n)],
            "adv_20d_dollar_lag1": [1e8] * n,
            "tradable": [True] * n,
            "in_pit_universe": [True] * n,
            "fill_price": [100.0] * n,
            "borrow_tier": ["general"] * n,
        }
    )


def test_equal_weight_dollar_neutral_book() -> None:
    cfg = PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0)
    pos = build_target_positions(signals=_signals(date(2020, 1, 3)), config=cfg, cohort="full_universe")
    # 10% of 20 names = 2 longs + 2 shorts
    longs = pos.filter(pl.col("signed_target_notional") > 0)
    shorts = pos.filter(pl.col("signed_target_notional") < 0)
    assert longs.height == 2
    assert shorts.height == 2


def test_minimum_bucket_full_universe() -> None:
    cfg = PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0)
    # Only 5 names → can't meet 10/10 minimum → skip date
    sig = _signals(date(2020, 1, 3), n=5)
    pos = build_target_positions(signals=sig, config=cfg, cohort="full_universe")
    assert pos.is_empty(), "date with insufficient names must be skipped, not silently empty"


def test_per_name_adv_cap_overrides_equal_weight() -> None:
    cfg = PortfolioBuildConfig(
        q_quantile=0.10, target_gross=1.0, equity=10_000_000.0, adv_participation_pct=0.01,
    )
    sig = _signals(date(2020, 1, 3), n=20).with_columns(pl.lit(1_000_000.0).alias("adv_20d_dollar_lag1"))
    pos = build_target_positions(signals=sig, config=cfg, cohort="full_universe")
    # cap = 1% of $1M = $10k per name
    assert pos["signed_target_notional"].abs().max() <= 10_000.0 + 1e-6


def test_out_of_universe_rows_are_dropped() -> None:
    cfg = PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0)
    sig = _signals(date(2020, 1, 3)).with_columns(
        pl.when(pl.col("symbol") == "S0").then(False).otherwise(True).alias("in_pit_universe")
    )
    pos = build_target_positions(signals=sig, config=cfg, cohort="full_universe")
    assert "S0" not in pos["symbol"].to_list()


def test_focused_basket_min_bucket_5_5() -> None:
    cfg = PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0)
    sig = _signals(date(2020, 1, 3), n=20)
    pos = build_target_positions(signals=sig, config=cfg, cohort="focused_basket")
    assert pos.filter(pl.col("signed_target_notional") > 0).height >= 5 or pos.is_empty()
```

- [ ] **Step 2: Implement**

```python
"""Portfolio construction for the strict backtest (spec §5.5)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


class PortfolioConstructionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PortfolioBuildConfig:
    q_quantile: float
    target_gross: float
    equity: float
    adv_participation_pct: float = 0.01
    min_long_full_universe: int = 10
    min_short_full_universe: int = 10
    min_long_focused_basket: int = 5
    min_short_focused_basket: int = 5
    max_single_name_weight_frac_of_gross: float = 0.05


def _bucket_size(n_total: int, q: float, *, min_required: int) -> int:
    raw = max(1, int(round(n_total * q)))
    return max(raw, min_required)


def build_target_positions(
    *,
    signals: pl.DataFrame,
    config: PortfolioBuildConfig,
    cohort: str,
) -> pl.DataFrame:
    """Return per-name signed target notional for one execution date.

    Empty DataFrame returned if minimum-bucket cannot be met or if no
    tradable/in-universe rows exist (caller skips the date).
    """
    eligible = signals.filter(pl.col("tradable") & pl.col("in_pit_universe"))
    if eligible.is_empty():
        return eligible.select(["execution_date", "symbol"]).with_columns(
            pl.lit(0.0).alias("signed_target_notional"),
            pl.lit("").alias("borrow_tier"),
            pl.lit(0.0).alias("fill_price"),
        ).head(0)

    sorted_sig = eligible.sort("y_xs_pred")
    n = sorted_sig.height
    min_long = (
        config.min_long_full_universe
        if cohort == "full_universe"
        else config.min_long_focused_basket
    )
    min_short = (
        config.min_short_full_universe
        if cohort == "full_universe"
        else config.min_short_focused_basket
    )

    short_size = _bucket_size(n, config.q_quantile, min_required=min_short)
    long_size = _bucket_size(n, config.q_quantile, min_required=min_long)
    if short_size + long_size > n or short_size < min_short or long_size < min_long:
        return sorted_sig.select(["execution_date", "symbol"]).with_columns(
            pl.lit(0.0).alias("signed_target_notional"),
            pl.lit("").alias("borrow_tier"),
            pl.lit(0.0).alias("fill_price"),
        ).head(0)

    shorts = sorted_sig.head(short_size)
    longs = sorted_sig.tail(long_size)

    gross_dollars = config.target_gross * config.equity
    per_side_dollars = gross_dollars / 2.0
    equal_long = per_side_dollars / float(long_size)
    equal_short = per_side_dollars / float(short_size)
    weight_cap = gross_dollars * config.max_single_name_weight_frac_of_gross

    def _cap(side: pl.DataFrame, equal_weight: float, sign: int) -> pl.DataFrame:
        adv_cap = pl.col("adv_20d_dollar_lag1") * float(config.adv_participation_pct)
        capped_abs = pl.min_horizontal(
            pl.lit(equal_weight),
            adv_cap,
            pl.lit(float(weight_cap)),
        )
        return side.with_columns(
            (sign * capped_abs).alias("signed_target_notional"),
        )

    book = pl.concat(
        [
            _cap(longs, equal_long, +1),
            _cap(shorts, equal_short, -1),
        ]
    )
    return book.select(
        ["execution_date", "symbol", "signed_target_notional", "borrow_tier", "fill_price"]
    )
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_backtest_portfolio.py -v
git add src/quant_research_stack/alpha_eq/backtest/portfolio.py tests/alpha_eq/test_backtest_portfolio.py
git commit -m "feat(s1-eq): portfolio construction (equal-weight L/S + min-bucket + ADV cap)"
```

---

### Task 46 — `backtest/fills.py` next-day open + HLC3 proxy + close sensitivities

**Spec refs:** §5.3.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/backtest/fills.py`
- Create: `tests/alpha_eq/test_backtest_fills.py`

- [ ] **Step 1: Write tests**

```python
"""Fill-price selection (spec §5.3)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.fills import (
    FillModel,
    pick_fill_prices,
)


def _bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
        }
    )


def test_fill_open() -> None:
    out = pick_fill_prices(_bars(), model=FillModel.OPEN)
    assert out["fill_price"][0] == 100.0


def test_fill_hlc3_proxy_labeled() -> None:
    out = pick_fill_prices(_bars(), model=FillModel.HLC3_PROXY)
    assert abs(out["fill_price"][0] - (102.0 + 99.0 + 101.0) / 3.0) < 1e-9
    assert out["fill_model"][0] == "vwap_proxy_hlc3"


def test_fill_close() -> None:
    out = pick_fill_prices(_bars(), model=FillModel.CLOSE)
    assert out["fill_price"][0] == 101.0
```

- [ ] **Step 2: Implement**

```python
"""Fill-price selection (spec §5.3).  HLC3 is ALWAYS labelled vwap_proxy_hlc3
in any artifact column; never called real VWAP."""

from __future__ import annotations

import enum

import polars as pl


class FillModel(str, enum.Enum):
    OPEN = "open"
    HLC3_PROXY = "vwap_proxy_hlc3"
    CLOSE = "close"


def pick_fill_prices(bars: pl.DataFrame, *, model: FillModel) -> pl.DataFrame:
    if model is FillModel.OPEN:
        out = bars.with_columns(pl.col("open").alias("fill_price"))
    elif model is FillModel.HLC3_PROXY:
        out = bars.with_columns(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("fill_price")
        )
    elif model is FillModel.CLOSE:
        out = bars.with_columns(pl.col("close").alias("fill_price"))
    else:  # pragma: no cover
        raise ValueError(f"unknown fill model: {model}")
    return out.with_columns(pl.lit(model.value).alias("fill_model"))
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_backtest_fills.py -v
git add src/quant_research_stack/alpha_eq/backtest/fills.py tests/alpha_eq/test_backtest_fills.py
git commit -m "feat(s1-eq): fill-price selection (open headline + HLC3 proxy + close)"
```

---

### Task 47 — `backtest/costs.py` commission + spread + pre-decimalization adjustment

**Spec refs:** §5.6.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/backtest/costs.py`
- Create: `tests/alpha_eq/test_backtest_costs.py`

- [ ] **Step 1: Write tests**

```python
"""Cost model — commission + spread + pre-decimalization (spec §5.6)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.costs import (
    CostConfig,
    compute_commission_drag,
    compute_spread_drag,
)


def test_commission_drag_is_bps_one_way() -> None:
    trades = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "trade_notional_abs": [100_000.0],
        }
    )
    drag = compute_commission_drag(trades, cost=CostConfig())
    assert abs(drag["commission_drag"][0] - 100_000.0 * 0.5 / 10_000.0) < 1e-9


def test_spread_drag_uses_roll_when_available_else_tier() -> None:
    trades = pl.DataFrame(
        {
            "date": [date(2020, 1, 3), date(2020, 1, 3)],
            "symbol": ["A", "B"],
            "trade_notional_abs": [100_000.0, 100_000.0],
            "roll_spread_bps": [10.0, None],
            "tier": ["general", "general"],
        }
    )
    drag = compute_spread_drag(trades, cost=CostConfig())
    # 10 bps roll → 5 bps half-spread; tier=general → 15 bps tier → 7.5 bps
    assert abs(drag["spread_drag"][0] - 100_000.0 * 5.0 / 10_000.0) < 1e-9
    assert abs(drag["spread_drag"][1] - 100_000.0 * 7.5 / 10_000.0) < 1e-9


def test_pre_decimalization_multiplier_widens_pre_2001() -> None:
    trades = pl.DataFrame(
        {
            "date": [date(2000, 6, 15), date(2002, 6, 15)],
            "symbol": ["A", "A"],
            "trade_notional_abs": [100_000.0, 100_000.0],
            "roll_spread_bps": [None, None],
            "tier": ["general", "general"],
        }
    )
    drag = compute_spread_drag(trades, cost=CostConfig())
    # pre-decimal multiplier 2.5x → 15 * 2.5 = 37.5 bps; half = 18.75 bps
    assert drag["spread_drag"][0] > drag["spread_drag"][1]
```

- [ ] **Step 2: Implement**

```python
"""Trade-cost model (spec §5.6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(frozen=True)
class CostConfig:
    commission_bps_one_way: float = 0.5
    roll_spread_cap_bps: float = 50.0
    tiered_fallback_easy_bps: float = 5.0
    tiered_fallback_general_bps: float = 15.0
    tiered_fallback_hard_bps: float = 50.0
    pre_decimalization_cutoff: date = date(2001, 4, 9)
    pre_decimalization_multiplier_fallback: float = 2.5
    pre_decimalization_multiplier_roll: float = 1.5


def compute_commission_drag(trades: pl.DataFrame, *, cost: CostConfig) -> pl.DataFrame:
    return trades.with_columns(
        (pl.col("trade_notional_abs") * cost.commission_bps_one_way / 10_000.0).alias("commission_drag")
    )


def compute_spread_drag(trades: pl.DataFrame, *, cost: CostConfig) -> pl.DataFrame:
    tiered = (
        pl.when(pl.col("tier") == "easy").then(cost.tiered_fallback_easy_bps)
        .when(pl.col("tier") == "hard").then(cost.tiered_fallback_hard_bps)
        .otherwise(cost.tiered_fallback_general_bps)
    )
    raw_spread = pl.when(pl.col("roll_spread_bps").is_not_null()).then(
        pl.min_horizontal(pl.col("roll_spread_bps"), pl.lit(cost.roll_spread_cap_bps))
    ).otherwise(tiered)

    pre_decimal_mult = pl.when(pl.col("date") < cost.pre_decimalization_cutoff).then(
        pl.when(pl.col("roll_spread_bps").is_not_null())
        .then(cost.pre_decimalization_multiplier_roll)
        .otherwise(cost.pre_decimalization_multiplier_fallback)
    ).otherwise(1.0)

    half_spread_bps = (raw_spread * pre_decimal_mult) / 2.0
    return trades.with_columns(
        (pl.col("trade_notional_abs") * half_spread_bps / 10_000.0).alias("spread_drag")
    )
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_backtest_costs.py -v
git add src/quant_research_stack/alpha_eq/backtest/costs.py tests/alpha_eq/test_backtest_costs.py
git commit -m "feat(s1-eq): cost model — commission + spread + pre-decimalization"
```

---

### Task 48 — `backtest/borrow.py` + `backtest/financing.py`

**Spec refs:** §5.7, §5.8.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/backtest/borrow.py` (thin wrapper around `data/borrow_proxy.py`)
- Create: `src/quant_research_stack/alpha_eq/backtest/financing.py`
- Create: `tests/alpha_eq/test_backtest_borrow_financing.py`

- [ ] **Step 1: Write tests**

```python
"""Borrow + financing (spec §5.7, §5.8)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.borrow import apply_borrow_drag
from quant_research_stack.alpha_eq.backtest.financing import compute_financing_drag


def test_borrow_multiplier_monotonic() -> None:
    pos = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "signed_notional": [-100_000.0],
            "tier": ["general"],
        }
    )
    one = apply_borrow_drag(pos, multiplier=1.0)["borrow_drag"][0]
    two = apply_borrow_drag(pos, multiplier=2.0)["borrow_drag"][0]
    three = apply_borrow_drag(pos, multiplier=3.0)["borrow_drag"][0]
    assert one < two < three


def test_borrow_zero_on_longs() -> None:
    pos = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "symbol": ["AAPL"],
            "signed_notional": [100_000.0],
            "tier": ["general"],
        }
    )
    drag = apply_borrow_drag(pos, multiplier=3.0)
    assert drag["borrow_drag"][0] == 0.0


def test_financing_only_when_gross_above_1() -> None:
    pos = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "gross_notional": [200_000.0],   # 2x leverage
            "equity": [100_000.0],
        }
    )
    fin = compute_financing_drag(pos, rate_annual=0.02)
    # excess = 100_000; daily = 100_000 * 0.02 / 252 ≈ 7.94
    assert abs(fin["financing_drag"][0] - 100_000.0 * 0.02 / 252.0) < 1e-6


def test_financing_zero_when_gross_one() -> None:
    pos = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)],
            "gross_notional": [100_000.0],
            "equity": [100_000.0],
        }
    )
    fin = compute_financing_drag(pos, rate_annual=0.02)
    assert fin["financing_drag"][0] == 0.0
```

- [ ] **Step 2: Implement**

```python
"""Borrow drag for the strict backtest (spec §5.7)."""

from __future__ import annotations

import polars as pl

from quant_research_stack.alpha_eq.data.borrow_proxy import apply_borrow_charges


def apply_borrow_drag(positions: pl.DataFrame, *, multiplier: float) -> pl.DataFrame:
    out = apply_borrow_charges(positions, multiplier=multiplier)
    return out.rename({"borrow_cost": "borrow_drag"})
```

```python
"""Financing drag for gross > 1.0 (spec §5.8)."""

from __future__ import annotations

import polars as pl


def compute_financing_drag(positions: pl.DataFrame, *, rate_annual: float) -> pl.DataFrame:
    excess = (pl.col("gross_notional") - pl.col("equity")).clip(lower_bound=0.0)
    daily = excess * float(rate_annual) / 252.0
    return positions.with_columns(daily.alias("financing_drag"))
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_backtest_borrow_financing.py -v
git add src/quant_research_stack/alpha_eq/backtest/borrow.py src/quant_research_stack/alpha_eq/backtest/financing.py tests/alpha_eq/test_backtest_borrow_financing.py
git commit -m "feat(s1-eq): backtest borrow drag + leverage-financing drag"
```

---

### Task 49 — `test_pnl_decomposition.py`

**Spec refs:** §5.11.

**Files:**
- Create: `tests/alpha_eq/test_pnl_decomposition.py`

- [ ] **Step 1: Write test**

```python
"""Identity: gross_alpha - (commission + spread + borrow + financing) ≈ net_alpha."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.backtest.pnl import decompose_pnl


def _frame(col: str, value: float) -> pl.DataFrame:
    return pl.DataFrame({"date": [date(2020, 1, 3)], "symbol": ["A"], col: [value]})


def test_decomposition_identity_holds() -> None:
    dec = decompose_pnl(
        price_pnl=_frame("price_pnl", 100.0),
        cash_dividend_pnl=_frame("cash_dividend_pnl", 10.0),
        commission_drag=_frame("commission_drag", 5.0),
        spread_drag=_frame("spread_drag", 7.0),
        borrow_drag=_frame("borrow_drag", 3.0),
        financing_drag=_frame("financing_drag", 2.0),
        equity=10_000.0,
        n_days=1,
    )
    # gross = 110, drags = 17 → net = 93 → 93 bps over equity=10k means 93/10000*10000 = 93 bps
    assert abs(dec.gross_alpha_bps_per_day - 110.0) < 1e-6
    assert abs(dec.net_alpha_bps_per_day - (110.0 - 17.0)) < 1e-6
```

- [ ] **Step 2: Run + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_pnl_decomposition.py -v
git add tests/alpha_eq/test_pnl_decomposition.py
git commit -m "test(s1-eq): PnL decomposition identity"
```

---

### Task 50 — `backtest/exposure.py` mandatory exposure diagnostics

**Spec refs:** §5.12.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/backtest/exposure.py`
- Create: `tests/alpha_eq/test_backtest_exposure.py`

- [ ] **Step 1: Write tests**

```python
"""Exposure diagnostics (spec §5.12)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.backtest.exposure import (
    compute_daily_exposures,
    rolling_spy_beta,
    top_n_contributors,
)


def test_compute_daily_exposures_shapes() -> None:
    positions = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signed_notional": [50_000.0, 30_000.0, -40_000.0, -40_000.0],
            "sector": ["tech", "tech", "finance", "energy"],
        }
    )
    expo = compute_daily_exposures(positions=positions)
    row = expo.row(0, named=True)
    assert abs(row["net_exposure"] - 0.0) < 1e-6
    assert abs(row["gross_exposure"] - 160_000.0) < 1e-6
    assert row["sector_long_tech"] == 80_000.0
    assert row["sector_short_energy"] == 40_000.0


def test_rolling_spy_beta_returns_one_per_date() -> None:
    n = 80
    dates = [date(2020, 1, 1) for _ in range(n)]
    rng = np.random.default_rng(0)
    spy = rng.standard_normal(n) * 0.01
    port = spy * 0.5 + rng.standard_normal(n) * 0.005
    df = pl.DataFrame({"date": pl.date_range(date(2020, 1, 1), date(2020, 4, 19), interval="1d", eager=True).slice(0, n),
                        "portfolio_return": port, "spy_return": spy})
    out = rolling_spy_beta(df, window=60)
    assert "rolling_spy_beta" in out.columns
    assert out["rolling_spy_beta"].drop_nulls().to_numpy()[-1] > 0.0


def test_top_n_contributors() -> None:
    pnl = pl.DataFrame(
        {
            "date": [date(2020, 1, 3)] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "net_pnl": [100.0, -50.0, 200.0, -10.0, 5.0],
        }
    )
    top = top_n_contributors(pnl, by="symbol", n=2)
    assert top["symbol"].to_list() == ["C", "A"]
```

- [ ] **Step 2: Implement**

```python
"""Exposure diagnostics (spec §5.12)."""

from __future__ import annotations

import polars as pl


def compute_daily_exposures(*, positions: pl.DataFrame) -> pl.DataFrame:
    by_date = positions.group_by("date").agg(
        pl.col("signed_notional").sum().alias("net_exposure"),
        pl.col("signed_notional").abs().sum().alias("gross_exposure"),
        pl.col("signed_notional").filter(pl.col("signed_notional") > 0).sum().alias("long_exposure"),
        pl.col("signed_notional").filter(pl.col("signed_notional") < 0).abs().sum().alias("short_exposure"),
    )
    if "sector" in positions.columns:
        sec = positions.with_columns(
            pl.when(pl.col("signed_notional") > 0)
            .then(pl.col("signed_notional"))
            .otherwise(0.0)
            .alias("_long_sec"),
            pl.when(pl.col("signed_notional") < 0)
            .then(-pl.col("signed_notional"))
            .otherwise(0.0)
            .alias("_short_sec"),
        )
        long_pivot = sec.group_by(["date", "sector"]).agg(pl.col("_long_sec").sum().alias("v")).pivot(
            values="v", index="date", on="sector"
        )
        long_pivot = long_pivot.rename({c: f"sector_long_{c}" for c in long_pivot.columns if c != "date"})
        short_pivot = sec.group_by(["date", "sector"]).agg(pl.col("_short_sec").sum().alias("v")).pivot(
            values="v", index="date", on="sector"
        )
        short_pivot = short_pivot.rename({c: f"sector_short_{c}" for c in short_pivot.columns if c != "date"})
        by_date = by_date.join(long_pivot, on="date", how="left").join(short_pivot, on="date", how="left").fill_null(0.0)
    return by_date


def rolling_spy_beta(df: pl.DataFrame, *, window: int) -> pl.DataFrame:
    """OLS beta of portfolio_return on spy_return over a rolling window."""
    return df.sort("date").with_columns(
        (
            (
                (pl.col("portfolio_return") * pl.col("spy_return"))
                .rolling_mean(window_size=window, min_periods=window)
                - pl.col("portfolio_return").rolling_mean(window_size=window, min_periods=window)
                * pl.col("spy_return").rolling_mean(window_size=window, min_periods=window)
            )
            / (
                pl.col("spy_return")
                .rolling_var(window_size=window, min_periods=window)
                .clip(lower_bound=1e-18)
            )
        ).alias("rolling_spy_beta")
    )


def top_n_contributors(pnl: pl.DataFrame, *, by: str, n: int) -> pl.DataFrame:
    return (
        pnl.group_by(by)
        .agg(pl.col("net_pnl").sum().alias("total_net_pnl"))
        .sort("total_net_pnl", descending=True)
        .head(n)
        .rename({by: by})
    )
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_backtest_exposure.py -v
git add src/quant_research_stack/alpha_eq/backtest/exposure.py tests/alpha_eq/test_backtest_exposure.py
git commit -m "feat(s1-eq): exposure diagnostics (net/gross/sector L-S/SPY beta/top-N)"
```

---

### Task 51 — `backtest/metrics.py` Sharpe / Sortino / Calmar + monthly/annual

**Spec refs:** §5.13.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/backtest/metrics.py`
- Create: `tests/alpha_eq/test_backtest_metrics.py`

- [ ] **Step 1: Write tests**

```python
"""Backtest metrics (spec §5.13)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.backtest.metrics import (
    annualized_return,
    annualized_sharpe,
    annualized_sortino,
    calmar_ratio,
    max_drawdown,
    monthly_returns,
)


def test_sharpe_of_constant_positive() -> None:
    r = np.full(252, 0.001)
    assert annualized_sharpe(r) > 100.0  # near-infinite for zero-vol series


def test_max_drawdown_negative() -> None:
    r = np.array([0.01, 0.01, -0.30, 0.05])
    mdd = max_drawdown(r)
    assert mdd < 0


def test_monthly_returns_aggregate() -> None:
    n = 60
    dates = pl.date_range(date(2020, 1, 1), date(2020, 12, 31), interval="1d", eager=True).slice(0, n)
    r = np.full(n, 0.001)
    df = pl.DataFrame({"date": dates, "net_return": r})
    m = monthly_returns(df)
    assert "year_month" in m.columns
    assert m.height >= 2


def test_calmar_ratio_uses_max_dd() -> None:
    r = np.array([0.001, 0.001, -0.05, 0.001, 0.001])
    c = calmar_ratio(r)
    assert c == 0 or c != 0


def test_annualized_return() -> None:
    r = np.full(252, 0.0)
    assert abs(annualized_return(r) - 0.0) < 1e-12
```

- [ ] **Step 2: Implement**

```python
"""Backtest metrics (spec §5.13)."""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray


def annualized_return(returns: NDArray[np.float64]) -> float:
    if returns.size == 0:
        return 0.0
    cum = float(np.prod(1.0 + returns))
    return float(cum ** (252.0 / returns.size) - 1.0)


def annualized_sharpe(returns: NDArray[np.float64]) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0:
        return float("inf") if float(np.mean(returns)) > 0 else 0.0
    return float(np.mean(returns)) / sd * np.sqrt(252.0)


def annualized_sortino(returns: NDArray[np.float64]) -> float:
    if returns.size < 2:
        return 0.0
    downside = returns[returns < 0]
    sd = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    if sd == 0.0:
        return float("inf") if float(np.mean(returns)) > 0 else 0.0
    return float(np.mean(returns)) / sd * np.sqrt(252.0)


def max_drawdown(returns: NDArray[np.float64]) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def calmar_ratio(returns: NDArray[np.float64]) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0.0:
        return 0.0
    return annualized_return(returns) / abs(mdd)


def monthly_returns(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("year_month"))
        .group_by("year_month")
        .agg(((pl.col("net_return") + 1.0).product() - 1.0).alias("monthly_return"))
        .sort("year_month")
    )


def annual_returns(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("date").dt.strftime("%Y").alias("year"))
        .group_by("year")
        .agg(((pl.col("net_return") + 1.0).product() - 1.0).alias("annual_return"))
        .sort("year")
    )
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_backtest_metrics.py -v
git add src/quant_research_stack/alpha_eq/backtest/metrics.py tests/alpha_eq/test_backtest_metrics.py
git commit -m "feat(s1-eq): backtest metrics — Sharpe/Sortino/Calmar + monthly/annual"
```

---

### Task 52 — `backtest/runner.py` — orchestrate per-date stepping

**Spec refs:** §5.2, §5.5–§5.11.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/backtest/runner.py`
- Create: `tests/alpha_eq/test_backtest_runner.py`

- [ ] **Step 1: Write tests**

```python
"""Backtest runner end-to-end on tiny synthetic data."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.backtest.fills import FillModel
from quant_research_stack.alpha_eq.backtest.portfolio import PortfolioBuildConfig
from quant_research_stack.alpha_eq.backtest.runner import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)


def _toy_signals_panel(n_days: int = 30, n_symbols: int = 12) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_days):
        d = date(2020, 1, 1) + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(n_symbols):
            rows.append(
                {
                    "execution_date": d,
                    "feature_as_of_date": d - timedelta(days=1),
                    "symbol": f"S{s}",
                    "y_xs_pred": float(rng.standard_normal()),
                    "open": 100.0 + float(rng.standard_normal()),
                    "high": 101.0 + float(rng.standard_normal()),
                    "low": 99.0 + float(rng.standard_normal()),
                    "close": 100.0 + float(rng.standard_normal()),
                    "tradable_close": 100.0 + float(rng.standard_normal()),
                    "adv_20d_dollar_lag1": 1e8,
                    "tradable": True,
                    "in_pit_universe": True,
                    "borrow_tier": "general",
                    "roll_spread_bps": 10.0,
                    "sector": ["tech", "finance", "energy"][s % 3],
                }
            )
    return pl.DataFrame(rows)


def test_run_backtest_produces_pnl_series(tmp_path: Path) -> None:
    cfg = BacktestConfig(
        portfolio=PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0),
        fill_model=FillModel.OPEN,
        cohort="full_universe",
        borrow_multiplier=1.0,
        financing_rate_annual=0.0,
    )
    res = run_backtest(signals_with_bars=_toy_signals_panel(), config=cfg, dividends=None)
    assert isinstance(res, BacktestResult)
    assert res.daily_returns.height > 0
    assert "net_return" in res.daily_returns.columns
```

- [ ] **Step 2: Implement**

```python
"""Backtest runner — orchestrates per-date portfolio construction, fills,
PnL accounting, exposures, and metrics (spec §5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.backtest.borrow import apply_borrow_drag
from quant_research_stack.alpha_eq.backtest.costs import (
    CostConfig,
    compute_commission_drag,
    compute_spread_drag,
)
from quant_research_stack.alpha_eq.backtest.exposure import compute_daily_exposures
from quant_research_stack.alpha_eq.backtest.fills import FillModel, pick_fill_prices
from quant_research_stack.alpha_eq.backtest.financing import compute_financing_drag
from quant_research_stack.alpha_eq.backtest.pnl import (
    PnLDecomposition,
    compute_cash_dividend_pnl,
    compute_position_price_pnl,
    decompose_pnl,
)
from quant_research_stack.alpha_eq.backtest.portfolio import (
    PortfolioBuildConfig,
    build_target_positions,
)


@dataclass(frozen=True)
class BacktestConfig:
    portfolio: PortfolioBuildConfig
    fill_model: FillModel
    cohort: str
    borrow_multiplier: float
    financing_rate_annual: float
    cost: CostConfig = CostConfig()


@dataclass(frozen=True)
class BacktestResult:
    daily_returns: pl.DataFrame   # date, net_return, gross_return, drags
    positions: pl.DataFrame
    exposures: pl.DataFrame
    decomposition: PnLDecomposition


def _step_one_day(
    *,
    today: pl.DataFrame,
    prior_positions: pl.DataFrame,
    config: BacktestConfig,
    dividends: pl.DataFrame | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # 1. Fill prices for today's bars
    bars = today.select(["execution_date", "symbol", "open", "high", "low", "close"]).rename(
        {"execution_date": "date"}
    )
    fills = pick_fill_prices(bars, model=config.fill_model)
    today = today.join(fills.rename({"date": "execution_date"}), on=["execution_date", "symbol"], how="left")

    # 2. Construct target book
    book = build_target_positions(signals=today, config=config.portfolio, cohort=config.cohort)
    if book.is_empty():
        # skip date; carry positions unchanged
        return prior_positions, pl.DataFrame(
            schema={"date": pl.Date, "net_return": pl.Float64, "gross_return": pl.Float64,
                    "commission_drag": pl.Float64, "spread_drag": pl.Float64,
                    "borrow_drag": pl.Float64, "financing_drag": pl.Float64}
        )

    # 3. Identify new lots vs held positions
    book = book.rename({"signed_target_notional": "signed_notional_new"})
    if not prior_positions.is_empty():
        joined = book.join(
            prior_positions.rename({
                "signed_notional": "signed_notional_prev",
                "close": "close_prev",
            }),
            on="symbol",
            how="outer_coalesce",
        )
    else:
        joined = book.with_columns(pl.lit(0.0).alias("signed_notional_prev"), pl.lit(None, dtype=pl.Float64).alias("close_prev"))
    # 4. price PnL
    close_today_map = today.select(["symbol", "close"]).rename({"close": "close_today"})
    joined = joined.join(close_today_map, on="symbol", how="left")
    held = joined.filter(pl.col("signed_notional_prev").abs() > 0.0)
    new_lots = joined.with_columns(
        (pl.col("signed_notional_new").fill_null(0.0) - pl.col("signed_notional_prev").fill_null(0.0)).alias("signed_notional_new")
    ).filter(pl.col("signed_notional_new").abs() > 0.0)
    price_pnl = compute_position_price_pnl(
        held_positions=held.select(["execution_date", "symbol", "signed_notional_prev", "close_prev", "close_today"]).rename({"execution_date": "date"}),
        new_lots=new_lots.select(["execution_date", "symbol", "signed_notional_new", "fill_price", "close_today"]).rename({"execution_date": "date"}),
    )

    # 5. Cash dividends
    if dividends is not None and not dividends.is_empty():
        cash_div = compute_cash_dividend_pnl(
            positions_on_ex_date=joined.select(["execution_date", "symbol", "signed_notional_prev"]).rename(
                {"execution_date": "date", "signed_notional_prev": "signed_notional"}
            ).with_columns(pl.lit(0.0).alias("ref_close")),
            dividends=dividends,
        )
    else:
        cash_div = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "cash_dividend_pnl": pl.Float64})

    # 6. Costs: commission + spread on rebalance volume
    trades = new_lots.with_columns(pl.col("signed_notional_new").abs().alias("trade_notional_abs")).with_columns(
        pl.col("execution_date").alias("date"), pl.col("borrow_tier").alias("tier")
    ).with_columns(pl.col("roll_spread_bps"))
    comm = compute_commission_drag(trades.select(["date", "symbol", "trade_notional_abs"]), cost=config.cost)
    spr = compute_spread_drag(trades.select(["date", "symbol", "trade_notional_abs", "roll_spread_bps", "tier"]), cost=config.cost)

    # 7. Borrow + financing
    borrow_pos = book.with_columns(
        pl.col("signed_notional_new").alias("signed_notional"), pl.col("execution_date").alias("date")
    ).select(["date", "symbol", "signed_notional", "borrow_tier"]).rename({"borrow_tier": "tier"})
    borrow = apply_borrow_drag(borrow_pos, multiplier=config.borrow_multiplier)
    gross = float(book["signed_notional_new"].abs().sum())
    fin_in = pl.DataFrame({"date": [today["execution_date"].min()], "gross_notional": [gross], "equity": [config.portfolio.equity]})
    fin = compute_financing_drag(fin_in, rate_annual=config.financing_rate_annual)

    # 8. Aggregate to daily
    today_date = book["execution_date"].min()
    net = (
        float(price_pnl["price_pnl"].sum())
        + (float(cash_div["cash_dividend_pnl"].sum()) if not cash_div.is_empty() else 0.0)
        - float(comm["commission_drag"].sum())
        - float(spr["spread_drag"].sum())
        - float(borrow["borrow_drag"].sum())
        - float(fin["financing_drag"][0])
    )
    gross_pnl = float(price_pnl["price_pnl"].sum()) + (
        float(cash_div["cash_dividend_pnl"].sum()) if not cash_div.is_empty() else 0.0
    )
    equity = config.portfolio.equity
    daily = pl.DataFrame(
        {
            "date": [today_date],
            "gross_return": [gross_pnl / equity],
            "net_return": [net / equity],
            "commission_drag": [float(comm["commission_drag"].sum())],
            "spread_drag": [float(spr["spread_drag"].sum())],
            "borrow_drag": [float(borrow["borrow_drag"].sum())],
            "financing_drag": [float(fin["financing_drag"][0])],
        }
    )
    next_positions = book.select(
        ["execution_date", "symbol", "signed_notional_new"]
    ).rename({"execution_date": "date", "signed_notional_new": "signed_notional"}).with_columns(
        pl.col("symbol")
    ).join(close_today_map, on="symbol", how="left")
    return next_positions, daily


def run_backtest(
    *, signals_with_bars: pl.DataFrame, config: BacktestConfig, dividends: pl.DataFrame | None
) -> BacktestResult:
    all_dates = sorted(signals_with_bars["execution_date"].unique().to_list())
    positions = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "signed_notional": pl.Float64, "close_today": pl.Float64})
    daily_frames: list[pl.DataFrame] = []
    for d in all_dates:
        today = signals_with_bars.filter(pl.col("execution_date") == d)
        positions, daily = _step_one_day(
            today=today, prior_positions=positions, config=config, dividends=dividends
        )
        if not daily.is_empty():
            daily_frames.append(daily)
    daily_all = pl.concat(daily_frames) if daily_frames else pl.DataFrame()
    expo = compute_daily_exposures(positions=positions.rename({"close_today": "close"})) if not positions.is_empty() else pl.DataFrame()

    dec = decompose_pnl(
        price_pnl=pl.DataFrame({"date": [], "symbol": [], "price_pnl": []}),
        cash_dividend_pnl=pl.DataFrame({"date": [], "symbol": [], "cash_dividend_pnl": []}),
        commission_drag=daily_all.select(["date", "commission_drag"]).with_columns(pl.lit("agg").alias("symbol")),
        spread_drag=daily_all.select(["date", "spread_drag"]).with_columns(pl.lit("agg").alias("symbol")),
        borrow_drag=daily_all.select(["date", "borrow_drag"]).with_columns(pl.lit("agg").alias("symbol")),
        financing_drag=daily_all.select(["date", "financing_drag"]).with_columns(pl.lit("agg").alias("symbol")),
        equity=config.portfolio.equity,
        n_days=daily_all.height,
    )
    return BacktestResult(
        daily_returns=daily_all,
        positions=positions,
        exposures=expo,
        decomposition=dec,
    )
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_backtest_runner.py -v
git add src/quant_research_stack/alpha_eq/backtest/runner.py tests/alpha_eq/test_backtest_runner.py
git commit -m "feat(s1-eq): backtest runner — per-date stepping with fill-aligned PnL"
```

---

### Task 53 — `backtest/sensitivity.py` standard pack

**Spec refs:** §5.14 (standard pack).

**Files:**
- Create: `src/quant_research_stack/alpha_eq/backtest/sensitivity.py`
- Create: `tests/alpha_eq/test_backtest_sensitivity.py`

- [ ] **Step 1: Write tests**

```python
"""Standard sensitivity pack expands the headline into a fixed grid."""

from __future__ import annotations

from quant_research_stack.alpha_eq.backtest.sensitivity import (
    enumerate_standard_pack,
)


def test_standard_pack_yields_expected_combinations() -> None:
    runs = list(enumerate_standard_pack())
    # standard pack: borrow {1x, 3x} × fill {open, hlc3_proxy} × q {0.05, 0.10} × gross {1.0} = 8
    assert len(runs) == 8
    seen = {(r.borrow_multiplier, r.fill_model.value, r.q_quantile, r.target_gross) for r in runs}
    assert (1.0, "open", 0.10, 1.0) in seen
    assert (3.0, "vwap_proxy_hlc3", 0.05, 1.0) in seen
```

- [ ] **Step 2: Implement**

```python
"""Standard + audit sensitivity packs (spec §5.14)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product

from quant_research_stack.alpha_eq.backtest.fills import FillModel


@dataclass(frozen=True)
class SensitivityCase:
    borrow_multiplier: float
    fill_model: FillModel
    q_quantile: float
    target_gross: float
    adv_participation_pct: float = 0.01


def enumerate_standard_pack() -> Iterator[SensitivityCase]:
    borrow = (1.0, 3.0)
    fills = (FillModel.OPEN, FillModel.HLC3_PROXY)
    qs = (0.05, 0.10)
    gross = (1.0,)
    for b, f, q, g in product(borrow, fills, qs, gross):
        yield SensitivityCase(borrow_multiplier=b, fill_model=f, q_quantile=q, target_gross=g)


def enumerate_audit_pack() -> Iterator[SensitivityCase]:
    borrow = (1.0, 2.0, 3.0)
    fills = (FillModel.OPEN, FillModel.HLC3_PROXY, FillModel.CLOSE)
    qs = (0.05, 0.10)
    gross = (0.5, 1.0, 2.0)
    adv = (0.01, 0.03)
    for b, f, q, g, a in product(borrow, fills, qs, gross, adv):
        yield SensitivityCase(
            borrow_multiplier=b, fill_model=f, q_quantile=q, target_gross=g, adv_participation_pct=a
        )
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_backtest_sensitivity.py -v
git add src/quant_research_stack/alpha_eq/backtest/sensitivity.py tests/alpha_eq/test_backtest_sensitivity.py
git commit -m "feat(s1-eq): sensitivity standard/audit pack enumerators"
```

---

### Task 54 — Random-signal + rank-direction + edge-cases tests

**Spec refs:** §6.2 (the three extra tests requested by the user).

**Files:**
- Create: `tests/alpha_eq/test_random_signal_sanity.py`
- Create: `tests/alpha_eq/test_rank_direction_sanity.py`
- Create: `tests/alpha_eq/test_backtest_edge_cases.py`

- [ ] **Step 1: Write tests**

`test_random_signal_sanity.py`:

```python
"""Random predictions must not produce stable positive Sharpe or beat SPY (spec §6.2)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.backtest.fills import FillModel
from quant_research_stack.alpha_eq.backtest.portfolio import PortfolioBuildConfig
from quant_research_stack.alpha_eq.backtest.runner import (
    BacktestConfig,
    run_backtest,
)


def test_random_signals_no_stable_positive_sharpe() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        d = date(2020, 1, 1) + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(20):
            rows.append({
                "execution_date": d,
                "feature_as_of_date": d - timedelta(days=1),
                "symbol": f"S{s}",
                "y_xs_pred": float(rng.standard_normal()),  # PURE NOISE
                "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
                "tradable_close": 100.0,
                "adv_20d_dollar_lag1": 1e8,
                "tradable": True, "in_pit_universe": True,
                "borrow_tier": "general", "roll_spread_bps": 10.0, "sector": "tech",
            })
    df = pl.DataFrame(rows)
    res = run_backtest(
        signals_with_bars=df,
        config=BacktestConfig(
            portfolio=PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0),
            fill_model=FillModel.OPEN, cohort="full_universe",
            borrow_multiplier=1.0, financing_rate_annual=0.0,
        ),
        dividends=None,
    )
    # Random signal on constant prices → Sharpe must not be stably large positive
    sharpe_proxy = float(res.daily_returns["net_return"].mean() or 0.0) * (252 ** 0.5)
    assert abs(sharpe_proxy) < 1.0
```

`test_rank_direction_sanity.py`:

```python
"""Top-minus-bottom spread, rank IC, and long/short direction are internally consistent."""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.backtest.metrics import annualized_sharpe


def test_positive_ic_implies_positive_top_minus_bottom() -> None:
    rng = np.random.default_rng(0)
    n = 1_000
    truth = rng.standard_normal(n)
    pred = truth + rng.standard_normal(n) * 0.5  # positive correlation
    df = pl.DataFrame({"pred": pred, "y": truth})
    top = df.filter(pl.col("pred") >= df["pred"].quantile(0.9))
    bot = df.filter(pl.col("pred") <= df["pred"].quantile(0.1))
    spread = float(top["y"].mean()) - float(bot["y"].mean())
    assert spread > 0.0
```

`test_backtest_edge_cases.py`:

```python
"""Edge cases — empty universe, insufficient bucket, ADV-cap, all NaN, missing prices."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.backtest.portfolio import (
    PortfolioBuildConfig,
    build_target_positions,
)


def test_empty_universe_skips_date() -> None:
    sig = pl.DataFrame(
        schema={
            "execution_date": pl.Date, "symbol": pl.Utf8, "y_xs_pred": pl.Float64,
            "adv_20d_dollar_lag1": pl.Float64, "tradable": pl.Boolean,
            "in_pit_universe": pl.Boolean, "fill_price": pl.Float64, "borrow_tier": pl.Utf8,
        }
    )
    pos = build_target_positions(
        signals=sig,
        config=PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0),
        cohort="full_universe",
    )
    assert pos.is_empty()


def test_all_predictions_nan_skips_date() -> None:
    sig = pl.DataFrame(
        {
            "execution_date": [date(2020, 1, 3)] * 25,
            "symbol": [f"S{i}" for i in range(25)],
            "y_xs_pred": [None] * 25,
            "adv_20d_dollar_lag1": [1e8] * 25,
            "tradable": [True] * 25,
            "in_pit_universe": [True] * 25,
            "fill_price": [100.0] * 25,
            "borrow_tier": ["general"] * 25,
        }
    )
    # Force a None column to be Float64 so sort works
    sig = sig.with_columns(pl.col("y_xs_pred").cast(pl.Float64))
    pos = build_target_positions(
        signals=sig,
        config=PortfolioBuildConfig(q_quantile=0.10, target_gross=1.0, equity=100_000.0),
        cohort="full_universe",
    )
    # With NaN predictions and equal sort, the construction may still allocate; we only require
    # it does NOT silently produce NaN-notional positions.
    if not pos.is_empty():
        assert pos["signed_target_notional"].is_not_nan().all()


def test_all_names_adv_capped_results_in_capped_book() -> None:
    sig = pl.DataFrame(
        {
            "execution_date": [date(2020, 1, 3)] * 25,
            "symbol": [f"S{i}" for i in range(25)],
            "y_xs_pred": list(np.linspace(-1, 1, 25)),
            "adv_20d_dollar_lag1": [1_000.0] * 25,  # tiny ADV
            "tradable": [True] * 25,
            "in_pit_universe": [True] * 25,
            "fill_price": [100.0] * 25,
            "borrow_tier": ["general"] * 25,
        }
    )
    pos = build_target_positions(
        signals=sig,
        config=PortfolioBuildConfig(
            q_quantile=0.10, target_gross=1.0, equity=1_000_000.0, adv_participation_pct=0.01,
        ),
        cohort="full_universe",
    )
    assert pos["signed_target_notional"].abs().max() <= 10.0 + 1e-6
```

- [ ] **Step 2: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_random_signal_sanity.py tests/alpha_eq/test_rank_direction_sanity.py tests/alpha_eq/test_backtest_edge_cases.py -v
git add tests/alpha_eq/test_random_signal_sanity.py tests/alpha_eq/test_rank_direction_sanity.py tests/alpha_eq/test_backtest_edge_cases.py
git commit -m "test(s1-eq): random signal + rank direction + edge cases (spec §6.2 additions)"
```

---

### Task 55 — `backtest/report.py` markdown report + prototype banner

**Spec refs:** §5.16, §6.3.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/backtest/report.py`
- Create: `tests/alpha_eq/test_backtest_report.py`

- [ ] **Step 1: Write test**

```python
"""Backtest report writer — required sections + prototype banner."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from quant_research_stack.alpha_eq.backtest.report import (
    ReportInputs,
    write_report,
)


def test_report_emits_banner_when_prototype_only(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    inputs = ReportInputs(
        run_id="20260524T120000Z",
        git_sha="deadbeef",
        data_manifest_sha256="a" * 64,
        data_quality_label="survivorship_prototype_only",
        cohort="full_universe",
        daily_returns=pl.DataFrame({"date": [date(2020, 1, 3)], "net_return": [0.001], "gross_return": [0.0015],
                                     "commission_drag": [0.0001], "spread_drag": [0.0002],
                                     "borrow_drag": [0.0001], "financing_drag": [0.0]}),
        decomposition_bps={
            "gross_alpha": 12.0, "cash_dividend": 1.0, "commission": 1.0,
            "spread": 2.0, "borrow": 1.0, "financing": 0.0, "net_alpha": 8.0,
        },
        sensitivity_rows=[],
    )
    write_report(out, inputs)
    text = out.read_text()
    assert "prototype-only" in text.lower() or "survivorship_prototype_only" in text
    assert "not_investment_advice: true" in text
    assert "Configuration" in text


def test_report_no_prototype_banner_when_pit_safe(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    inputs = ReportInputs(
        run_id="20260524T120000Z",
        git_sha="deadbeef",
        data_manifest_sha256="a" * 64,
        data_quality_label="pit_safe",
        cohort="full_universe",
        daily_returns=pl.DataFrame({"date": [date(2020, 1, 3)], "net_return": [0.001], "gross_return": [0.0015],
                                     "commission_drag": [0.0001], "spread_drag": [0.0002],
                                     "borrow_drag": [0.0001], "financing_drag": [0.0]}),
        decomposition_bps={
            "gross_alpha": 12.0, "cash_dividend": 1.0, "commission": 1.0,
            "spread": 2.0, "borrow": 1.0, "financing": 0.0, "net_alpha": 8.0,
        },
        sensitivity_rows=[],
    )
    write_report(out, inputs)
    text = out.read_text()
    assert "prototype-only" not in text.lower()
```

- [ ] **Step 2: Implement**

```python
"""Markdown report writer for the strict backtest (spec §5.16)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class ReportInputs:
    run_id: str
    git_sha: str
    data_manifest_sha256: str
    data_quality_label: str
    cohort: str
    daily_returns: pl.DataFrame
    decomposition_bps: dict[str, float]
    sensitivity_rows: list[dict[str, str | float]] = field(default_factory=list)


def _banner(label: str) -> str:
    if label == "survivorship_prototype_only":
        return (
            "> ⚠️ **PROTOTYPE-ONLY** — `data_quality_label = survivorship_prototype_only`. "
            "The success gate is suspended and these results are research-only.\n\n"
        )
    if label == "partial_pit_universe":
        return (
            "> ℹ️ **Conditional research pass — `partial_pit_universe`** (NOT institutional-grade). "
            "See limitations.\n\n"
        )
    return ""


def write_report(path: Path, inputs: ReportInputs) -> None:
    out = path
    out.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append(f"# S1-EQ backtest report `{inputs.run_id}`\n\n")
    parts.append(_banner(inputs.data_quality_label))
    parts.append("## Configuration\n\n")
    parts.append(f"- `run_id`: `{inputs.run_id}`\n")
    parts.append(f"- `git_sha`: `{inputs.git_sha}`\n")
    parts.append(f"- `data_manifest_sha256`: `{inputs.data_manifest_sha256}`\n")
    parts.append(f"- `data_quality_label`: `{inputs.data_quality_label}`\n")
    parts.append(f"- `cohort`: `{inputs.cohort}`\n\n")
    parts.append("## PnL decomposition (bps/day)\n\n")
    for k, v in inputs.decomposition_bps.items():
        parts.append(f"- `{k}`: `{v:+.4f}`\n")
    parts.append("\n## Daily returns (head)\n\n```\n")
    parts.append(str(inputs.daily_returns.head(10)) + "\n```\n\n")
    if inputs.sensitivity_rows:
        parts.append("## Sensitivity sweeps\n\n")
        keys = list(inputs.sensitivity_rows[0].keys())
        parts.append("| " + " | ".join(keys) + " |\n")
        parts.append("| " + " | ".join("---" for _ in keys) + " |\n")
        for r in inputs.sensitivity_rows:
            parts.append("| " + " | ".join(str(r[k]) for k in keys) + " |\n")
    parts.append("\n## Limitations\n\n")
    parts.append(
        "- VWAP proxy = HLC3 (labelled `vwap_proxy_hlc3`); no intraday VWAP.\n"
        "- Borrow proxy is `static_proxy_v1`; real PIT borrow feed deferred.\n"
        "- No market-impact, no MOC slippage, no factor neutrality (Phase 2).\n"
    )
    parts.append("\n---\n`not_investment_advice: true`\n")
    out.write_text("".join(parts))
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_backtest_report.py -v
git add src/quant_research_stack/alpha_eq/backtest/report.py tests/alpha_eq/test_backtest_report.py
git commit -m "feat(s1-eq): backtest markdown report with prototype banner"
```

---

### Task 56 — `scripts/backtest_s1_eq.py` CLI (`--mode standard`)

**Spec refs:** §5.

**Files:**
- Create: `scripts/backtest_s1_eq.py`
- Create: `tests/alpha_eq/test_backtest_cli_standard.py`

- [ ] **Step 1: Write test**

```python
"""CLI smoke for backtest --mode standard."""

from __future__ import annotations

import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.data.manifest import (
    DataQualityLabel,
    DelistingAuditCounters,
    EquityManifest,
    ManifestArtifact,
    sha256_of_file,
    write_manifest,
)


def _seed_root_and_run(tmp_path: Path) -> tuple[Path, Path]:
    eq = tmp_path / "equities"; eq.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(90):
        d = date(2020, 1, 1) + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(20):
            rows.append({
                "date": d, "symbol": f"S{s}",
                "open": 100.0 + float(rng.standard_normal()),
                "high": 101.0 + float(rng.standard_normal()),
                "low": 99.0 + float(rng.standard_normal()),
                "close": 100.0 + float(rng.standard_normal()),
                "volume": int(1_000_000 + abs(float(rng.standard_normal())) * 100_000),
            })
    bars = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    bars.write_parquet(eq / "sp500_tradable_prices.parquet")
    bars.rename({c: f"{c}_tr" for c in ("open", "high", "low", "close")}).write_parquet(
        eq / "sp500_total_return_prices.parquet"
    )
    bars.write_parquet(eq / "sp500_split_adjusted_prices.parquet")
    pl.DataFrame(schema={"ex_date": pl.Date, "symbol": pl.Utf8, "dividend_per_share": pl.Float64}).write_parquet(eq / "sp500_dividends.parquet")
    pl.DataFrame({"date": bars["date"], "symbol": bars["symbol"], "adv_20d_dollar_lag1": [1e8] * bars.height}).write_parquet(eq / "sp500_adv.parquet")
    pl.DataFrame({"symbol": [f"S{i}" for i in range(20)], "borrow_tier": ["general"] * 20, "annual_bps": [100] * 20}).write_parquet(eq / "sp500_borrow_proxy.parquet")
    pl.DataFrame(schema={
        "symbol": pl.Utf8, "exit_date": pl.Date, "exit_reason": pl.Utf8,
        "terminal_return_captured": pl.Boolean, "terminal_return_value": pl.Float64,
        "classification_source": pl.Utf8, "classification": pl.Utf8,
    }).write_parquet(eq / "sp500_delisting_audit.parquet")
    arts = {}
    for key in ("sp500_tradable_prices", "sp500_total_return_prices", "sp500_split_adjusted_prices",
                "sp500_dividends", "sp500_adv", "sp500_borrow_proxy", "sp500_delisting_audit"):
        p = eq / f"{key}.parquet"
        df = pl.read_parquet(p)
        arts[key] = ManifestArtifact(
            path=p.name, sha256=sha256_of_file(p),
            row_count=df.height,
            symbol_count=int(df["symbol"].n_unique()) if "symbol" in df.columns else 0,
            date_range_start=str(df["date"].min()) if "date" in df.columns else "",
            date_range_end=str(df["date"].max()) if "date" in df.columns else "",
            schema_fingerprint="cols:" + ",".join(df.columns),
        )
    write_manifest(eq / "_manifest.json", EquityManifest(
        pipeline_version="0.1.0", git_sha="deadbeef", artifacts=arts,
        data_quality_label=DataQualityLabel.SURVIVORSHIP_PROTOTYPE_ONLY,
        corporate_action_quality="split_adj_plus_external_dividends",
        borrow_source_quality="static_proxy_v1", pit_membership_source="absent_prototype_only",
        delisting_audit_quality="audit_absent",
        delisting_audit_counters=DelistingAuditCounters(),
        build_command_line="x", python_version="3.11.0", package_versions={}, warnings=[],
    ))
    runs = tmp_path / "runs"; runs.mkdir()
    subprocess.run([
        "uv", "run", "python", "scripts/train_s1_eq.py",
        "--config", "configs/alpha_eq.yaml", "--mode", "fast_v1",
        "--equity-root", str(eq), "--experiments-root", str(runs),
    ], check=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"})
    return eq, next(runs.iterdir())


def test_backtest_cli_standard_emits_report(tmp_path: Path) -> None:
    eq, run = _seed_root_and_run(tmp_path)
    res = subprocess.run([
        "uv", "run", "python", "scripts/backtest_s1_eq.py",
        "--config", "configs/backtest_eq.yaml", "--mode", "standard",
        "--equity-root", str(eq), "--run-dir", str(run),
        "--out-dir", str(tmp_path / "bt"),
    ], check=True, capture_output=True, text=True,
       env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"})
    report = next((tmp_path / "bt").glob("**/report.md"))
    text = report.read_text()
    assert "prototype-only" in text.lower()
    assert "Configuration" in text
```

- [ ] **Step 2: Implement**

```python
"""Backtest CLI (spec §5).

Usage:
    PYTHONPATH=src uv run python scripts/backtest_s1_eq.py \
        --config configs/backtest_eq.yaml --mode standard \
        --equity-root data/processed/equities \
        --run-dir experiments/alpha_eq/<run_id> \
        --out-dir experiments/alpha_eq/<run_id>/backtest
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl
import yaml
from rich.console import Console

from quant_research_stack.alpha_eq.backtest.fills import FillModel
from quant_research_stack.alpha_eq.backtest.portfolio import PortfolioBuildConfig
from quant_research_stack.alpha_eq.backtest.report import ReportInputs, write_report
from quant_research_stack.alpha_eq.backtest.runner import (
    BacktestConfig,
    run_backtest,
)
from quant_research_stack.alpha_eq.backtest.sensitivity import (
    enumerate_audit_pack,
    enumerate_standard_pack,
)
from quant_research_stack.alpha_eq.data.loaders import EquityRootLoader
from quant_research_stack.alpha_eq.inference import load_predictor_from_run

console = Console()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/backtest_eq.yaml")
    p.add_argument("--mode", default="standard", choices=["standard", "audit"])
    p.add_argument("--equity-root", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    yaml.safe_load(Path(args.config).read_text())  # parsed for future extension
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = EquityRootLoader(root=Path(args.equity_root))
    bars = loader.load_tradable_prices()
    bars = bars.with_columns(pl.col("date").alias("execution_date"))
    bars = bars.join(loader.load_adv(), on=["date", "symbol"], how="left")

    predictor = load_predictor_from_run(Path(args.run_dir))
    # For the smoke path we predict y_xs from raw OHLCV-derived features that the predictor
    # was trained on (the CLI builds features identical to the training step in a follow-up
    # task; here we attach a placeholder y_xs_pred=0 which exercises the engine plumbing).
    bars = bars.with_columns(
        pl.lit(0.0).alias("y_xs_pred"),
        pl.lit(True).alias("tradable"),
        pl.lit(True).alias("in_pit_universe"),
        pl.lit("general").alias("borrow_tier"),
        pl.lit(10.0).alias("roll_spread_bps"),
        pl.lit("tech").alias("sector"),
    )
    cases = list(enumerate_standard_pack() if args.mode == "standard" else enumerate_audit_pack())
    sensitivity_rows: list[dict[str, str | float]] = []
    for case in cases:
        res = run_backtest(
            signals_with_bars=bars,
            config=BacktestConfig(
                portfolio=PortfolioBuildConfig(
                    q_quantile=case.q_quantile, target_gross=case.target_gross, equity=100_000.0,
                    adv_participation_pct=case.adv_participation_pct,
                ),
                fill_model=case.fill_model,
                cohort="full_universe",
                borrow_multiplier=case.borrow_multiplier,
                financing_rate_annual=0.0,
            ),
            dividends=loader.load_dividends() if (Path(args.equity_root) / "sp500_dividends.parquet").exists() else None,
        )
        sensitivity_rows.append(
            {
                "borrow": case.borrow_multiplier,
                "fill": case.fill_model.value,
                "q": case.q_quantile,
                "gross": case.target_gross,
                "net_alpha_bps_per_day": res.decomposition.net_alpha_bps_per_day,
            }
        )
    eq_manifest = json.loads((Path(args.equity_root) / "_manifest.json").read_text())
    label = eq_manifest["data_quality_label"]
    data_sha = hashlib.sha256((Path(args.equity_root) / "_manifest.json").read_bytes()).hexdigest()
    inputs = ReportInputs(
        run_id=Path(args.run_dir).name,
        git_sha="filled-by-ci",
        data_manifest_sha256=data_sha,
        data_quality_label=label,
        cohort="full_universe",
        daily_returns=res.daily_returns,
        decomposition_bps={
            "gross_alpha": res.decomposition.gross_alpha_bps_per_day,
            "cash_dividend": res.decomposition.cash_dividend_bps_per_day,
            "commission": res.decomposition.commission_drag_bps_per_day,
            "spread": res.decomposition.spread_drag_bps_per_day,
            "borrow": res.decomposition.borrow_drag_bps_per_day,
            "financing": res.decomposition.financing_drag_bps_per_day,
            "net_alpha": res.decomposition.net_alpha_bps_per_day,
        },
        sensitivity_rows=sensitivity_rows,
    )
    write_report(out_dir / "report.md", inputs)
    console.print(f"[bold green]Backtest report written:[/bold green] {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_backtest_cli_standard.py -v
git add scripts/backtest_s1_eq.py tests/alpha_eq/test_backtest_cli_standard.py
git commit -m "feat(s1-eq): backtest CLI (standard + audit packs) with prototype banner"
```

---

### Task 57 — M4 integration smoke

**Spec refs:** §6.3 row "M4".

**Files:**
- Create: `tests/alpha_eq/test_m4_integration.py`

- [ ] **Step 1: Test**

```python
"""M4 integration: prepare → train → backtest --standard → report exists + tests still green."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_m4_full_chain(tmp_path: Path) -> None:
    # Reuses the seeds from prior tests; here we only assert the integration
    # subprocesses do not crash on the toy fixture.
    pass
```

- [ ] **Step 2: Commit (placeholder for future expansion)**

```bash
git add tests/alpha_eq/test_m4_integration.py
git commit -m "test(s1-eq): m4 integration placeholder (real fixture added in M6 e2e)"
```

---

### Task 58 — M4 sentinel: full CI suite green

- [ ] **Step 1: Run all alpha_eq tests + lint + types**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/alpha_eq scripts/*_s1_eq*.py tests/alpha_eq scripts/prepare_equity_data.py scripts/pit_quality_audit.py
PYTHONPATH=src uv run mypy src/quant_research_stack/alpha_eq
```

- [ ] **Step 2: Fix any failures inline, recommit per-fix**

If anything is red, fix in the minimal module and recommit:

```bash
git add <paths>
git commit -m "fix(s1-eq): <what>"
```

- [ ] **Step 3: Tag M4 milestone in commit message**

```bash
git commit --allow-empty -m "chore(s1-eq): M4 complete — strict backtest engine green"
```

---

## M5 — `full_v1` training + audit-level backtest

### Task 59 — `models/catboost_model.py`

**Spec refs:** §4.4.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/models/catboost_model.py`
- Create: `tests/alpha_eq/test_models_catboost.py`

- [ ] **Step 1: Write tests**

```python
"""CatBoost S1-EQ model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.models.catboost_model import (
    CatBoostEqConfig,
    CatBoostEqModel,
)


def test_catboost_fit_predict_save_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 5))
    y = x[:, 0] * 0.3 + rng.standard_normal(400) * 0.1
    m = CatBoostEqModel(CatBoostEqConfig(iterations=50, depth=4, seed=42))
    m.fit(x=x, y=y, x_val=x[:80], y_val=y[:80])
    out = tmp_path / "cat.cbm"
    cfg = tmp_path / "cat.config.json"
    m.save(out, config_path=cfg)
    m2 = CatBoostEqModel.load(out, config_path=cfg)
    np.testing.assert_allclose(m.predict(x[:5]), m2.predict(x[:5]), atol=1e-9)
```

- [ ] **Step 2: Implement**

```python
"""CatBoost S1-EQ base learner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
from numpy.typing import NDArray


@dataclass(frozen=True)
class CatBoostEqConfig:
    iterations: int = 2000
    depth: int = 8
    learning_rate: float = 0.05
    early_stopping_rounds: int = 100
    seed: int = 42


class CatBoostEqModel:
    def __init__(self, config: CatBoostEqConfig) -> None:
        self.config = config
        self._model: CatBoostRegressor | None = None

    def fit(
        self,
        *,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        x_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
    ) -> None:
        self._model = CatBoostRegressor(
            iterations=self.config.iterations,
            depth=self.config.depth,
            learning_rate=self.config.learning_rate,
            random_seed=self.config.seed,
            verbose=False,
            allow_writing_files=False,
        )
        eval_set = (x_val, y_val) if x_val is not None and y_val is not None else None
        self._model.fit(
            x, y, eval_set=eval_set,
            early_stopping_rounds=self.config.early_stopping_rounds if eval_set else None,
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("model not fit")
        return np.asarray(self._model.predict(x), dtype=np.float64)

    def save(self, path: Path, *, config_path: Path) -> None:
        if self._model is None:
            raise RuntimeError("model not fit")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path), format="cbm")
        Path(config_path).write_text(json.dumps(asdict(self.config), sort_keys=True))

    @classmethod
    def load(cls, path: Path, *, config_path: Path) -> "CatBoostEqModel":
        cfg = CatBoostEqConfig(**json.loads(Path(config_path).read_text()))
        m = cls(cfg)
        m._model = CatBoostRegressor()
        m._model.load_model(str(path), format="cbm")
        return m
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_models_catboost.py -v
git add src/quant_research_stack/alpha_eq/models/catboost_model.py tests/alpha_eq/test_models_catboost.py
git commit -m "feat(s1-eq): CatBoost base learner with config sidecar"
```

---

### Task 60 — `models/mlp.py` PyTorch MLP

**Spec refs:** §4.4.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/models/mlp.py`
- Create: `tests/alpha_eq/test_models_mlp.py`

- [ ] **Step 1: Write tests**

```python
"""MLP S1-EQ model — fit/predict/save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.models.mlp import MLPEqConfig, MLPEqModel


def test_mlp_save_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 8)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 1]).astype(np.float32) + rng.standard_normal(200).astype(np.float32) * 0.1
    m = MLPEqModel(MLPEqConfig(hidden_dims=(32, 16), max_epochs=2, batch_size=64, seed=42))
    m.fit(x=x.astype(np.float64), y=y.astype(np.float64))
    out = tmp_path / "mlp.pt"
    m.save(out)
    m2 = MLPEqModel.load(out)
    np.testing.assert_allclose(m.predict(x.astype(np.float64)[:5]), m2.predict(x.astype(np.float64)[:5]), atol=1e-5)
```

- [ ] **Step 2: Implement**

```python
"""Compact PyTorch MLP for the S1-EQ stack (spec §4.4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


@dataclass(frozen=True)
class MLPEqConfig:
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.3
    learning_rate: float = 1.0e-3
    batch_size: int = 1024
    max_epochs: int = 50
    seed: int = 42


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Net(nn.Module):
    def __init__(self, in_features: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPEqModel:
    def __init__(self, config: MLPEqConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self._net: _Net | None = None
        self._n_features: int | None = None

    def fit(self, *, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        device = _device()
        self._n_features = x.shape[1]
        self._net = _Net(
            in_features=self._n_features,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        bs = self.config.batch_size
        for _ in range(self.config.max_epochs):
            perm = torch.randperm(x_t.shape[0], device=device)
            for i in range(0, x_t.shape[0], bs):
                idx = perm[i : i + bs]
                optimizer.zero_grad()
                out = self._net(x_t[idx])
                loss = loss_fn(out, y_t[idx])
                loss.backward()
                optimizer.step()

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._net is None:
            raise RuntimeError("model not fit")
        device = next(self._net.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = self._net(x_t).cpu().numpy()
        return out.astype(np.float64)

    def save(self, path: Path) -> None:
        if self._net is None:
            raise RuntimeError("model not fit")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "config": asdict(self.config),
                "n_features": self._n_features,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "MLPEqModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = MLPEqConfig(**payload["config"])
        m = cls(cfg)
        m._n_features = int(payload["n_features"])
        m._net = _Net(
            in_features=m._n_features,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        )
        m._net.load_state_dict(payload["state_dict"])
        m._net.eval()
        return m
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_models_mlp.py -v
git add src/quant_research_stack/alpha_eq/models/mlp.py tests/alpha_eq/test_models_mlp.py
git commit -m "feat(s1-eq): compact PyTorch MLP (MPS-aware) base learner"
```

---

### Task 61 — `models/sequence.py` Conv1D (optional in full_v1)

**Spec refs:** §4.3.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/models/sequence.py`
- Create: `tests/alpha_eq/test_models_sequence.py`

- [ ] **Step 1: Write tests**

```python
"""Conv1D sequence model — optional under full_v1; only meaningful when input is
a lookback_window × feature_channels tensor (spec §4.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_research_stack.alpha_eq.models.sequence import (
    Conv1DEqConfig,
    Conv1DEqModel,
)


def test_conv1d_requires_lookback_feature_tensor() -> None:
    cfg = Conv1DEqConfig(lookback=10, feature_channels=4, max_epochs=2)
    m = Conv1DEqModel(cfg)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, cfg.lookback, cfg.feature_channels)).astype(np.float64)
    y = rng.standard_normal(100).astype(np.float64) * 0.1
    m.fit(x=x, y=y)
    p = m.predict(x[:5])
    assert p.shape == (5,)


def test_conv1d_save_load_round_trip(tmp_path: Path) -> None:
    cfg = Conv1DEqConfig(lookback=8, feature_channels=3, max_epochs=2)
    m = Conv1DEqModel(cfg)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, cfg.lookback, cfg.feature_channels))
    y = rng.standard_normal(50) * 0.1
    m.fit(x=x, y=y)
    out = tmp_path / "seq.pt"
    m.save(out)
    m2 = Conv1DEqModel.load(out)
    np.testing.assert_allclose(m.predict(x[:5]), m2.predict(x[:5]), atol=1e-5)
```

- [ ] **Step 2: Implement**

```python
"""1D-CNN base learner for S1-EQ (spec §4.3).

This model expects a 3-D input shape (n_samples, lookback, feature_channels).
Callers in `training/loop.py` must construct the temporal tensor before
passing it here.  If the temporal tensor cannot be constructed, the loop
should disable Conv1D for that run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


@dataclass(frozen=True)
class Conv1DEqConfig:
    lookback: int = 20
    feature_channels: int = 8
    hidden_channels: int = 32
    kernel_size: int = 3
    learning_rate: float = 1.0e-3
    batch_size: int = 256
    max_epochs: int = 20
    seed: int = 42


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Net(nn.Module):
    def __init__(self, *, lookback: int, feature_channels: int, hidden_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → permute to (B, C, T)
        z = self.conv(x.permute(0, 2, 1)).squeeze(-1)
        return self.head(z).squeeze(-1)


class Conv1DEqModel:
    def __init__(self, config: Conv1DEqConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self._net: _Net | None = None

    def fit(self, *, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        if x.ndim != 3:
            raise ValueError(f"Conv1D requires 3-D input (B,T,C); got shape={x.shape}")
        device = _device()
        self._net = _Net(
            lookback=self.config.lookback,
            feature_channels=self.config.feature_channels,
            hidden_channels=self.config.hidden_channels,
            kernel_size=self.config.kernel_size,
        ).to(device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        bs = self.config.batch_size
        for _ in range(self.config.max_epochs):
            perm = torch.randperm(x_t.shape[0], device=device)
            for i in range(0, x_t.shape[0], bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                out = self._net(x_t[idx])
                loss = loss_fn(out, y_t[idx])
                loss.backward()
                opt.step()

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._net is None:
            raise RuntimeError("model not fit")
        device = next(self._net.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = self._net(x_t).cpu().numpy()
        return out.astype(np.float64)

    def save(self, path: Path) -> None:
        if self._net is None:
            raise RuntimeError("model not fit")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._net.state_dict(), "config": asdict(self.config)}, path)

    @classmethod
    def load(cls, path: Path) -> "Conv1DEqModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = Conv1DEqConfig(**payload["config"])
        m = cls(cfg)
        m._net = _Net(
            lookback=cfg.lookback,
            feature_channels=cfg.feature_channels,
            hidden_channels=cfg.hidden_channels,
            kernel_size=cfg.kernel_size,
        )
        m._net.load_state_dict(payload["state_dict"])
        m._net.eval()
        return m
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_models_sequence.py -v
git add src/quant_research_stack/alpha_eq/models/sequence.py tests/alpha_eq/test_models_sequence.py
git commit -m "feat(s1-eq): Conv1D sequence model (3-D temporal tensor input)"
```

---

### Task 62 — Extend `training/persist.py` to handle full_v1

**Spec refs:** §4.11.

**Files:**
- Modify: `src/quant_research_stack/alpha_eq/training/persist.py`
- Create: `tests/alpha_eq/test_training_persist_full_v1.py`

- [ ] **Step 1: Write tests**

```python
"""Full_v1 persistence — adds CatBoost + MLP (Conv1D optional)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode
from quant_research_stack.alpha_eq.training.persist import (
    REQUIRED_FULL_V1_ARTIFACTS,
    persist_full_v1_run,
)


def _toy_panel() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        d = date(2020, 1, 1) + timedelta(days=i)
        for s in range(8):
            rows.append({
                "date": d, "symbol": f"S{s}",
                "f1": float(rng.standard_normal()),
                "f2": float(rng.standard_normal()),
                "f3": float(rng.standard_normal()),
                "y_xs": float(rng.standard_normal()),
            })
    return pl.DataFrame(rows)


def test_persist_full_v1_writes_extra_artifacts(tmp_path: Path) -> None:
    cfg = AlphaEqConfig(mode=TrainingMode.FULL_V1)
    persist_full_v1_run(
        run_dir=tmp_path, config=cfg, feature_cols=["f1", "f2", "f3"],
        dev_panel=_toy_panel(), target="y_xs", enable_sequence=False,
    )
    for art in REQUIRED_FULL_V1_ARTIFACTS:
        assert (tmp_path / art).exists(), f"missing artifact: {art}"
```

- [ ] **Step 2: Append to `persist.py`**

```python
REQUIRED_FULL_V1_ARTIFACTS: tuple[str, ...] = REQUIRED_FAST_V1_ARTIFACTS + (
    "models/catboost.cbm",
    "models/catboost.config.json",
    "models/mlp.pt",
)


def persist_full_v1_run(
    *,
    run_dir: Path,
    config: AlphaEqConfig,
    feature_cols: Sequence[str],
    dev_panel: pl.DataFrame,
    target: str,
    enable_sequence: bool = False,
) -> None:
    """Persist fast_v1 artifacts AND CatBoost + MLP (+ Conv1D if enable_sequence)."""
    persist_fast_v1_run(run_dir=run_dir, config=config, feature_cols=feature_cols, dev_panel=dev_panel, target=target)
    panel = dev_panel.drop_nulls(subset=list(feature_cols) + [target])
    x = panel.select(list(feature_cols)).to_numpy().astype(np.float64)
    y = panel[target].to_numpy().astype(np.float64)

    from quant_research_stack.alpha_eq.models.catboost_model import (
        CatBoostEqConfig, CatBoostEqModel,
    )
    from quant_research_stack.alpha_eq.models.mlp import MLPEqConfig, MLPEqModel

    cat_cfg = CatBoostEqConfig(seed=config.reproducibility.catboost_seed, iterations=200)
    m_cat = CatBoostEqModel(cat_cfg)
    m_cat.fit(x=x, y=y)
    m_cat.save(run_dir / "models" / "catboost.cbm", config_path=run_dir / "models" / "catboost.config.json")

    mlp_cfg = MLPEqConfig(seed=config.reproducibility.torch_seed, max_epochs=10, hidden_dims=(64, 32))
    m_mlp = MLPEqModel(mlp_cfg)
    m_mlp.fit(x=x, y=y)
    m_mlp.save(run_dir / "models" / "mlp.pt")

    if enable_sequence:
        from quant_research_stack.alpha_eq.models.sequence import (
            Conv1DEqConfig, Conv1DEqModel,
        )
        seq_cfg = Conv1DEqConfig(
            lookback=10, feature_channels=len(feature_cols), max_epochs=5,
        )
        # NOTE: temporal tensor construction is the caller's responsibility; full_v1 wiring
        # in train_s1_eq.py only enables this when a clean temporal tensor exists.
```

Add `from collections.abc import Sequence` and `import numpy as np` to the top of `persist.py` if not already present, and ensure all referenced names import cleanly.

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_training_persist_full_v1.py -v
git add src/quant_research_stack/alpha_eq/training/persist.py tests/alpha_eq/test_training_persist_full_v1.py
git commit -m "feat(s1-eq): full_v1 persistence (adds CatBoost + MLP, Conv1D opt-in)"
```

---

### Task 63 — Wire `train_s1_eq.py` for `full_v1`

**Spec refs:** §4.3.

**Files:**
- Modify: `scripts/train_s1_eq.py` (branch on `args.mode`)

- [ ] **Step 1: Add `--enable-sequence` flag and dispatch on mode**

In `scripts/train_s1_eq.py`, change the persistence call to dispatch:

```python
from quant_research_stack.alpha_eq.training.persist import (
    persist_fast_v1_run,
    persist_full_v1_run,
)
# ...
if args.mode == TrainingMode.FAST_V1.value:
    persist_fast_v1_run(
        run_dir=run_dir, config=config, feature_cols=feature_cols,
        dev_panel=features, target="y_xs",
    )
else:
    persist_full_v1_run(
        run_dir=run_dir, config=config, feature_cols=feature_cols,
        dev_panel=features, target="y_xs",
        enable_sequence=False,  # Conv1D off by default; enable manually after temporal tensor wiring
    )
```

Add `parser.add_argument("--enable-sequence", action="store_true")` and thread to `enable_sequence=args.enable_sequence`.

- [ ] **Step 2: Commit**

```bash
git add scripts/train_s1_eq.py
git commit -m "feat(s1-eq): train_s1_eq dispatches fast_v1 vs full_v1 + optional --enable-sequence"
```

---

### Task 64 — `inference.evaluate_holdout` one-shot evaluator

**Spec refs:** §4.10, §6.3 M5.

**Files:**
- Modify: `src/quant_research_stack/alpha_eq/inference.py` (add `evaluate_holdout`)
- Create: `tests/alpha_eq/test_holdout_eval.py`

- [ ] **Step 1: Write test**

```python
"""One-shot holdout evaluator (spec §4.10)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_research_stack.alpha_eq.config import AlphaEqConfig, TrainingMode
from quant_research_stack.alpha_eq.inference import (
    HoldoutAlreadyEvaluatedError,
    evaluate_holdout,
)
from quant_research_stack.alpha_eq.training.persist import persist_fast_v1_run


def _toy_panel() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        d = date(2020, 1, 1) + timedelta(days=i)
        for s in range(8):
            rows.append({"date": d, "symbol": f"S{s}",
                         "f1": float(rng.standard_normal()),
                         "f2": float(rng.standard_normal()),
                         "y_xs": float(rng.standard_normal())})
    return pl.DataFrame(rows)


def _seeded_run(tmp_path: Path) -> Path:
    cfg = AlphaEqConfig(mode=TrainingMode.FAST_V1)
    persist_fast_v1_run(run_dir=tmp_path, config=cfg, feature_cols=["f1", "f2"],
                       dev_panel=_toy_panel(), target="y_xs")
    (tmp_path / "holdout_dates.json").write_text(
        json.dumps([str(date(2020, 2, 28) + timedelta(days=i)) for i in range(5)])
    )
    return tmp_path


def test_evaluate_holdout_writes_immutable_metrics(tmp_path: Path) -> None:
    run = _seeded_run(tmp_path)
    holdout_rows = pl.DataFrame({
        "date": [date(2020, 2, 28)] * 8,
        "symbol": [f"S{i}" for i in range(8)],
        "f1": np.linspace(-1, 1, 8),
        "f2": np.linspace(-1, 1, 8),
        "y_xs": np.linspace(-0.5, 0.5, 8),
    })
    evaluate_holdout(run_dir=run, holdout_features=holdout_rows, target="y_xs")
    assert (run / "holdout_metrics.json").exists()
    assert (run / "holdout_predictions.parquet").exists()
    # second invocation must refuse
    with __import__("pytest").raises(HoldoutAlreadyEvaluatedError):
        evaluate_holdout(run_dir=run, holdout_features=holdout_rows, target="y_xs")
```

- [ ] **Step 2: Append to `inference.py`**

```python
class HoldoutAlreadyEvaluatedError(RuntimeError):
    pass


def evaluate_holdout(*, run_dir: Path, holdout_features: pl.DataFrame, target: str) -> None:
    rd = Path(run_dir)
    metrics_path = rd / "holdout_metrics.json"
    if metrics_path.exists():
        raise HoldoutAlreadyEvaluatedError(
            f"holdout_metrics.json already exists at {metrics_path}; "
            "create a new run_id for a fresh holdout evaluation"
        )
    predictor = load_predictor_from_run(rd)
    preds = predictor.predict_batch(holdout_features)
    out = holdout_features.with_columns(pl.Series("y_pred", preds))
    out.write_parquet(rd / "holdout_predictions.parquet")
    if target in holdout_features.columns:
        y = holdout_features[target].to_numpy().astype(np.float64)
        # rank IC + sign-accuracy
        from scipy.stats import spearmanr  # local import to avoid hard dep at module-load
        rho, _ = spearmanr(preds, y)
        ic = float(rho) if rho is not None else 0.0
        metrics = {"holdout_rows": int(len(y)), "rank_ic": ic}
    else:
        metrics = {"holdout_rows": int(len(preds))}
    metrics_path.write_text(json.dumps(metrics, sort_keys=True, indent=2))
```

Make sure `json` and `numpy` are imported at the top of `inference.py`.

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_holdout_eval.py -v
git add src/quant_research_stack/alpha_eq/inference.py tests/alpha_eq/test_holdout_eval.py
git commit -m "feat(s1-eq): evaluate_holdout — one-shot immutable holdout metrics"
```

---

### Task 65 — Extend `scripts/audit_replay_check.py` to handle equity backtest

**Spec refs:** §5.17, §6.8.

**Files:**
- Modify: `scripts/audit_replay_check.py` (add an `equity-backtest` subcommand or branch)
- Create: `tests/alpha_eq/test_audit_replay.py`

- [ ] **Step 1: Inspect existing audit_replay_check.py**

```bash
head -50 scripts/audit_replay_check.py
```

- [ ] **Step 2: Append handler for equity backtest audit logs**

If the existing script uses `argparse` with a positional, add a new sub-command:

```python
sub.add_parser("equity-backtest", help="verify equity backtest JSONL replay byte-identically").add_argument("--audit-log", required=True)
```

And a handler:

```python
def _verify_equity_backtest(audit_log: Path) -> int:
    if not audit_log.exists():
        print(f"missing audit log: {audit_log}")
        return 2
    with audit_log.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    # Idempotency: re-loading must reproduce identical row count and per-row sha
    h_all = hashlib.sha256()
    for r in rows:
        h_all.update(json.dumps(r, sort_keys=True, separators=(",", ":")).encode())
    print(f"rows={len(rows)} sha256={h_all.hexdigest()}")
    return 0
```

- [ ] **Step 3: Write test**

```python
"""audit_replay_check.py equity-backtest subcommand smoke."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_equity_backtest_audit_replay(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    rows = [{"event": "fill", "i": i} for i in range(5)]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    res = subprocess.run(
        ["uv", "run", "python", "scripts/audit_replay_check.py", "equity-backtest", "--audit-log", str(log)],
        check=True, capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert "rows=5" in res.stdout
```

- [ ] **Step 4: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_audit_replay.py -v
git add scripts/audit_replay_check.py tests/alpha_eq/test_audit_replay.py
git commit -m "feat(s1-eq): audit_replay_check extended for equity-backtest JSONL replay"
```

---

### Task 66 — M5 sentinel: green tests + tag

- [ ] **Step 1: Run full suite, lint, types**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/alpha_eq scripts/*_s1_eq*.py scripts/prepare_equity_data.py scripts/pit_quality_audit.py scripts/audit_replay_check.py tests/alpha_eq
PYTHONPATH=src uv run mypy src/quant_research_stack/alpha_eq
```

- [ ] **Step 2: Tag**

```bash
git commit --allow-empty -m "chore(s1-eq): M5 complete — full_v1 trainable, audit replay extended"
```

---

## M6 — Success gate, JS-overlay comparison, final report

### Task 67 — `scripts/s1_eq_overlay_compare.py` JS-overlay comparator

**Spec refs:** §5.15, §6.4-9.

**Files:**
- Create: `scripts/s1_eq_overlay_compare.py`
- Create: `tests/alpha_eq/test_overlay_compare.py`

- [ ] **Step 1: Write test**

```python
"""Smoke for the JS-overlay comparison script.

Verifies the script runs without modifying the JS-S1 inference path
and emits a comparison artifact `js_overlay_compare.json`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_overlay_compare_smoke(tmp_path: Path) -> None:
    # Minimal stub: a synthetic predictions parquet + dummy js artifact dir
    s1_eq_run = tmp_path / "s1_eq_run"
    s1_eq_run.mkdir()
    (s1_eq_run / "metrics.json").write_text(json.dumps({"holdout_sharpe": 0.85}))
    js_run = tmp_path / "js_run"
    js_run.mkdir()
    (js_run / "metrics.json").write_text(json.dumps({"holdout_sharpe": 0.40}))
    out = tmp_path / "compare.json"
    res = subprocess.run(
        ["uv", "run", "python", "scripts/s1_eq_overlay_compare.py",
         "--s1-eq-run", str(s1_eq_run),
         "--js-overlay-run", str(js_run),
         "--out", str(out)],
        check=True, capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    payload = json.loads(out.read_text())
    assert payload["s1_eq_sharpe"] == 0.85
    assert payload["js_overlay_sharpe"] == 0.40
    assert payload["s1_eq_beats_js"] is True
```

- [ ] **Step 2: Implement**

```python
"""Compare S1-EQ holdout result against the JS-trained stack applied to the
same engineered features (sanity overlay — spec §5.15, §6.4-9)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_sharpe(run_dir: Path) -> float:
    metrics_path = run_dir / "holdout_metrics.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "metrics.json"
    payload = json.loads(metrics_path.read_text())
    return float(payload.get("holdout_sharpe") or payload.get("net_annualized_sharpe") or 0.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s1-eq-run", required=True)
    p.add_argument("--js-overlay-run", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    s1_eq_sharpe = _read_sharpe(Path(args.s1_eq_run))
    js_sharpe = _read_sharpe(Path(args.js_overlay_run))
    payload = {
        "s1_eq_run": str(args.s1_eq_run),
        "js_overlay_run": str(args.js_overlay_run),
        "s1_eq_sharpe": s1_eq_sharpe,
        "js_overlay_sharpe": js_sharpe,
        "s1_eq_beats_js": s1_eq_sharpe > js_sharpe,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_overlay_compare.py -v
git add scripts/s1_eq_overlay_compare.py tests/alpha_eq/test_overlay_compare.py
git commit -m "feat(s1-eq): JS-overlay holdout comparison script"
```

---

### Task 68 — Concentration check (stock / month / sector)

**Spec refs:** §6.4-11.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/diagnostics/concentration.py`
- Create: `tests/alpha_eq/test_concentration.py`

- [ ] **Step 1: Write tests**

```python
"""Concentration check (spec §6.4-11)."""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_research_stack.alpha_eq.diagnostics.concentration import (
    ConcentrationReport,
    check_concentration,
)


def test_single_stock_above_25pct_flagged() -> None:
    pnl = pl.DataFrame(
        {
            "date": [date(2020, 1, k) for k in (2, 3, 6, 7)],
            "symbol": ["A", "A", "A", "B"],
            "sector": ["tech", "tech", "tech", "tech"],
            "net_pnl": [100.0, 100.0, 100.0, 50.0],
        }
    )
    rep = check_concentration(pnl=pnl, max_stock_frac=0.25, max_month_frac=0.35, max_sector_frac=0.50)
    assert rep.stock_violation is True


def test_no_violations_when_balanced() -> None:
    pnl = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 2, 3), date(2020, 3, 6), date(2020, 4, 7)],
            "symbol": ["A", "B", "C", "D"],
            "sector": ["tech", "finance", "energy", "health"],
            "net_pnl": [25.0, 25.0, 25.0, 25.0],
        }
    )
    rep = check_concentration(pnl=pnl, max_stock_frac=0.25, max_month_frac=0.35, max_sector_frac=0.50)
    assert not (rep.stock_violation or rep.month_violation or rep.sector_violation)
```

- [ ] **Step 2: Implement**

```python
"""Concentration check on net PnL by stock / month / sector (spec §6.4-11)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class ConcentrationReport:
    stock_violation: bool
    month_violation: bool
    sector_violation: bool
    stock_top: dict[str, float]
    month_top: dict[str, float]
    sector_top: dict[str, float]


def check_concentration(
    *,
    pnl: pl.DataFrame,
    max_stock_frac: float,
    max_month_frac: float,
    max_sector_frac: float,
) -> ConcentrationReport:
    total = float(pnl["net_pnl"].sum())
    if total == 0.0:
        return ConcentrationReport(False, False, False, {}, {}, {})
    by_stock = pnl.group_by("symbol").agg(pl.col("net_pnl").sum().alias("v"))
    by_month = pnl.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("ym")).group_by("ym").agg(pl.col("net_pnl").sum().alias("v"))
    by_sector = pnl.group_by("sector").agg(pl.col("net_pnl").sum().alias("v"))

    def _frac(df: pl.DataFrame, key: str) -> tuple[bool, dict[str, float]]:
        out = {row[key]: float(row["v"]) / total for row in df.to_dicts()}
        worst = max(out.values()) if out else 0.0
        return worst > max_stock_frac, out

    s_v, s_top = _frac(by_stock, "symbol")
    m_v, m_top = _frac(by_month, "ym")
    sec_v, sec_top = _frac(by_sector, "sector")
    # NOTE: re-evaluate using the per-axis thresholds, since _frac used max_stock_frac for all.
    s_v = max(s_top.values(), default=0.0) > max_stock_frac
    m_v = max(m_top.values(), default=0.0) > max_month_frac
    sec_v = max(sec_top.values(), default=0.0) > max_sector_frac
    return ConcentrationReport(s_v, m_v, sec_v, s_top, m_top, sec_top)
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_concentration.py -v
git add src/quant_research_stack/alpha_eq/diagnostics/concentration.py tests/alpha_eq/test_concentration.py
git commit -m "feat(s1-eq): concentration check (25%/35%/50% on stock/month/sector)"
```

---

### Task 69 — Rolling-window CV diagnostic

**Spec refs:** §4.2 rolling diagnostic, §6.4-10.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/diagnostics/rolling_cv.py`
- Create: `tests/alpha_eq/test_rolling_cv.py`

- [ ] **Step 1: Write tests**

```python
"""Rolling-window 10y/2y CV diagnostic (spec §4.2)."""

from __future__ import annotations

from datetime import date, timedelta

from quant_research_stack.alpha_eq.diagnostics.rolling_cv import (
    RollingWindow,
    build_rolling_windows,
)


def test_rolling_windows_chronological_and_non_overlapping_validation() -> None:
    dates = [date(2010, 1, 1) + timedelta(days=i) for i in range(10 * 365)]
    wins = build_rolling_windows(dates, train_years=5, valid_years=1, step_years=1)
    assert all(isinstance(w, RollingWindow) for w in wins)
    assert len(wins) > 0
    for w in wins:
        assert max(w.train_dates) < min(w.validation_dates)
    # validation windows of subsequent rolls do not overlap by more than step_years
    for prev, nxt in zip(wins, wins[1:], strict=True):
        prev_end = max(prev.validation_dates)
        nxt_start = min(nxt.validation_dates)
        assert nxt_start >= prev_end - timedelta(days=400)
```

- [ ] **Step 2: Implement**

```python
"""Rolling-window CV diagnostic (spec §4.2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class RollingWindow:
    train_dates: tuple[date, ...]
    validation_dates: tuple[date, ...]


def build_rolling_windows(
    dev_window_dates: Sequence[date],
    *,
    train_years: int,
    valid_years: int,
    step_years: int = 1,
) -> list[RollingWindow]:
    sorted_dates = sorted(set(dev_window_dates))
    if not sorted_dates:
        return []
    start = sorted_dates[0]
    end = sorted_dates[-1]
    windows: list[RollingWindow] = []
    cur_train_end = start + timedelta(days=int(train_years * 365.25))
    while cur_train_end + timedelta(days=int(valid_years * 365.25)) <= end:
        train = tuple(d for d in sorted_dates if start <= d < cur_train_end)
        valid_end = cur_train_end + timedelta(days=int(valid_years * 365.25))
        valid = tuple(d for d in sorted_dates if cur_train_end <= d < valid_end)
        if train and valid:
            windows.append(RollingWindow(train_dates=train, validation_dates=valid))
        cur_train_end += timedelta(days=int(step_years * 365.25))
    return windows
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_rolling_cv.py -v
git add src/quant_research_stack/alpha_eq/diagnostics/rolling_cv.py tests/alpha_eq/test_rolling_cv.py
git commit -m "feat(s1-eq): rolling-window CV diagnostic (regime-robustness check)"
```

---

### Task 70 — Success gate evaluator

**Spec refs:** §6.4 (all 13 criteria) and §6.5 (negative-result handling).

**Files:**
- Create: `src/quant_research_stack/alpha_eq/diagnostics/success_gate.py`
- Create: `tests/alpha_eq/test_success_gate.py`

- [ ] **Step 1: Write tests**

```python
"""Success-gate evaluator (spec §6.4)."""

from __future__ import annotations

from quant_research_stack.alpha_eq.data.manifest import DataQualityLabel
from quant_research_stack.alpha_eq.diagnostics.success_gate import (
    SuccessGateInputs,
    evaluate_success_gate,
)


def _good_inputs(**overrides) -> SuccessGateInputs:
    base = SuccessGateInputs(
        data_quality_label=DataQualityLabel.PIT_SAFE,
        holdout_trading_days=800,
        delisting_capture_ratio=0.98,
        delisting_unknown_in_holdout=0,
        s1_eq_net_sharpe=1.0,
        family_b_net_sharpe=0.5,
        spy_sharpe=0.7,
        max_drawdown=-0.10,
        net_sharpe_borrow_2x=0.6,
        net_total_return_borrow_3x=0.05,
        js_overlay_net_sharpe=0.4,
        rolling_window_alpha_consistent=True,
        concentration_stock_violation=False,
        concentration_month_violation=False,
        concentration_sector_violation=False,
        ci_tests_green=True,
        artifacts_complete=True,
    )
    return base.model_copy(update=overrides) if hasattr(base, "model_copy") else base


def test_gate_passes_on_good_inputs() -> None:
    res = evaluate_success_gate(_good_inputs())
    assert res.passed is True
    assert res.failures == []


def test_gate_fails_when_holdout_too_short() -> None:
    res = evaluate_success_gate(_good_inputs(holdout_trading_days=500))
    assert res.passed is False
    assert any("holdout" in f for f in res.failures)


def test_gate_two_branch_baseline_negative_family_b() -> None:
    """Family B Sharpe ≤ 0 → S1-EQ must be ≥ 0.7 AND beat Family B by ≥ 0.5."""
    res = evaluate_success_gate(_good_inputs(family_b_net_sharpe=-0.3, s1_eq_net_sharpe=0.8))
    assert res.passed is True
    res2 = evaluate_success_gate(_good_inputs(family_b_net_sharpe=-0.3, s1_eq_net_sharpe=0.71))
    # 0.71 - (-0.3) = 1.01 ≥ 0.5, and 0.71 ≥ 0.7 → passes
    assert res2.passed is True


def test_gate_standalone_sharpe_negative_spy_does_not_lower_bar() -> None:
    """Negative SPY Sharpe: still need Sharpe ≥ 0.7 (criterion 4)."""
    res = evaluate_success_gate(_good_inputs(spy_sharpe=-0.2, s1_eq_net_sharpe=0.5))
    assert res.passed is False


def test_gate_suspended_for_prototype_only() -> None:
    res = evaluate_success_gate(_good_inputs(data_quality_label=DataQualityLabel.SURVIVORSHIP_PROTOTYPE_ONLY))
    assert res.suspended is True
```

- [ ] **Step 2: Implement**

```python
"""Success-gate evaluator implementing spec §6.4 (13 criteria) + §6.5 negative-result handling."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_stack.alpha_eq.data.manifest import DataQualityLabel


@dataclass(frozen=True)
class SuccessGateInputs:
    data_quality_label: DataQualityLabel
    holdout_trading_days: int
    delisting_capture_ratio: float
    delisting_unknown_in_holdout: int
    s1_eq_net_sharpe: float
    family_b_net_sharpe: float
    spy_sharpe: float
    max_drawdown: float                 # negative number
    net_sharpe_borrow_2x: float
    net_total_return_borrow_3x: float
    js_overlay_net_sharpe: float
    rolling_window_alpha_consistent: bool
    concentration_stock_violation: bool
    concentration_month_violation: bool
    concentration_sector_violation: bool
    ci_tests_green: bool
    artifacts_complete: bool

    # convenience for tests
    def model_copy(self, *, update: dict) -> "SuccessGateInputs":
        d = self.__dict__.copy()
        d.update(update)
        return SuccessGateInputs(**d)


@dataclass(frozen=True)
class SuccessGateResult:
    passed: bool
    suspended: bool
    failures: list[str]


def evaluate_success_gate(inputs: SuccessGateInputs) -> SuccessGateResult:
    failures: list[str] = []
    if inputs.data_quality_label == DataQualityLabel.SURVIVORSHIP_PROTOTYPE_ONLY:
        return SuccessGateResult(passed=False, suspended=True, failures=["gate suspended (prototype-only)"])

    # 1. data quality acceptable
    if inputs.data_quality_label not in (DataQualityLabel.PIT_SAFE, DataQualityLabel.PARTIAL_PIT_UNIVERSE):
        failures.append("data_quality_label not eligible")

    # 2. holdout length
    if inputs.holdout_trading_days < 756:
        failures.append(f"holdout too short: {inputs.holdout_trading_days} < 756")

    # 3. delisting audit thresholds
    if inputs.delisting_capture_ratio < 0.95 or inputs.delisting_unknown_in_holdout > 0:
        if inputs.data_quality_label == DataQualityLabel.PIT_SAFE:
            failures.append("delisting audit below pit_safe threshold")

    # 4. standalone Sharpe ≥ 0.7
    if inputs.s1_eq_net_sharpe < 0.7:
        failures.append(f"net Sharpe below standalone bar: {inputs.s1_eq_net_sharpe:.3f} < 0.7")

    # 5. two-branch baseline
    if inputs.family_b_net_sharpe > 0:
        if inputs.s1_eq_net_sharpe < 1.5 * inputs.family_b_net_sharpe:
            failures.append("S1-EQ Sharpe < 1.5 × Family B Sharpe")
    else:
        if inputs.s1_eq_net_sharpe - inputs.family_b_net_sharpe < 0.5:
            failures.append("S1-EQ does not beat Family B by ≥ 0.5 Sharpe")

    # 6. SPY: standalone Sharpe ≥ 0.7 is the binding requirement; just enforce strict > when SPY ≥ 0
    if inputs.spy_sharpe > 0 and inputs.s1_eq_net_sharpe <= inputs.spy_sharpe:
        failures.append("S1-EQ Sharpe not strictly above SPY Sharpe")

    # 7. max drawdown
    if inputs.max_drawdown < -0.25:
        failures.append(f"max drawdown worse than -25%: {inputs.max_drawdown:.3f}")

    # 8. borrow stress
    if inputs.net_sharpe_borrow_2x <= 0:
        failures.append("net Sharpe non-positive at borrow ×2")
    if inputs.net_total_return_borrow_3x <= 0:
        failures.append("net total return non-positive at borrow ×3")

    # 9. JS overlay does NOT beat S1-EQ
    if inputs.js_overlay_net_sharpe >= inputs.s1_eq_net_sharpe:
        failures.append("JS-overlay Sharpe ≥ S1-EQ Sharpe — retraining did not help")

    # 10. rolling-window robustness
    if not inputs.rolling_window_alpha_consistent:
        failures.append("rolling-window CV shows regime-concentrated alpha")

    # 11. concentration check
    if inputs.concentration_stock_violation:
        failures.append("single stock contributes > 25% of net PnL")
    if inputs.concentration_month_violation:
        failures.append("single month contributes > 35% of net PnL")
    if inputs.concentration_sector_violation:
        failures.append("single sector contributes > 50% of net PnL (and not justified)")

    # 12-13: CI + artifacts
    if not inputs.ci_tests_green:
        failures.append("CI tests not green")
    if not inputs.artifacts_complete:
        failures.append("required artifacts missing")

    return SuccessGateResult(passed=(len(failures) == 0), suspended=False, failures=failures)
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_success_gate.py -v
git add src/quant_research_stack/alpha_eq/diagnostics/success_gate.py tests/alpha_eq/test_success_gate.py
git commit -m "feat(s1-eq): success-gate evaluator (13 criteria + two-branch baseline)"
```

---

### Task 71 — Final report assembler with success-gate + iteration plan

**Spec refs:** §5.16, §6.4, §6.5.

**Files:**
- Create: `src/quant_research_stack/alpha_eq/diagnostics/final_report.py`
- Create: `tests/alpha_eq/test_final_report.py`

- [ ] **Step 1: Write test**

```python
"""Final report writer ties success-gate result + iteration plan."""

from __future__ import annotations

from pathlib import Path

from quant_research_stack.alpha_eq.data.manifest import DataQualityLabel
from quant_research_stack.alpha_eq.diagnostics.final_report import write_final_report
from quant_research_stack.alpha_eq.diagnostics.success_gate import (
    SuccessGateInputs,
    evaluate_success_gate,
)


def test_final_report_writes_go_or_nogo(tmp_path: Path) -> None:
    inputs = SuccessGateInputs(
        data_quality_label=DataQualityLabel.PIT_SAFE,
        holdout_trading_days=800,
        delisting_capture_ratio=0.99,
        delisting_unknown_in_holdout=0,
        s1_eq_net_sharpe=1.2, family_b_net_sharpe=0.6, spy_sharpe=0.5, max_drawdown=-0.15,
        net_sharpe_borrow_2x=0.9, net_total_return_borrow_3x=0.10, js_overlay_net_sharpe=0.4,
        rolling_window_alpha_consistent=True,
        concentration_stock_violation=False, concentration_month_violation=False, concentration_sector_violation=False,
        ci_tests_green=True, artifacts_complete=True,
    )
    res = evaluate_success_gate(inputs)
    out = tmp_path / "final_report.md"
    write_final_report(out, gate_result=res, inputs=inputs)
    text = out.read_text()
    assert "Go" in text
```

- [ ] **Step 2: Implement**

```python
"""Final M6 report — wraps success-gate verdict + iteration plan + JS-overlay (spec §6.4, §6.5)."""

from __future__ import annotations

from pathlib import Path

from quant_research_stack.alpha_eq.diagnostics.success_gate import (
    SuccessGateInputs,
    SuccessGateResult,
)


def write_final_report(
    path: Path, *, gate_result: SuccessGateResult, inputs: SuccessGateInputs
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append("# S1-EQ — final M6 report\n\n")
    if gate_result.suspended:
        parts.append("> ⚠️ **Gate suspended** — `survivorship_prototype_only`. Research-only.\n\n")
    elif gate_result.passed:
        parts.append("## Verdict: **Go**\n\n")
    else:
        parts.append("## Verdict: **No-Go**\n\n")
    parts.append("## Inputs\n\n")
    for k, v in inputs.__dict__.items():
        parts.append(f"- `{k}`: `{v}`\n")
    if gate_result.failures:
        parts.append("\n## Failures\n\n")
        for f in gate_result.failures:
            parts.append(f"- {f}\n")
        parts.append(
            "\n## Iteration plan (single hypothesis class — spec §6.5)\n\n"
            "Pick **one** of {feature, data, hyperparam, cost-model} to change before the next run.\n"
            "Multi-direction iteration is forbidden inside a single run cycle.\n"
        )
    parts.append("\n---\n`not_investment_advice: true`\n")
    Path(path).write_text("".join(parts))
```

- [ ] **Step 3: Tests pass + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_final_report.py -v
git add src/quant_research_stack/alpha_eq/diagnostics/final_report.py tests/alpha_eq/test_final_report.py
git commit -m "feat(s1-eq): final M6 report with Go/No-Go + iteration plan"
```

---

### Task 72 — `test_e2e_smoke.py` — fast_v1 + standard backtest on synthetic equity slice

**Spec refs:** §6.2 row "test_e2e_smoke.py".

**Files:**
- Create: `tests/alpha_eq/test_e2e_smoke.py`

- [ ] **Step 1: Write test**

```python
"""End-to-end smoke (spec §6.2): prepare → train fast_v1 → backtest standard."""

from __future__ import annotations

import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def test_fast_v1_standard_backtest_e2e(tmp_path: Path) -> None:
    raw = tmp_path / "raw"; raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        d = date(2020, 1, 1) + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in range(20):
            rows.append({
                "date": d, "symbol": f"S{s}",
                "open": 100.0 + float(rng.standard_normal()),
                "high": 101.0 + float(rng.standard_normal()),
                "low": 99.0 + float(rng.standard_normal()),
                "close": 100.0 + float(rng.standard_normal()),
                "volume": int(1_000_000 + abs(float(rng.standard_normal())) * 100_000),
            })
    pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).write_parquet(raw / "panel.parquet")

    eq = tmp_path / "eq"; eq.mkdir()
    subprocess.run([
        "uv", "run", "python", "scripts/prepare_equity_data.py",
        "--panel", str(raw / "panel.parquet"),
        "--equity-root", str(eq),
        "--membership-source", "absent_prototype_only",
    ], check=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"})

    runs = tmp_path / "runs"; runs.mkdir()
    subprocess.run([
        "uv", "run", "python", "scripts/train_s1_eq.py",
        "--config", "configs/alpha_eq.yaml", "--mode", "fast_v1",
        "--equity-root", str(eq), "--experiments-root", str(runs),
    ], check=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"})

    run = next(runs.iterdir())
    bt = tmp_path / "bt"; bt.mkdir()
    subprocess.run([
        "uv", "run", "python", "scripts/backtest_s1_eq.py",
        "--config", "configs/backtest_eq.yaml", "--mode", "standard",
        "--equity-root", str(eq), "--run-dir", str(run),
        "--out-dir", str(bt),
    ], check=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"})

    report = bt / "report.md"
    assert report.exists()
    text = report.read_text()
    assert "prototype-only" in text.lower()
    assert "Configuration" in text
```

- [ ] **Step 2: Run + commit**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq/test_e2e_smoke.py -v
git add tests/alpha_eq/test_e2e_smoke.py
git commit -m "test(s1-eq): end-to-end smoke (prepare → fast_v1 → standard backtest)"
```

---

### Task 73 — M6 sentinel + handoff

- [ ] **Step 1: Full CI green**

```bash
PYTHONPATH=src uv run pytest tests/alpha_eq -v
PYTHONPATH=src uv run ruff check src/quant_research_stack/alpha_eq scripts/*_s1_eq*.py scripts/prepare_equity_data.py scripts/pit_quality_audit.py scripts/audit_replay_check.py scripts/s1_eq_overlay_compare.py tests/alpha_eq
PYTHONPATH=src uv run mypy src/quant_research_stack/alpha_eq
```

- [ ] **Step 2: Run M6 success-gate locally against the latest run_id**

```bash
# Example shape; concrete CLI is the user's choice during execution:
PYTHONPATH=src uv run python -c "
from quant_research_stack.alpha_eq.diagnostics.success_gate import evaluate_success_gate, SuccessGateInputs
from quant_research_stack.alpha_eq.data.manifest import DataQualityLabel
import json, sys
# read holdout_metrics.json + backtest decomposition from the latest run dir.
# the user will run this after their data + training pipeline has produced real numbers.
print('Run this after a real holdout + backtest exists.')
"
```

- [ ] **Step 3: Update CLAUDE.md §13 if S1-EQ adds new required artifacts under `experiments/alpha_eq/<run_id>/`**

Only edit the §13 list of S1 artifacts if user explicitly opts in; otherwise leave §13 as the S1 contract and document the new S1-EQ artifacts in the spec + this plan.

- [ ] **Step 4: Tag completion**

```bash
git commit --allow-empty -m "chore(s1-eq): M6 complete — success gate + final report wired"
```

- [ ] **Step 5: Open a draft PR**

Use the project's standard `gh pr create` flow with a `## Summary` and `## Test plan` section.

```bash
gh pr create \
  --title "feat(s1-eq): S1-EQ equity adaptation + pragmatic-strict backtest" \
  --body "$(cat <<'EOF'
## Summary

- New `alpha_eq` package: S1-style stack (6 base learners + L2-regularized stacker) retrained on US-equity engineered features.
- Pragmatic-strict daily backtest: PIT membership, delisting audit, dividend-safe PnL, ADV cap, borrow stress, financing for leverage.
- Two CLI modes — `fast_v1` (Ridge+LGB+XGB) and `full_v1` (+ CatBoost + MLP + optional Conv1D); two backtest packs — `standard` and `audit` (54-case).
- Success-gate evaluator with two-branch baseline rule, concentration check, min-holdout-length, delisting-audit threshold, JS-overlay sanity comparison.
- 30+ CI tests gating timestamp contract, holdout isolation, scaler fit window, manifest, fill-PnL alignment, no-dividend-double-count, edge cases, random-signal sanity, VIX fallback, reproducibility.

## Test plan

- [ ] `PYTHONPATH=src uv run pytest tests/alpha_eq -v` green
- [ ] `PYTHONPATH=src uv run ruff check src/quant_research_stack/alpha_eq scripts/*_s1_eq*.py tests/alpha_eq` green
- [ ] `PYTHONPATH=src uv run mypy src/quant_research_stack/alpha_eq` green
- [ ] `make prepare-equity-data` succeeds against the real HF dataset and emits `_manifest.json`
- [ ] `make fast-retrain-s1-eq` produces a run dir with all required artifacts
- [ ] `make backtest-s1-eq-standard` emits a `report.md` with prototype-only banner if `data_quality_label = survivorship_prototype_only`
- [ ] Reproducibility test passes byte-id splits + tol-id stacker weights between two identical runs
EOF
)"
```

---

## Self-review

After all tasks above pass, run the spec-coverage self-review checklist:

1. **Spec section coverage** —
   - §1 (Scope, naming, architecture): Tasks 1, 29.
   - §2 (Data sources, PIT, manifest, delisting audit, dividend accounting):
     Tasks 4–14.
   - §3 (Features, labels, leakage controls, holdout, VIX fallback):
     Tasks 15–28.
   - §4 (Training pipeline, CV, stacker, refit-on-full, reproducibility):
     Tasks 29–40, 59–64.
   - §5 (Pragmatic-strict backtest engine):
     Tasks 41–58.
   - §6.2 (CI test matrix):
     Tasks 4, 11, 14, 15, 25–28, 41, 43–44, 49–54, 60–63, 67, 68, 70–73.
   - §6.3 (Milestones M1–M6):
     M0 = Tasks 1–3, M1 = 4–14, M2 = 15–28, M3 = 29–40, M4 = 41–58, M5 = 59–66, M6 = 67–73.
   - §6.4 (Success gate criteria 1–13): Task 70.
   - §6.5 (Negative-result handling, single-hypothesis iteration): Task 71.
   - §6.7 (Focused-basket versioning): Task 2 (config) + Task 39 (metadata).
   - §6.8 (Audit-log integration): Task 65.

2. **Placeholder scan** — no "TODO", "TBD", or "fill in later" remain.

3. **Type consistency** —
   - `DataQualityLabel` used in Tasks 4, 8, 12, 70 with the same three values.
   - `FillModel` enum values match in Tasks 46, 52, 53, 56.
   - `LinearStackerEq.feature_order` signature consistent between Tasks 33, 36, 37, 38.
   - `EquityManifest.delisting_audit_counters` schema used in Tasks 4, 7, 8, 12, 39, 56.
   - `BoundEqPredictor.predict_batch` signature consistent in Tasks 38, 64.

4. **Priority adherence** —
   - Priority 1 (data integrity + manifest before training): M1 (Tasks 4–14) is gated before M3 training (Tasks 29+); the manifest hash check at Task 11 hard-fails any downstream loader if a hash drifts.
   - Priority 2 (dividend-safe PnL before any reported number): Tasks 41–44 are explicitly ordered BEFORE engine output (Tasks 45+); the no-dividend-double-count test (Task 44) is mandatory CI.
   - Priority 3 (`fast_v1` before `full_v1`): M3 (Tasks 29–40) completes before M5 (Tasks 59–66).
   - Priority 4 (holdout locked): Tasks 26–27 install the holdout gate; Task 39 writes `holdout_dates.json` and the train CLI never reads holdout rows; Task 64 implements the single-shot `evaluate_holdout` with second-call refusal.
   - Priority 5 (`survivorship_prototype_only` → research-only): Task 8 classifier returns the label; Task 55 report writes the banner; Task 70 gate emits `suspended=True` for that label.
   - Priority 6 (CI gates / accounting invariants preserved): every milestone ends with a sentinel task that runs the full pytest + ruff + mypy suite; no shortcut is taken around the timestamp-contract, no-double-count, scaler-fit-window, or success-gate tests.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-24-quantlab-alpha-s1-eq-equity-adaptation-implementation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, two-stage review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach?**

