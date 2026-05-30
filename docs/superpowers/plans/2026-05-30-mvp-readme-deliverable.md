# QuantLab Alpha MVP Deliverable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finalize QuantLab Alpha as a team-leader-facing MVP: a research-paper-format `README.md` (formulas/derivations, honest results, in-repo hyperlinks) plus a `scripts/make_mvp_figures.py` + `make mvp` quickstart that regenerates the funding-carry capstone figures from cached free data.

**Architecture:** Two small, focused Python scripts (figure generation + README link checker), each with pure testable helpers and a thin I/O `main()`. A `Makefile` `mvp` target chains the existing funding-carry runners with the figures script. The README is rewritten as an honest negative-results research paper sourced from committed manifests/metrics (never hand-typed performance numbers). No model retraining, no new research, no PR.

**Tech Stack:** Python 3.11, `uv`, `polars`, `numpy`, `matplotlib` (already a dep, 3.10.9), `pytest`, `ruff`, `mypy`. Reuses `quant_research_stack.crypto_research.funding` (`data`, `prices`, `carry`).

**Honesty guardrail (program rule, applies to every task):** real numbers only — S1 holdout weighted zero-mean R² = 0.0055 (< 0.012 release gate), 0 deployable/paper/live strategies, funding-carry = DO_NOT_ADVANCE. No fabricated performance, no gate-weakening, no promotion language. research_only.

---

## File Structure

- Create: `scripts/make_mvp_figures.py` — pure data-prep helpers + matplotlib plotting + `main()` that writes 4 PNGs to `figures/`.
- Create: `tests/test_make_mvp_figures.py` — unit tests for helpers + a plotting smoke test.
- Create: `scripts/check_readme_links.py` — extract relative markdown links + report missing targets.
- Create: `tests/test_check_readme_links.py` — unit tests for the link extractor/checker.
- Create: `figures/*.png` — 4 committed figures (generated, deterministic).
- Modify: `Makefile` — add the `mvp` target (and a `mvp-figures` helper).
- Modify: `README.md` — full rewrite as the research paper.
- Reference (read-only): `experiments/alpha_s1/20260523-160541/metrics.json`, `manifests/funding_carry/funding_carry_realism_manifest.json`, the funding module.

---

## Task 1: Figure data-prep helpers (pure, testable)

**Files:**
- Create: `scripts/make_mvp_figures.py`
- Test: `tests/test_make_mvp_figures.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_make_mvp_figures.py
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_mvp_figures", Path(__file__).resolve().parents[1] / "scripts" / "make_mvp_figures.py")
mmf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mmf)  # type: ignore[union-attr]


def test_s1_fold_model_r2_groups_by_model() -> None:
    metrics = {"fold_metrics": [
        {"fold": 0.0, "ridge_r2": 0.1, "lgb_r2": 0.2, "xgb_r2": 0.0,
         "cat_r2": 0.0, "mlp_r2": 0.0, "seq_r2": 0.0},
        {"fold": 1.0, "ridge_r2": 0.3, "lgb_r2": 0.4, "xgb_r2": 0.0,
         "cat_r2": 0.0, "mlp_r2": 0.0, "seq_r2": 0.0},
    ]}
    out = mmf.s1_fold_model_r2(metrics)
    assert out["ridge"] == [0.1, 0.3]
    assert out["lgb"] == [0.2, 0.4]
    assert set(out) == {"ridge", "lgb", "xgb", "cat", "mlp", "seq"}


def test_leverage_stress_rows_converts_return_to_pct() -> None:
    manifest = {"liquidation_stressed_pooled": {
        "3x": {"sharpe": -0.47, "ann_return": -0.17, "n_liquidations": 10.0},
        "10x": {"sharpe": -5.17, "ann_return": -0.896, "n_liquidations": 284.0},
    }}
    rows = mmf.leverage_stress_rows(manifest)
    assert rows[0] == ("3x", -0.47, -17.0)
    assert rows[1][0] == "10x"
    assert abs(rows[1][2] - (-89.6)) < 1e-6


def test_per_year_bar_rows_from_manifest() -> None:
    manifest = {"honest_pooled_per_year": {
        "2024": {"total_pct": 13.3}, "2026": {"total_pct": -0.16}}}
    rows = mmf.per_year_bar_rows(manifest)
    assert rows == [(2024, 13.3), (2026, -0.16)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_make_mvp_figures.py -q`
Expected: FAIL (module `make_mvp_figures` has no attribute `s1_fold_model_r2`).

- [ ] **Step 3: Write the helpers**

```python
# scripts/make_mvp_figures.py
"""Generate the QuantLab Alpha MVP research-paper figures from committed artifacts.

Deterministic + honesty-safe: every figure is sourced from a committed manifest/metrics
file or recomputed from cached free data — no hand-typed performance numbers, no
training, no network (if caches are present). Missing artifacts are skipped with a
warning; the script never fabricates a figure.

Run: PYTHONPATH=src uv run python scripts/make_mvp_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

S1_METRICS = Path("experiments/alpha_s1/20260523-160541/metrics.json")
REALISM_MANIFEST = Path("manifests/funding_carry/funding_carry_realism_manifest.json")
FIG_DIR = Path("figures")
S1_MODELS = ["ridge", "lgb", "xgb", "cat", "mlp", "seq"]
FUNDING_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def s1_fold_model_r2(metrics: dict) -> dict[str, list[float]]:
    """{model: [r2 per fold]} from an S1 metrics.json `fold_metrics` list."""
    out: dict[str, list[float]] = {m: [] for m in S1_MODELS}
    for fm in metrics["fold_metrics"]:
        for m in S1_MODELS:
            out[m].append(float(fm[f"{m}_r2"]))
    return out


def leverage_stress_rows(manifest: dict) -> list[tuple[str, float, float]]:
    """(leverage_label, sharpe, ann_return_pct) per leverage from the realism manifest."""
    rows: list[tuple[str, float, float]] = []
    for k, v in manifest["liquidation_stressed_pooled"].items():
        rows.append((k, float(v["sharpe"]), float(v["ann_return"]) * 100.0))
    return rows


def per_year_bar_rows(manifest: dict) -> list[tuple[int, float]]:
    """(year, net_pct) from the realism manifest `honest_pooled_per_year`."""
    return [(int(y), float(d["total_pct"]))
            for y, d in manifest["honest_pooled_per_year"].items()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_make_mvp_figures.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/make_mvp_figures.py tests/test_make_mvp_figures.py
git commit -m "feat(mvp): figure data-prep helpers (S1 R2, leverage stress, per-year)"
```

---

## Task 2: Pooled 8h carry recompute helper

**Files:**
- Modify: `scripts/make_mvp_figures.py`
- Test: `tests/test_make_mvp_figures.py`

- [ ] **Step 1: Write the failing test** (append to the test file)

```python
def test_pooled_equity_from_net_compounds() -> None:
    import numpy as np
    net = np.array([0.01, -0.005, 0.02])
    eq = mmf.pooled_equity(net)
    assert eq[0] == 1.0  # equity starts at 1.0 (pre-first-bar)
    assert abs(eq[-1] - (1.01 * 0.995 * 1.02)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_make_mvp_figures.py::test_pooled_equity_from_net_compounds -q`
Expected: FAIL (no attribute `pooled_equity`).

- [ ] **Step 3: Implement `pooled_equity` + the cached recompute**

```python
# add to scripts/make_mvp_figures.py
def pooled_equity(net) -> "list":
    import numpy as np
    eq = np.cumprod(1.0 + np.asarray(net, dtype=float))
    return np.concatenate([[1.0], eq])


def pooled_8h_carry():
    """Recompute the pooled BTC+ETH 8h-marked carry from cached free data.

    Mirrors scripts/run_funding_carry_realism.py (base costs 10/5/5 bps). Returns a
    CarryResult with `.net` (per-8h) and `.dates` for the equity + per-year figures.
    """
    from quant_research_stack.crypto_research.funding import carry, data as fdata, prices
    p8 = {s: prices.align_carry_8h(fdata.load_funding(s),
                                   prices.load_klines(s, "spot", "8h"),
                                   prices.load_klines(s, "perp", "8h"))
          for s in FUNDING_SYMBOLS}
    common = p8[FUNDING_SYMBOLS[0]].select("ts")
    for s in FUNDING_SYMBOLS[1:]:
        common = common.join(p8[s].select("ts"), on="ts", how="inner")
    p8 = {s: p8[s].join(common, on="ts", how="inner").sort("ts") for s in FUNDING_SYMBOLS}
    legs = {s: carry.carry_returns_8h(p8[s], spot_taker_bps=10.0, perp_taker_bps=5.0,
                                      slip_bps=5.0) for s in FUNDING_SYMBOLS}
    return carry.pooled_book(legs, periods_per_year=carry.PPY_8H)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_make_mvp_figures.py::test_pooled_equity_from_net_compounds -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_mvp_figures.py tests/test_make_mvp_figures.py
git commit -m "feat(mvp): pooled 8h carry recompute + equity helper"
```

---

## Task 3: Plotting functions, `main()`, and the 4 committed figures

**Files:**
- Modify: `scripts/make_mvp_figures.py`
- Test: `tests/test_make_mvp_figures.py`

- [ ] **Step 1: Write the failing smoke test** (append)

```python
def test_plot_leverage_stress_writes_png(tmp_path) -> None:
    rows = [("3x", -0.47, -17.0), ("5x", -1.5, -37.8), ("10x", -5.17, -89.6)]
    out = tmp_path / "lev.png"
    mmf.plot_leverage_stress(rows, out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_make_mvp_figures.py::test_plot_leverage_stress_writes_png -q`
Expected: FAIL (no attribute `plot_leverage_stress`).

- [ ] **Step 3: Implement plotting + main**

```python
# add to scripts/make_mvp_figures.py
def _mpl():
    import matplotlib
    matplotlib.use("Agg")  # headless, deterministic
    import matplotlib.pyplot as plt
    return plt


def plot_leverage_stress(rows: list[tuple[str, float, float]], out: Path) -> None:
    plt = _mpl()
    labels = [r[0] for r in rows]
    ann = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, ann, color=["#c0392b" if a < 0 else "#27ae60" for a in ann])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Funding carry: annual return vs perp leverage (liquidation stress)")
    ax.set_ylabel("ann return (%)")
    ax.set_xlabel("isolated-margin leverage on the short perp")
    ax.bar_label(bars, fmt="%.0f%%")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_equity(dates, net, out: Path) -> None:
    plt = _mpl()
    eq = pooled_equity(net)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(eq)), eq, color="#2c3e50", lw=1.2)
    ax.set_title("Funding carry (pooled BTC+ETH, 8h-marked, unlevered): growth of 1")
    ax.set_ylabel("equity (start = 1.0)")
    ax.set_xlabel(f"8h settlement bars ({dates[0]} .. {dates[-1]})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_per_year(rows: list[tuple[int, float]], out: Path) -> None:
    plt = _mpl()
    years = [str(r[0]) for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(years, vals, color=["#c0392b" if v < 0 else "#27ae60" for v in vals])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Funding carry: per-year net return (after cost, unlevered)")
    ax.set_ylabel("net return (%)")
    ax.bar_label(bars, fmt="%.1f")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_s1_fold_model_r2(grouped: dict[str, list[float]], out: Path) -> None:
    plt = _mpl()
    import numpy as np
    n_folds = len(next(iter(grouped.values())))
    x = np.arange(n_folds)
    width = 0.13
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (model, vals) in enumerate(grouped.items()):
        ax.bar(x + i * width, vals, width, label=model)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("S1 base-model R² by fold (holdout gate = 0.012)")
    ax.set_ylabel("R²")
    ax.set_xlabel("fold")
    ax.set_xticks(x + width * (len(grouped) - 1) / 2)
    ax.set_xticklabels([f"fold {i}" for i in range(n_folds)])
    ax.legend(ncol=6, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    # S1 figure (from committed metrics)
    if S1_METRICS.exists():
        m = json.loads(S1_METRICS.read_text())
        plot_s1_fold_model_r2(s1_fold_model_r2(m), FIG_DIR / "s1_fold_model_r2.png")
        print(f"wrote {FIG_DIR/'s1_fold_model_r2.png'}")
    else:
        print(f"WARN: {S1_METRICS} absent — skipping S1 figure", file=sys.stderr)
    # funding-carry figures
    if REALISM_MANIFEST.exists():
        man = json.loads(REALISM_MANIFEST.read_text())
        plot_leverage_stress(leverage_stress_rows(man), FIG_DIR / "funding_carry_leverage_stress.png")
        plot_per_year(per_year_bar_rows(man), FIG_DIR / "funding_carry_per_year.png")
        print(f"wrote {FIG_DIR/'funding_carry_leverage_stress.png'}, {FIG_DIR/'funding_carry_per_year.png'}")
    else:
        print(f"WARN: {REALISM_MANIFEST} absent — run scripts/run_funding_carry_realism.py first", file=sys.stderr)
    # equity curve (recompute from cached data)
    try:
        pooled = pooled_8h_carry()
        plot_equity(pooled.dates, pooled.net, FIG_DIR / "funding_carry_equity.png")
        print(f"wrote {FIG_DIR/'funding_carry_equity.png'}")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never fabricate
        print(f"WARN: equity figure skipped ({exc})", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test + full file tests**

Run: `PYTHONPATH=src uv run pytest tests/test_make_mvp_figures.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Generate the real figures (cached data) + lint/type**

Run:
```bash
PYTHONPATH=src uv run python scripts/run_funding_carry_realism.py
PYTHONPATH=src uv run python scripts/make_mvp_figures.py
PYTHONPATH=src uv run ruff check scripts/make_mvp_figures.py tests/test_make_mvp_figures.py
PYTHONPATH=src uv run mypy scripts/make_mvp_figures.py
ls -la figures/
```
Expected: 4 PNGs in `figures/`; ruff + mypy clean.

- [ ] **Step 6: Commit (script + figures)**

```bash
git add scripts/make_mvp_figures.py tests/test_make_mvp_figures.py figures/
git commit -m "feat(mvp): matplotlib figures + generate 4 committed PNGs from cached artifacts"
```

---

## Task 4: `make mvp` quickstart target

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the targets** (append to `Makefile`)

```makefile
.PHONY: mvp mvp-figures
mvp-figures:
	PYTHONPATH=src uv run python scripts/make_mvp_figures.py

## mvp: regenerate the funding-carry capstone result + figures on cached free data
mvp:
	PYTHONPATH=src uv run python scripts/run_funding_carry_v1.py
	PYTHONPATH=src uv run python scripts/run_funding_carry_realism.py
	PYTHONPATH=src uv run python scripts/make_mvp_figures.py
	@echo ""
	@echo "MVP quickstart complete. Verdict: funding-carry = DO_NOT_ADVANCE (research_only)."
	@echo "Figures: figures/*.png   Reports: reports/signal_research/funding_carry_v1/*.md"
```

- [ ] **Step 2: Run the target**

Run: `make mvp`
Expected: both runners print their summaries, 4 figures written, final "DO_NOT_ADVANCE" line.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat(mvp): make mvp quickstart — funding-carry pipeline + figures on cached data"
```

---

## Task 5: README link checker (guards the hyperlink map)

**Files:**
- Create: `scripts/check_readme_links.py`
- Test: `tests/test_check_readme_links.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_check_readme_links.py
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_readme_links", Path(__file__).resolve().parents[1] / "scripts" / "check_readme_links.py")
crl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crl)  # type: ignore[union-attr]


def test_extract_skips_http_and_anchors() -> None:
    md = "[a](https://x.com) [b](docs/y.md) [c](#section) [d](mailto:x@y.z) [e](docs/z.md#frag)"
    assert crl.extract_relative_links(md) == ["docs/y.md", "docs/z.md"]


def test_missing_links_detects_only_absent(tmp_path) -> None:
    (tmp_path / "README.md").write_text("[x](nope.md) [y](real.md)")
    (tmp_path / "real.md").write_text("hi")
    assert crl.missing_links(tmp_path / "README.md") == ["nope.md"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_check_readme_links.py -q`
Expected: FAIL (no module attribute).

- [ ] **Step 3: Implement the checker**

```python
# scripts/check_readme_links.py
"""Verify every relative markdown link in README.md resolves to a real file.

Guards the research paper's in-repo hyperlink map ("blue underlines"). Skips http(s)/
mailto and pure #anchors; strips #fragments from relative links before existence-check.

Run: PYTHONPATH=src uv run python scripts/check_readme_links.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def extract_relative_links(md: str) -> list[str]:
    out: list[str] = []
    for raw in _LINK_RE.findall(md):
        target = raw.split("#")[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        out.append(target)
    return out


def missing_links(md_path: Path) -> list[str]:
    md = md_path.read_text()
    miss = [link for link in extract_relative_links(md)
            if not (md_path.parent / link).resolve().exists()]
    return sorted(set(miss))


def main() -> None:
    miss = missing_links(Path("README.md"))
    if miss:
        print("MISSING README LINKS:")
        for x in miss:
            print(f"  {x}")
        sys.exit(1)
    print("all README relative links resolve")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_check_readme_links.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/check_readme_links.py tests/test_check_readme_links.py
git commit -m "feat(mvp): README relative-link checker + tests"
```

---

## Task 6: Write the research-paper README

**Files:**
- Modify: `README.md` (full rewrite)

This is a documentation task, not TDD. Write the 10 sections below. Use real numbers only
(values are pre-filled from committed artifacts). Embed LaTeX with `$...$`/`$$...$$`.
Every named doc/module is a relative markdown link. Keep the existing badge-link pattern.

- [ ] **Step 1: Write the full README**

Structure and required content (write the prose around these anchors):

1. **Title + badges.** Keep `QUANTLAB_STAGE=paper` and kill-switch badges linking to
   `docs/runbooks/stage_promotion.md` and `docs/runbooks/kill_switch.md`. Add an honest
   badge: `alpha: none-deployable (research-only)`.

2. **Abstract (~150 words).** QuantLab Alpha is a local-first, stage-gated alpha research
   platform (S1 tabular predictor → S2 LLM governor → S3 feeds/brokers → S4 execution).
   Its delivered contribution is a **validation harness** and the reproducible, honest
   finding that under a **zero-cost data constraint** no taker-tradable alpha survives
   out-of-sample validation; the binding constraint is the information set, not method.
   0 deployable/paper/live strategies. State research-only explicitly.

3. **§1 Introduction.** Problem (find honest, tradable alpha on free data); two-layer
   thesis (numeric S1 ⊥ evidence-based S2 governance); RQ1–RQ4 table (preserve from the
   current README, keep the evidence links to `src/quant_research_stack/alpha/` etc.).

4. **§2 Platform Architecture.** A `mermaid` flowchart:
   ```mermaid
   flowchart LR
     raw[raw free data] --> feat[features + meta-features]
     feat --> S1[S1 tabular predictor]
     S1 --> S2[S2 LLM governor\nGBNF + paper citations]
     S2 --> S4[S4 execution\nQUANTLAB_STAGE gate]
     S4 --> audit[(append-only audit log)]
   ```
   Then prose on stage-gating (`paper`/`live_shadow`/`live`, operator-only promotion),
   kill-switch precedence, append-only audit + byte-for-byte replay. Link
   `docs/architecture/adrs/0001-two-tier-tabular-llm.md`,
   `docs/architecture/adrs/0002-three-stage-promotion-gate.md`,
   `docs/architecture/adrs/0014-kill-switch-precedence.md`,
   `docs/superpowers/specs/2026-05-14-quantlab-alpha-platform-design.md`.

5. **§3 Machine-Learning Methodology.** One subsection per item, each with the formula and
   a link to the implementing module + `docs/research/VALIDATION_RUNBOOK.md`:
   - **Purged & embargoed walk-forward / CPCV** — link `src/quant_research_stack/alpha/cv.py`.
     Explain purging label-overlapping train rows + embargo buffer.
   - **Weighted zero-mean R²** — `$$R^2_{w}=1-\frac{\sum_i w_i (y_i-\hat y_i)^2}{\sum_i w_i y_i^2}$$`
     link `src/quant_research_stack/alpha/metrics.py`. Note denominator is about 0, can be negative.
   - **Adversarial validation** — AUC > 0.6 ⇒ drop; link `src/quant_research_stack/alpha/adversarial.py`.
   - **Noise-floor control** — seeded `$N(0,1)$` feature; drop features below it in ≥3/5 folds.
   - **Stacking meta-learner** — OOF matrix → linear meta-model; link `src/quant_research_stack/alpha/stacking.py`.
   - **PBO (CSCV)** — `$$\mathrm{PBO}=\Pr\!\big[\operatorname{logit}(\bar{r})\le 0\big]$$`
     link `src/quant_research_stack/crypto_research/perps/validation.py`.
   - **Deflated Sharpe Ratio** —
     `$$\widehat{\mathrm{DSR}}=\Phi\!\left(\frac{(\hat{SR}-SR_0)\sqrt{T-1}}{\sqrt{1-\gamma_3\hat{SR}+\frac{\gamma_4-1}{4}\hat{SR}^2}}\right)$$`
     with `$SR_0$` inflated for the number of trials (Bailey–López de Prado).
   - **Stationary bootstrap** Sharpe CI (Politis–Romano); gate lower bound > 0.
   - **Funding-carry identity** — `$$r_t=(r^{\text{spot}}_t-r^{\text{perp}}_t)+f_t-c_t$$`
     long spot/short perp; short receives `$f_t$` when positive; isolated-margin liquidation
     model; link `src/quant_research_stack/crypto_research/funding/carry.py`.

6. **§4 Experimental Results.**
   - **4.1 S1 tabular predictor.** Embed `figures/s1_fold_model_r2.png`. Table of the real
     holdout numbers (from `experiments/alpha_s1/20260523-160541/metrics.json`):
     holdout weighted zero-mean R² = **0.0055**; per-model holdout R²: cat +0.0062,
     ridge +0.0039, lgb +0.0025, seq −0.0007, xgb −0.0093, mlp −0.0094; 79 features;
     4,011,392 train / 1,008,656 holdout rows. Verdict: **below the 0.012 release gate**.
     Link `docs/superpowers/plans/2026-05-14-quantlab-alpha-s1-implementation.md`.
   - **4.2 Signal-research ledger.** A table of the ~13 closed branches with verdict +
     link, sourced from `docs/research/2026-05-ZERO-COST-ALPHA-SEARCH-CLOSEOUT.md` and
     `docs/research/2026-05-PROGRAM-REVIEW-SIGNAL-RESEARCH.md`. Rows + links:
     OHLCV (`docs/research/2026-05-NEGATIVE-RESULT-OHLCV-ALPHA.md`), VRP & HMM & FOMC
     (`docs/research/2026-05-NEGATIVE-RESULT-EVENT-MACRO-FOMC.md`,
     `docs/research/intake/2026-05-28-vrp-index-v1.md`,
     `docs/research/intake/2026-05-28-hmm-single-index-v1.md`), microstructure L2/L1/tick
     (`docs/research/2026-05-NEGATIVE-RESULT-MICROSTRUCTURE.md`), futures carry
     (`docs/research/intake/2026-05-30-futures-carry-term-structure-v1.md`), options-IV
     (`docs/research/intake/2026-05-30-options-iv-features-v1.md`), EDGAR 10-K/10-Q
     (`docs/research/intake/2026-05-30-edgar-10k-text-features-v1.md`), zero-cost
     allocators v1/v2 (`docs/research/intake/2026-05-30-zero-cost-deployable-v1.md`).
   - **4.3 Funding-carry capstone.** Embed `figures/funding_carry_equity.png`,
     `figures/funding_carry_per_year.png`, `figures/funding_carry_leverage_stress.png`.
     Narrate: data audit PASS → 8h-marked delta-neutral backtest → liquidation stress →
     §5 gate → **DO_NOT_ADVANCE**. Real numbers: net-positive 6/7 years; unlevered pooled
     Sharpe ~8.6 / ~14%/yr but a fat crash-liquidation tail (3x −17%/yr, 10x −90%); fails
     the pre-registered 2026 regime gate. Link
     `docs/research/2026-05-NEGATIVE-RESULT-FUNDING-CARRY.md`,
     `docs/research/intake/2026-05-30-funding-carry-v1.md`,
     `reports/signal_research/funding_carry_v1/funding_carry_realism_results.md`.

7. **§5 Discussion — the Four Walls.** cost / subsumption / data-access / frequency; the
   meta-conclusion (data acquisition is binding). Link
   `docs/research/2026-05-ZERO-COST-ALPHA-SEARCH-CLOSEOUT.md`,
   `docs/research/2026-05-30-PAID-DATA-ACQUISITION-RECOMMENDATION.md`,
   `docs/research/2026-05-30-ZERO-COST-CONSTRAINT.md`.

8. **§6 Reproducibility.** The `make mvp` one-liner + what it regenerates; environment
   (`uv`, `PYTHONPATH=src`); artifact SHA-256 (`experiments/alpha_s1/20260523-160541/_artifact_sha256.json`);
   audit replay; verification: `PYTHONPATH=src uv run pytest -q`, `ruff check`, `mypy src`.

9. **§7 Limitations & honest disclosures.** No deployable alpha; S1 below gate; funding
   tail + regime decay; free-data scope; what would change the answer (paid data — link
   `docs/research/2026-05-DATA-PURCHASE-FEASIBILITY-SHARADAR.md`).

10. **§8 References & repository map.** A bulleted hyperlink index grouped by: ADRs (0001–
    0014), Specs, Plans, Negative-result notes, Intakes, Runbooks. These provide the bulk
    of the "blue underlines."

- [ ] **Step 2: Verify links resolve**

Run: `PYTHONPATH=src uv run python scripts/check_readme_links.py`
Expected: "all README relative links resolve". If any are missing, fix the path or create
nothing — only link to files that exist.

- [ ] **Step 3: Honesty review (manual)**

Re-read every quantitative claim. Confirm each matches a committed artifact:
S1 R² 0.0055 (metrics.json), funding-carry DO_NOT_ADVANCE + per-year (realism manifest),
0 deployable (close-out). No promotion language. Fix any drift.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(mvp): rewrite README as honest research-paper deliverable (formulas, results, link map)"
```

---

## Task 7: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full gate**

Run:
```bash
PYTHONPATH=src uv run pytest tests/test_make_mvp_figures.py tests/test_check_readme_links.py tests/crypto_research/funding/ -q
PYTHONPATH=src uv run ruff check scripts/make_mvp_figures.py scripts/check_readme_links.py
PYTHONPATH=src uv run mypy scripts/make_mvp_figures.py scripts/check_readme_links.py
PYTHONPATH=src uv run python scripts/check_readme_links.py
make mvp
```
Expected: all tests pass; ruff + mypy clean; links resolve; `make mvp` ends with the
DO_NOT_ADVANCE line and 4 figures present.

- [ ] **Step 2: Confirm figures are committed and referenced**

Run: `git status --porcelain figures/ && grep -c "figures/" README.md`
Expected: figures tracked (no untracked PNGs); README references all 4.

- [ ] **Step 3: Final commit (if anything pending) — DO NOT open a PR**

```bash
git add -A && git commit -m "chore(mvp): finalize MVP deliverable — README paper + quickstart + figures" || echo "nothing to commit"
```
**Do NOT run `gh pr create` or push a PR. The operator opens the PR when they say so.**

---

## Self-Review (completed by plan author)

- **Spec coverage:** README sections (§3.1–3.10) → Task 6; formulas (§4) → Task 6 step 1;
  figures (§5) → Tasks 1–3; `make mvp` (§6) → Task 4; link map (§7) → Task 6 + Task 5
  checker; testing/acceptance (§8) → Tasks 1–3,5,7; honesty guardrail (§0) → Task 6 step 3.
  No gaps.
- **Placeholder scan:** all code steps contain complete code; the README task lists exact
  content/values/links (prose written at execution, anchors specified). No TBD/TODO.
- **Type consistency:** helper names (`s1_fold_model_r2`, `leverage_stress_rows`,
  `per_year_bar_rows`, `pooled_equity`, `pooled_8h_carry`, `plot_*`, `extract_relative_links`,
  `missing_links`) are used identically across tasks and tests. `pooled_book` called with
  `periods_per_year=carry.PPY_8H` (matches the funding module API).
