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
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from quant_research_stack.crypto_research.funding.carry import CarryResult

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


def pooled_equity(net: NDArray[np.float64]) -> NDArray[np.float64]:
    """Growth-of-1 equity curve from a per-bar net-return array (starts at 1.0)."""
    eq = np.cumprod(1.0 + np.asarray(net, dtype=float))
    return np.concatenate([[1.0], eq])


def pooled_8h_carry() -> CarryResult:
    """Recompute the pooled BTC+ETH 8h-marked carry from cached free data.

    Mirrors scripts/run_funding_carry_realism.py (base costs 10/5/5 bps). Returns a
    CarryResult with `.net` (per-8h) and `.dates` for the equity figure.
    """
    from quant_research_stack.crypto_research.funding import carry, prices
    from quant_research_stack.crypto_research.funding import data as fdata
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


def plot_equity(dates: list, net: NDArray[np.float64], out: Path) -> None:
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
    ax.bar_label(bars, fmt="%.1f%%")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_s1_fold_model_r2(grouped: dict[str, list[float]], out: Path) -> None:
    plt = _mpl()
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
    if S1_METRICS.exists():
        m = json.loads(S1_METRICS.read_text())
        plot_s1_fold_model_r2(s1_fold_model_r2(m), FIG_DIR / "s1_fold_model_r2.png")
        print(f"wrote {FIG_DIR/'s1_fold_model_r2.png'}")
    else:
        print(f"WARN: {S1_METRICS} absent — skipping S1 figure", file=sys.stderr)
    if REALISM_MANIFEST.exists():
        man = json.loads(REALISM_MANIFEST.read_text())
        plot_leverage_stress(leverage_stress_rows(man), FIG_DIR / "funding_carry_leverage_stress.png")
        plot_per_year(per_year_bar_rows(man), FIG_DIR / "funding_carry_per_year.png")
        print("wrote leverage_stress + per_year figures")
    else:
        print(f"WARN: {REALISM_MANIFEST} absent — run scripts/run_funding_carry_realism.py first", file=sys.stderr)
    try:
        pooled = pooled_8h_carry()
        plot_equity(pooled.dates, pooled.net, FIG_DIR / "funding_carry_equity.png")
        print(f"wrote {FIG_DIR/'funding_carry_equity.png'}")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never fabricate
        print(f"WARN: equity figure skipped ({exc})", file=sys.stderr)


if __name__ == "__main__":
    main()
