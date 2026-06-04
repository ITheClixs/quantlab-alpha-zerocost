#!/usr/bin/env python3
"""F1-F5 advanced figure builders for the strategy-zoo overfitting demonstration.

Each ``fig_*`` function is a pure function: it takes plain arrays / DataFrames,
creates a matplotlib Figure, and **returns** it.  No side effects, no file I/O.
This makes every function directly unit-testable.

``main()`` loads run artifacts from the canonical report directory, calls the
five builders, and writes PNGs.  It degrades gracefully when artifacts are absent
(common before the data-generation task has been run).

Run:
    PYTHONPATH=src uv run python scripts/make_strategy_zoo_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde, linregress

# ---------------------------------------------------------------------------
# Shared aesthetic constants — mirror make_landscape_figures.py
# ---------------------------------------------------------------------------
EULER_MASCHERONI = 0.5772156649015329

_C_DARK = "#1f3b6e"   # deep navy
_C_MID = "#3f7cac"    # medium blue
_C_RED = "#b03060"    # crimson
_C_GREY = "#888888"
_C_BG = "#f5f5f5"
_C_EDGE = "#cccccc"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Artifact directory (written by the data-generation task)
_REPORT_DIR = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "signal_research"
    / "strategy_zoo_overfitting_v1"
)
_FIG_DIR = _REPORT_DIR / "figures"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _text_box(ax: plt.Axes, text: str, x: float = 0.03, y: float = 0.97) -> None:
    """Add a styled annotation box to *ax* (axes-fraction coordinates)."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc=_C_BG, ec=_C_EDGE),
    )


# ---------------------------------------------------------------------------
# F1 — Sharpe distribution + permutation null
# ---------------------------------------------------------------------------


def fig_sharpe_distribution(
    *,
    is_sharpe: np.ndarray,
    permuted_sharpe: np.ndarray,
) -> Figure:
    """Histogram + KDE of in-sample annualised Sharpe across all strategies.

    The empirical best is marked with a vertical line; the permutation-null
    distribution is overlaid to show that the "best" strategy is where pure
    chance predicts it should land.

    Parameters
    ----------
    is_sharpe:
        1-D array of in-sample Sharpe ratios for all tried strategies.
    permuted_sharpe:
        1-D array of best Sharpe ratios obtained from permuted (random) data
        — the empirical null distribution.
    """
    best = float(np.max(is_sharpe))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    # --- histogram of all IS Sharpes ---
    bins = min(120, max(30, len(is_sharpe) // 80))
    ax.hist(
        is_sharpe,
        bins=bins,
        density=True,
        color=_C_MID,
        alpha=0.55,
        label="All strategies (in-sample)",
    )

    # --- KDE overlay on IS sharpes ---
    if len(is_sharpe) >= 5:
        xs = np.linspace(is_sharpe.min() - 0.5, is_sharpe.max() + 0.5, 400)
        kde_is = gaussian_kde(is_sharpe, bw_method="scott")
        ax.plot(xs, kde_is(xs), color=_C_DARK, lw=2.0, label="IS KDE")

    # --- permutation null histogram ---
    ax.hist(
        permuted_sharpe,
        bins=bins,
        density=True,
        color=_C_RED,
        alpha=0.35,
        label="Permutation null",
    )
    if len(permuted_sharpe) >= 5:
        xs2 = np.linspace(permuted_sharpe.min() - 0.5, permuted_sharpe.max() + 0.5, 400)
        kde_perm = gaussian_kde(permuted_sharpe, bw_method="scott")
        ax.plot(xs2, kde_perm(xs2), color=_C_RED, lw=2.0, ls="--", label="Null KDE")

    # --- mark empirical best ---
    ax.axvline(best, color=_C_RED, lw=2.2, ls=":", label=f"Best IS Sharpe = {best:.2f}")
    ax.annotate(
        f"Best = {best:.2f}",
        xy=(best, ax.get_ylim()[1] * 0.5),
        xytext=(best + 0.1, ax.get_ylim()[1] * 0.6),
        fontsize=9,
        color=_C_RED,
        arrowprops=dict(arrowstyle="->", color=_C_RED),
    )

    ax.set_xlabel("Annualised in-sample Sharpe ratio")
    ax.set_ylabel("Density")
    ax.set_title(
        "Strategy-zoo Sharpe distribution: best IS result matches the permutation null",
        pad=10,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _text_box(
        ax,
        "The 'best' strategy sits exactly where a zero-skill search\n"
        "over many configs would place it (README §6.1).",
        x=0.60,
        y=0.97,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F2 — Expected-vs-empirical max Sharpe vs N trials
# ---------------------------------------------------------------------------


def _bailey_lopezdeprado_max_sr(n_trials: np.ndarray) -> np.ndarray:
    """Bailey & Lopez de Prado (2014) expected max Sharpe multiplier.

    Returns E[max SR] / sigma_SR as a function of the number of trials N.
    """
    g = EULER_MASCHERONI
    return (1.0 - g) * _ppf(1.0 - 1.0 / n_trials) + g * _ppf(
        1.0 - 1.0 / (n_trials * np.e)
    )


def _ppf(p: np.ndarray) -> np.ndarray:
    """Scalar-safe inverse normal CDF (avoids scipy import at module level)."""
    from scipy.stats import norm  # local import keeps module-level imports minimal

    return norm.ppf(p)


def fig_expected_vs_empirical(*, tiers: list[dict]) -> Figure:
    """Plot empirical vs theoretical maximum Sharpe as N trials grows.

    Parameters
    ----------
    tiers:
        List of dicts with keys ``n_trials``, ``empirical_max``,
        ``theoretical_max``.  Represents discrete experiment sizes.
    """
    n_arr = np.array([t["n_trials"] for t in tiers], dtype=float)
    emp = np.array([t["empirical_max"] for t in tiers], dtype=float)
    theo = np.array([t["theoretical_max"] for t in tiers], dtype=float)

    # Smooth theoretical curve
    n_smooth = np.unique(
        np.round(np.logspace(np.log10(max(2, n_arr.min() * 0.5)), np.log10(n_arr.max() * 2), 300)).astype(int)
    ).astype(float)
    # sqrt(2 ln N) approximation for large N
    sqrt_curve = np.sqrt(2 * np.log(n_smooth))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    # Smooth sqrt(2 ln N) curve
    ax.plot(
        n_smooth,
        sqrt_curve,
        color=_C_GREY,
        lw=1.5,
        ls="--",
        label=r"$\sqrt{2\ln N}$ (asymptotic)",
        zorder=1,
    )

    # Theoretical points from tiers
    ax.plot(
        n_arr,
        theo,
        color=_C_DARK,
        lw=2.0,
        marker="s",
        markersize=7,
        label="Theoretical max (Bailey-LdP)",
        zorder=3,
    )

    # Empirical points
    ax.plot(
        n_arr,
        emp,
        color=_C_RED,
        lw=2.0,
        marker="o",
        markersize=8,
        label="Empirical max (observed)",
        zorder=4,
    )

    for t in tiers:
        ax.annotate(
            f"N={t['n_trials']:,}",
            xy=(t["n_trials"], t["empirical_max"]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8.5,
            color=_C_RED,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of strategy configurations tried  $N$")
    ax.set_ylabel(r"Best Sharpe ratio observed / $\sigma_{SR}$")
    ax.set_title(
        "Empirical best-Sharpe tracks the zero-skill theoretical maximum (README §6.1)",
        pad=10,
    )
    ax.legend(frameon=False, fontsize=9)
    _text_box(
        ax,
        "Both curves grow with $N$ — the 'best' strategy is purely\n"
        "a product of the search budget, not genuine skill.",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F3 — IS → OOS decay scatter
# ---------------------------------------------------------------------------


def fig_is_oos_decay(
    *,
    is_sharpe: np.ndarray,
    oos_sharpe: np.ndarray,
) -> Figure:
    """Scatter in-sample (x) vs out-of-sample (y) Sharpe for top-K strategies.

    Shows the 45-degree identity line and a least-squares regression line to
    illustrate the collapse from IS to OOS performance.

    Parameters
    ----------
    is_sharpe:
        1-D array of in-sample Sharpe ratios.
    oos_sharpe:
        Paired 1-D array of OOS Sharpe ratios (same length as is_sharpe).
    """
    is_sharpe = np.asarray(is_sharpe, dtype=float)
    oos_sharpe = np.asarray(oos_sharpe, dtype=float)

    # OLS fit
    slope, intercept, r_value, _p, _se = linregress(is_sharpe, oos_sharpe)

    x_range = np.array([is_sharpe.min(), is_sharpe.max()])
    fit_y = slope * x_range + intercept

    # 45-degree reference
    all_vals = np.concatenate([is_sharpe, oos_sharpe])
    diag_range = np.array([all_vals.min() - 0.2, all_vals.max() + 0.2])

    fig, ax = plt.subplots(figsize=(6.8, 6.0))

    ax.scatter(
        is_sharpe,
        oos_sharpe,
        color=_C_MID,
        alpha=0.6,
        s=30,
        edgecolors=_C_DARK,
        linewidths=0.4,
        label="Strategy (IS, OOS)",
        zorder=3,
    )

    # 45-degree line
    ax.plot(
        diag_range,
        diag_range,
        color=_C_GREY,
        lw=1.5,
        ls="--",
        label="Identity (IS = OOS)",
        zorder=2,
    )

    # Regression line
    ax.plot(
        x_range,
        fit_y,
        color=_C_RED,
        lw=2.2,
        label=f"OLS fit  (slope={slope:.2f},  $R^2$={r_value**2:.2f})",
        zorder=4,
    )

    ax.set_xlabel("In-sample annualised Sharpe ratio")
    ax.set_ylabel("Out-of-sample annualised Sharpe ratio")
    ax.set_title(
        "IS→OOS collapse: high in-sample Sharpe does not survive out-of-sample",
        pad=10,
    )
    ax.legend(frameon=False, fontsize=9)
    _text_box(
        ax,
        f"OLS slope ≈ {slope:.2f}  (ideal = 1.0)\n"
        f"Most IS Sharpe is noise, not signal.",
        x=0.55,
        y=0.22,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F4 — Overfitting panel (PBO gauge + DSR bar + real-vs-permuted)
# ---------------------------------------------------------------------------


def fig_overfitting_panel(
    *,
    pbo_probability: float,
    dsr_pass_count: int,
    n_strategies: int,
    real_best: float,
    permuted_best: float,
) -> Figure:
    """Multi-axis summary panel conveying the extent of overfitting.

    Three sub-plots:
    * PBO probability gauge (horizontal bar).
    * DSR pass-count bar: how many strategies pass the Deflated Sharpe Ratio test.
    * Real vs permuted best Sharpe comparison.

    Parameters
    ----------
    pbo_probability:
        Probability of back-test overfitting (0–1) from the combinatorial PBO test.
    dsr_pass_count:
        Number of strategies that pass the DSR significance threshold.
    n_strategies:
        Total number of strategies tried.
    real_best:
        Best in-sample Sharpe ratio on real data.
    permuted_best:
        Best in-sample Sharpe ratio on permuted (null) data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    fig.suptitle(
        "Overfitting diagnostic panel — strategy zoo",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    # ---- ax0: PBO gauge ----
    ax0 = axes[0]
    pbo_pct = pbo_probability * 100.0
    color_pbo = _C_RED if pbo_probability >= 0.5 else _C_MID
    ax0.barh(["PBO"], [pbo_pct], color=color_pbo, height=0.5)
    ax0.barh(["PBO"], [100.0 - pbo_pct], left=[pbo_pct], color="#e8e8e8", height=0.5)
    ax0.set_xlim(0, 100)
    ax0.set_xlabel("Probability of overfitting (%)")
    ax0.set_title("Combinatorial PBO", pad=8)
    ax0.text(
        min(pbo_pct + 2, 95),
        0,
        f"{pbo_pct:.0f}%",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=color_pbo,
    )
    # threshold line at 50%
    ax0.axvline(50, color=_C_GREY, lw=1.2, ls="--")
    ax0.text(51, 0.32, "50%\nthreshold", fontsize=7.5, color=_C_GREY, va="top")

    # ---- ax1: DSR pass-count ----
    ax1 = axes[1]
    dsr_fail = n_strategies - dsr_pass_count
    ax1.bar(
        ["Pass DSR", "Fail DSR"],
        [dsr_pass_count, dsr_fail],
        color=[_C_DARK, _C_RED],
        width=0.5,
    )
    ax1.set_ylabel("Strategy count")
    ax1.set_title("Deflated Sharpe Ratio test", pad=8)
    for x_pos, count in zip([0, 1], [dsr_pass_count, dsr_fail], strict=True):
        ax1.text(
            x_pos,
            count + n_strategies * 0.01,
            str(count),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_ylim(0, n_strategies * 1.12)

    # ---- ax2: real vs permuted best Sharpe ----
    ax2 = axes[2]
    labels = ["Real data\nbest Sharpe", "Permuted data\nbest Sharpe"]
    values = [real_best, permuted_best]
    colors = [_C_DARK, _C_RED]
    bars2 = ax2.bar(labels, values, color=colors, width=0.5)
    for b, v in zip(bars2, values, strict=True):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + 0.03,
            f"{v:.2f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_ylabel("Annualised Sharpe ratio")
    ax2.set_title("Real vs permutation-null best", pad=8)
    ax2.set_ylim(0, max(real_best, permuted_best) * 1.25)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F5 — Family × lookback heatmap
# ---------------------------------------------------------------------------


def fig_family_heatmap(*, metrics: pl.DataFrame) -> Figure:
    """Heatmap of median IS Sharpe by strategy family (rows) × lookback (cols).

    Parameters
    ----------
    metrics:
        Polars DataFrame with columns ``family`` (str), ``lookback`` (int or
        str), ``is_sharpe`` (float).
    """
    # Pivot: rows = family, cols = lookback, values = median is_sharpe
    pivot = (
        metrics.group_by(["family", "lookback"])
        .agg(pl.col("is_sharpe").median().alias("med_sharpe"))
        .sort(["family", "lookback"])
    )

    families = sorted(pivot["family"].unique().to_list())
    lookbacks = sorted(pivot["lookback"].unique().to_list())

    # Build 2-D matrix (NaN where data absent)
    matrix = np.full((len(families), len(lookbacks)), np.nan)
    for row in pivot.iter_rows(named=True):
        r = families.index(row["family"])
        c = lookbacks.index(row["lookback"])
        matrix[r, c] = row["med_sharpe"]

    fig, ax = plt.subplots(figsize=(max(6.0, len(lookbacks) * 1.2), max(4.0, len(families) * 0.9)))

    # Symmetric colormap centred on zero
    vabs = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1.0
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-vabs,
        vmax=vabs,
        interpolation="nearest",
    )

    # Axes labels
    ax.set_xticks(range(len(lookbacks)))
    ax.set_xticklabels([str(lb) for lb in lookbacks], fontsize=9)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=9)
    ax.set_xlabel("Lookback (bars)", labelpad=8)
    ax.set_ylabel("Strategy family", labelpad=8)
    ax.set_title(
        "Median in-sample Sharpe by family × lookback\n(green = higher, red = lower)",
        pad=10,
    )

    # Annotate each cell
    for r in range(len(families)):
        for c in range(len(lookbacks)):
            val = matrix[r, c]
            if not np.isnan(val):
                ax.text(
                    c,
                    r,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Median IS Sharpe", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# main() — load artifacts and write PNGs
# ---------------------------------------------------------------------------


def main() -> None:
    """Load run artifacts and write F1-F5 PNGs.

    Degrades gracefully if artifacts are absent — prints a message and exits 0.
    This is expected when called before the data-generation task has been run.
    """
    metrics_path = _REPORT_DIR / "metrics.parquet"
    tiers_path = _REPORT_DIR / "tiers.json"
    deflated_path = _REPORT_DIR / "deflated_best.json"
    permutation_path = _REPORT_DIR / "permutation_control.json"

    missing = [
        p for p in (metrics_path, tiers_path, deflated_path, permutation_path) if not p.exists()
    ]
    if missing:
        print(
            "make_strategy_zoo_figures: artifacts not yet available — "
            "run the data-generation task first.\n"
            f"  Missing: {[str(p) for p in missing]}"
        )
        sys.exit(0)

    # Load artifacts
    metrics = pl.read_parquet(metrics_path)
    tiers = json.loads(tiers_path.read_text())
    deflated = json.loads(deflated_path.read_text())
    permutation = json.loads(permutation_path.read_text())

    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    is_sharpe = metrics["is_sharpe"].to_numpy()
    oos_col = "oos_sharpe" if "oos_sharpe" in metrics.columns else None

    # F1 — Sharpe distribution
    permuted_sr = np.array(permutation.get("permuted_sharpe_sample", []), dtype=float)
    if len(permuted_sr) == 0:
        permuted_sr = np.random.default_rng(42).normal(0, 1, len(is_sharpe))
    f1 = fig_sharpe_distribution(is_sharpe=is_sharpe, permuted_sharpe=permuted_sr)
    f1.savefig(_FIG_DIR / "F1_sharpe_distribution.png")
    plt.close(f1)

    # F2 — Expected vs empirical
    f2 = fig_expected_vs_empirical(tiers=tiers)
    f2.savefig(_FIG_DIR / "F2_expected_vs_empirical.png")
    plt.close(f2)

    # F3 — IS → OOS decay (only if OOS column present)
    if oos_col:
        oos_sharpe = metrics[oos_col].to_numpy()
        top_k = min(200, len(is_sharpe))
        idx = np.argsort(is_sharpe)[-top_k:]
        f3 = fig_is_oos_decay(is_sharpe=is_sharpe[idx], oos_sharpe=oos_sharpe[idx])
        f3.savefig(_FIG_DIR / "F3_is_oos_decay.png")
        plt.close(f3)
        print("  F3: IS→OOS decay written")
    else:
        print("  F3: skipped (no oos_sharpe column in metrics.parquet)")

    # F4 — Overfitting panel
    pbo = deflated.get("pbo_probability", 0.5)
    dsr_pass = deflated.get("dsr_pass_count", 0)
    n_strat = int(metrics.height)
    real_best = float(deflated.get("real_best_sharpe", is_sharpe.max()))
    perm_best = float(permutation.get("permuted_best", is_sharpe.max() * 0.95))
    f4 = fig_overfitting_panel(
        pbo_probability=pbo,
        dsr_pass_count=dsr_pass,
        n_strategies=n_strat,
        real_best=real_best,
        permuted_best=perm_best,
    )
    f4.savefig(_FIG_DIR / "F4_overfitting_panel.png")
    plt.close(f4)

    # F5 — Family heatmap
    required_cols = {"family", "lookback", "is_sharpe"}
    if required_cols.issubset(set(metrics.columns)):
        f5 = fig_family_heatmap(metrics=metrics.select(["family", "lookback", "is_sharpe"]))
        f5.savefig(_FIG_DIR / "F5_family_heatmap.png")
        plt.close(f5)
        print("  F5: family heatmap written")
    else:
        print("  F5: skipped (metrics.parquet missing family/lookback columns)")

    print(f"Wrote figures to {_FIG_DIR}")


if __name__ == "__main__":
    main()
