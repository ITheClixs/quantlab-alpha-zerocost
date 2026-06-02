#!/usr/bin/env python3
"""Generate the figures for the README "Related Work — Public-Alpha Landscape"
section and the competitive-landscape report.

Every figure here is *derived from a closed-form model* (no market data), so it is
fully reproducible and deterministic. The figures formalise why publicly available
"successful" quant systems do not survive the QuantLab validation gate, and how the
external literature corroborates the program's four-wall thesis.

Run:
    PYTHONPATH=src uv run python scripts/make_landscape_figures.py

Outputs (figures/):
    landscape_max_sharpe_vs_trials.png   multiple-testing inflation of the Sharpe bar
    landscape_publication_decay.png      McLean-Pontiff post-publication alpha decay
    landscape_crowding_decay.png         Grossman-Stiglitz / Red-Queen alpha -> 0
    landscape_min_detectable_ic.png      minimum-detectable IC vs sample size
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
EULER_MASCHERONI = 0.5772156649015329

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


def expected_max_sharpe_multiplier(n_trials: np.ndarray) -> np.ndarray:
    """Bailey & Lopez de Prado (2014) expected maximum of N i.i.d. zero-skill
    Sharpe estimates, in units of the cross-trial std of the estimates.

        E[max SR] / sigma_SR ~= (1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e))

    with g the Euler-Mascheroni constant and Z = Phi^{-1} the inverse normal CDF.
    """
    g = EULER_MASCHERONI
    return (1.0 - g) * norm.ppf(1.0 - 1.0 / n_trials) + g * norm.ppf(
        1.0 - 1.0 / (n_trials * np.e)
    )


def fig_max_sharpe_vs_trials() -> None:
    n = np.unique(np.round(np.logspace(np.log10(2), 4, 200)).astype(int))
    mult = expected_max_sharpe_multiplier(n)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(n, mult, color="#b03060", lw=2.2)
    ax.set_xscale("log")
    ax.set_xlabel("Number of strategy configurations tried  $N$")
    ax.set_ylabel(r"Expected best Sharpe under the null  $\mathbb{E}[\max]/\sigma_{SR}$")
    ax.set_title("Multiple testing inflates the Sharpe bar even with zero true skill")
    for nn in (10, 100, 1000):
        m = expected_max_sharpe_multiplier(np.array([nn]))[0]
        ax.scatter([nn], [m], color="#1f3b6e", zorder=5, s=28)
        ax.annotate(
            f"N={nn}: {m:.2f}$\\sigma$",
            (nn, m),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=9,
        )
    ax.text(
        0.02,
        0.96,
        "A zero-skill search over many configs produces an apparently\n"
        "high best-Sharpe purely by selection. The Deflated Sharpe Ratio\n"
        r"benchmark $SR_0$ is set to this curve (README $\S3.7$).",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#cccccc"),
    )
    fig.savefig(FIG_DIR / "landscape_max_sharpe_vs_trials.png")
    plt.close(fig)


def fig_publication_decay() -> None:
    # McLean & Pontiff (2016): ~26% lower out-of-sample (statistical bias),
    # ~58% lower post-publication (additional ~32% from investor learning).
    stages = ["In-sample\n(published)", "Out-of-sample\n(pre-pub)", "Post-publication\n(live)"]
    values = [1.00, 0.74, 0.42]
    colors = ["#1f3b6e", "#3f7cac", "#b03060"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(stages, values, color=colors, width=0.6)
    ax.set_ylabel("Mean anomaly return (fraction of in-sample)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Published anomalies decay: the easy-to-arbitrage edge is competed away")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.annotate(
        "",
        xy=(1, 0.74),
        xytext=(0, 1.00),
        arrowprops=dict(arrowstyle="->", color="#555555"),
    )
    ax.text(0.5, 0.90, "-26%\nstatistical bias\n(overfitting)", ha="center", fontsize=8.5, color="#555555")
    ax.annotate(
        "",
        xy=(2, 0.42),
        xytext=(1, 0.74),
        arrowprops=dict(arrowstyle="->", color="#555555"),
    )
    ax.text(1.5, 0.60, "-32%\ninvestor learning\n(crowding)", ha="center", fontsize=8.5, color="#555555")
    fig.savefig(FIG_DIR / "landscape_publication_decay.png")
    plt.close(fig)


def fig_crowding_decay() -> None:
    phi = np.linspace(0.0, 1.0, 200)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for k, c, lbl in [(1, "#3f7cac", "k=1 (linear)"), (2, "#1f3b6e", "k=2"), (4, "#b03060", "k=4 (fast)")]:
        ax.plot(phi, (1.0 - phi) ** k, color=c, lw=2.0, label=lbl)
    ax.set_xlabel(r"Fraction of capital running the same signal  $\varphi$")
    ax.set_ylabel(r"Net alpha  $\alpha_{\mathrm{net}}(\varphi)/\alpha_0$")
    ax.set_title("Red-Queen / Grossman-Stiglitz: shared signals decay to zero alpha")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.30,
        0.55,
        r"$\alpha_{\mathrm{net}}(\varphi)=\alpha_0\,(1-\varphi)^k,\quad"
        r"\lim_{\varphi\to 1}\alpha_{\mathrm{net}}=0$",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#cccccc"),
    )
    fig.savefig(FIG_DIR / "landscape_crowding_decay.png")
    plt.close(fig)


def fig_min_detectable_ic() -> None:
    # SE(IC) ~= 1/sqrt(N); two-sided 5% significance needs |t| = IC*sqrt(N) >= 1.96.
    n = np.logspace(1, 7, 300)
    ic95 = 1.96 / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(n, ic95, color="#1f3b6e", lw=2.2, label=r"min detectable IC ($p<0.05$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Independent cross-sectional observations  $N$")
    ax.set_ylabel(r"Minimum detectable information coefficient $|\mathrm{IC}|$")
    ax.set_ylim(top=2.0)
    ax.set_title(
        "Low-frequency free data cannot resolve the small IC that real alpha carries",
        pad=12,
    )
    ax.axhline(0.03, color="#b03060", ls="--", lw=1.5, label="typical real IC ~ 0.03 (e.g. Qlib benchmarks)")
    # Annotate regimes
    ax.axvspan(10, 500, color="#b03060", alpha=0.08)
    ax.text(70, 0.22, "EDGAR 10-K\nannual\n(sample wall)", fontsize=8.5, color="#b03060", ha="center")
    ax.axvspan(1e5, 1e7, color="#3f7cac", alpha=0.08)
    ax.text(1.2e6, 0.22, "tick / daily panel\n(detectable, but\nthen cost wall)", fontsize=8.5, color="#1f3b6e", ha="center")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.savefig(FIG_DIR / "landscape_min_detectable_ic.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig_max_sharpe_vs_trials()
    fig_publication_decay()
    fig_crowding_decay()
    fig_min_detectable_ic()
    print(f"wrote 4 landscape figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
