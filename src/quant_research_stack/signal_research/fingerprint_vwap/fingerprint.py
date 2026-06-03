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
        out_frames.append(
            group.with_columns(
                [pl.Series(k, v).fill_nan(None) for k, v in cols.items()]
            )
        )
    return pl.concat(out_frames, how="vertical")


def fingerprint_columns(windows: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"{b}_{w}" for w in windows for b in _BASES)
