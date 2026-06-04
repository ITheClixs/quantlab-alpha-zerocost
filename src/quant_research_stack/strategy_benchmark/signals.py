"""15 signal families used to enumerate ~1500 strategy variants.

Each family is a callable: `signals(bars: pl.DataFrame, *, lookback: int,
threshold: float) -> pl.Series`. The output series is the position weight
∈ [-1, +1] (full long / full short / flat). Backtest applies a 1-day shift
so today's signal trades tomorrow's open.

Family roster (peer-reviewed quant lineage in parentheses):
 1. TS_MOMENTUM       (Moskowitz Ooi Pedersen 2012 — Time Series Momentum)
 2. LAGGED_MOMENTUM   (Jegadeesh Titman 1993 — 12-1 momentum, generalised)
 3. MA_CROSSOVER      (classic dual-MA filter; Faber 2007)
 4. DONCHIAN_BREAKOUT (Turtle Trader / Donchian channel)
 5. BOLLINGER_REVERT  (Bollinger 1980s — buy lower-band touch)
 6. BOLLINGER_BREAKOUT(buy upper-band breakout)
 7. RSI_MEANREVERT    (Wilder 1978 — RSI<30 → buy)
 8. MACD              (Appel 1979 — fast/slow EMA with signal line)
 9. VOLTGT_MOMENTUM   (vol-scaled trend; Hurst Ooi Pedersen 2017)
10. ZSCORE_MEANREVERT (rolling (close-mean)/std)
11. AROON             (Chande 1995)
12. STOCHASTIC        (Lane 1950s)
13. ROC               (rate of change)
14. CCI               (Lambert 1980 — Commodity Channel Index)
15. KELTNER_BREAKOUT  (Keltner / Chester 1960s)
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl  # noqa: I001

# Helper rolling expressions ---------------------------------------------------

def _rolling_mean(col: str, window: int) -> pl.Expr:
    return pl.col(col).rolling_mean(window_size=window, min_samples=window)


def _rolling_std(col: str, window: int) -> pl.Expr:
    return pl.col(col).rolling_std(window_size=window, min_samples=window)


def _rolling_max(col: str, window: int) -> pl.Expr:
    return pl.col(col).rolling_max(window_size=window, min_samples=window)


def _rolling_min(col: str, window: int) -> pl.Expr:
    return pl.col(col).rolling_min(window_size=window, min_samples=window)


# 15 signal generators ---------------------------------------------------------

def signal_ts_momentum(bars: pl.DataFrame, *, lookback: int, threshold: float) -> pl.Series:
    """Sign of N-day log return, scaled by intensity in [0.5, 2.5]."""
    df = bars.with_columns(
        (pl.col("close").log() - pl.col("close").shift(lookback).log()).alias("_r")
    )
    df = df.with_columns(
        pl.when(pl.col("_r") > 0)
        .then(pl.lit(1.0) * threshold / 2.5)
        .when(pl.col("_r") < 0)
        .then(pl.lit(-1.0) * threshold / 2.5)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_lagged_momentum(
    bars: pl.DataFrame, *, lookback: int, threshold: float
) -> pl.Series:
    """Jegadeesh-Titman style: log-return between lookback*2 and lookback
    days ago (skip-month equivalent on daily bars)."""
    df = bars.with_columns(
        (
            pl.col("close").shift(lookback).log()
            - pl.col("close").shift(lookback * 2).log()
        ).alias("_r")
    )
    df = df.with_columns(
        pl.when(pl.col("_r") > 0)
        .then(pl.lit(1.0) * threshold / 2.5)
        .when(pl.col("_r") < 0)
        .then(pl.lit(-1.0) * threshold / 2.5)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_ma_crossover(
    bars: pl.DataFrame, *, lookback: int, threshold: float
) -> pl.Series:
    fast = max(2, lookback // 4)
    slow = lookback
    df = bars.with_columns(
        _rolling_mean("close", fast).alias("_fast"),
        _rolling_mean("close", slow).alias("_slow"),
    )
    df = df.with_columns(
        ((pl.col("_fast") - pl.col("_slow")) / pl.col("_slow")).alias("_gap")
    )
    df = df.with_columns(
        pl.when(pl.col("_gap") > threshold / 100.0)
        .then(1.0)
        .when(pl.col("_gap") < -threshold / 100.0)
        .then(-1.0)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_donchian_breakout(
    bars: pl.DataFrame, *, lookback: int, threshold: float
) -> pl.Series:
    df = bars.with_columns(
        _rolling_max("high", lookback).shift(1).alias("_hh"),
        _rolling_min("low", lookback).shift(1).alias("_ll"),
    )
    df = df.with_columns(
        pl.when(pl.col("close") > pl.col("_hh") * (1.0 + threshold / 1000.0))
        .then(1.0)
        .when(pl.col("close") < pl.col("_ll") * (1.0 - threshold / 1000.0))
        .then(-1.0)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def _bollinger_bands(bars: pl.DataFrame, lookback: int, k: float) -> pl.DataFrame:
    return bars.with_columns(
        _rolling_mean("close", lookback).alias("_mu"),
        _rolling_std("close", lookback).alias("_sd"),
    ).with_columns(
        (pl.col("_mu") + k * pl.col("_sd")).alias("_upper"),
        (pl.col("_mu") - k * pl.col("_sd")).alias("_lower"),
    )


def signal_bollinger_revert(
    bars: pl.DataFrame, *, lookback: int, threshold: float
) -> pl.Series:
    df = _bollinger_bands(bars, lookback, k=threshold)
    df = df.with_columns(
        pl.when(pl.col("close") < pl.col("_lower"))
        .then(1.0)
        .when(pl.col("close") > pl.col("_upper"))
        .then(-1.0)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_bollinger_breakout(
    bars: pl.DataFrame, *, lookback: int, threshold: float
) -> pl.Series:
    df = _bollinger_bands(bars, lookback, k=threshold)
    df = df.with_columns(
        pl.when(pl.col("close") > pl.col("_upper"))
        .then(1.0)
        .when(pl.col("close") < pl.col("_lower"))
        .then(-1.0)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_rsi_meanrevert(
    bars: pl.DataFrame, *, lookback: int, threshold: float
) -> pl.Series:
    df = bars.with_columns(
        (pl.col("close") - pl.col("close").shift(1)).alias("_d")
    )
    df = df.with_columns(
        pl.when(pl.col("_d") > 0).then(pl.col("_d")).otherwise(0.0).alias("_gain"),
        pl.when(pl.col("_d") < 0).then(-pl.col("_d")).otherwise(0.0).alias("_loss"),
    )
    df = df.with_columns(
        _rolling_mean("_gain", lookback).alias("_ag"),
        _rolling_mean("_loss", lookback).alias("_al"),
    )
    df = df.with_columns(
        (100.0 - 100.0 / (1.0 + pl.col("_ag") / pl.col("_al").clip(lower_bound=1e-9))).alias("_rsi")
    )
    # threshold is the distance from 50: e.g. threshold=20 → buy<30, sell>70
    df = df.with_columns(
        pl.when(pl.col("_rsi") < 50.0 - threshold)
        .then(1.0)
        .when(pl.col("_rsi") > 50.0 + threshold)
        .then(-1.0)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_macd(bars: pl.DataFrame, *, lookback: int, threshold: float) -> pl.Series:
    fast = max(4, lookback // 2)
    slow = lookback
    sig = max(2, lookback // 4)
    df = bars.with_columns(
        pl.col("close").ewm_mean(span=fast).alias("_ef"),
        pl.col("close").ewm_mean(span=slow).alias("_es"),
    )
    df = df.with_columns((pl.col("_ef") - pl.col("_es")).alias("_macd"))
    df = df.with_columns(pl.col("_macd").ewm_mean(span=sig).alias("_sig_line"))
    df = df.with_columns(
        pl.when(pl.col("_macd") > pl.col("_sig_line") * (1.0 + threshold / 1000.0))
        .then(1.0)
        .when(pl.col("_macd") < pl.col("_sig_line") * (1.0 - threshold / 1000.0))
        .then(-1.0)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_voltgt_momentum(
    bars: pl.DataFrame, *, lookback: int, threshold: float
) -> pl.Series:
    """Sign of N-day return, sized inversely to realised vol; threshold sets the
    target annualised vol the position is sized to."""
    df = bars.with_columns(
        (pl.col("close").log() - pl.col("close").shift(lookback).log()).alias("_r")
    )
    df = df.with_columns(
        (
            pl.col("close").log() - pl.col("close").shift(1).log()
        ).rolling_std(window_size=lookback, min_samples=lookback).alias("_vol")
    )
    # target vol in decimal points; threshold ∈ [0.5,2.5] → target_vol ∈ [4%, 20%]
    target_vol = 0.04 * threshold
    df = df.with_columns(
        (target_vol / (pl.col("_vol").clip(lower_bound=1e-6) * (252 ** 0.5))).clip(
            lower_bound=0.0, upper_bound=1.0
        ).alias("_size")
    )
    df = df.with_columns(
        pl.when(pl.col("_r") > 0)
        .then(pl.col("_size"))
        .when(pl.col("_r") < 0)
        .then(-pl.col("_size"))
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_zscore_meanrevert(
    bars: pl.DataFrame, *, lookback: int, threshold: float
) -> pl.Series:
    df = bars.with_columns(
        _rolling_mean("close", lookback).alias("_mu"),
        _rolling_std("close", lookback).alias("_sd"),
    )
    df = df.with_columns(
        ((pl.col("close") - pl.col("_mu")) / pl.col("_sd").clip(lower_bound=1e-9)).alias("_z")
    )
    df = df.with_columns(
        pl.when(pl.col("_z") < -threshold)
        .then(1.0)
        .when(pl.col("_z") > threshold)
        .then(-1.0)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_aroon(bars: pl.DataFrame, *, lookback: int, threshold: float) -> pl.Series:
    """Aroon Up = ((lookback - days_since_high) / lookback) × 100, etc.
    threshold sets the activation gap: aroon_up - aroon_down > threshold → buy.
    """
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    n = len(bars)
    aroon_up = pl.Series([0.0] * n)
    aroon_down = pl.Series([0.0] * n)
    import numpy as np
    up_vals = np.full(n, np.nan, dtype=np.float64)
    down_vals = np.full(n, np.nan, dtype=np.float64)
    for i in range(lookback, n):
        window_high = high[i - lookback + 1 : i + 1]
        window_low = low[i - lookback + 1 : i + 1]
        days_since_high = lookback - 1 - int(window_high.argmax())
        days_since_low = lookback - 1 - int(window_low.argmin())
        up_vals[i] = (lookback - days_since_high) / lookback * 100.0
        down_vals[i] = (lookback - days_since_low) / lookback * 100.0
    aroon_up = pl.Series("up", up_vals, dtype=pl.Float64)
    aroon_down = pl.Series("down", down_vals, dtype=pl.Float64)
    df = bars.with_columns(aroon_up.alias("_aup"), aroon_down.alias("_adn"))
    df = df.with_columns(
        pl.when(pl.col("_aup") - pl.col("_adn") > threshold * 20.0)
        .then(1.0)
        .when(pl.col("_adn") - pl.col("_aup") > threshold * 20.0)
        .then(-1.0)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_stochastic(
    bars: pl.DataFrame, *, lookback: int, threshold: float
) -> pl.Series:
    df = bars.with_columns(
        _rolling_max("high", lookback).alias("_hh"),
        _rolling_min("low", lookback).alias("_ll"),
    )
    df = df.with_columns(
        (
            100.0 * (pl.col("close") - pl.col("_ll"))
            / (pl.col("_hh") - pl.col("_ll")).clip(lower_bound=1e-9)
        ).alias("_k")
    )
    df = df.with_columns(
        pl.when(pl.col("_k") < 50.0 - threshold * 10.0)
        .then(1.0)
        .when(pl.col("_k") > 50.0 + threshold * 10.0)
        .then(-1.0)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_roc(bars: pl.DataFrame, *, lookback: int, threshold: float) -> pl.Series:
    df = bars.with_columns(
        (pl.col("close") / pl.col("close").shift(lookback) - 1.0).alias("_roc")
    )
    df = df.with_columns(
        pl.when(pl.col("_roc") > threshold / 100.0)
        .then(1.0)
        .when(pl.col("_roc") < -threshold / 100.0)
        .then(-1.0)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_cci(bars: pl.DataFrame, *, lookback: int, threshold: float) -> pl.Series:
    df = bars.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_tp")
    )
    df = df.with_columns(
        _rolling_mean("_tp", lookback).alias("_tp_ma"),
        pl.col("_tp")
        .rolling_map(
            lambda s: float(((s - s.mean()).abs().mean()) or 0.0),  # type: ignore[arg-type]
            window_size=lookback,
        )
        .alias("_md"),
    )
    df = df.with_columns(
        ((pl.col("_tp") - pl.col("_tp_ma")) / (0.015 * pl.col("_md").clip(lower_bound=1e-9))).alias("_cci")
    )
    df = df.with_columns(
        pl.when(pl.col("_cci") > threshold * 50.0)
        .then(1.0)
        .when(pl.col("_cci") < -threshold * 50.0)
        .then(-1.0)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


def signal_keltner_breakout(
    bars: pl.DataFrame, *, lookback: int, threshold: float
) -> pl.Series:
    df = bars.with_columns(
        _rolling_mean("close", lookback).alias("_mid"),
        (pl.col("high") - pl.col("low")).rolling_mean(
            window_size=lookback, min_samples=lookback
        ).alias("_atr"),
    )
    df = df.with_columns(
        (pl.col("_mid") + threshold * pl.col("_atr")).alias("_upper"),
        (pl.col("_mid") - threshold * pl.col("_atr")).alias("_lower"),
    )
    df = df.with_columns(
        pl.when(pl.col("close") > pl.col("_upper"))
        .then(1.0)
        .when(pl.col("close") < pl.col("_lower"))
        .then(-1.0)
        .otherwise(0.0)
        .alias("_s")
    )
    return df["_s"]


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


SignalFn = Callable[[pl.DataFrame], pl.Series]
SIGNAL_FAMILIES: dict[str, Callable[..., pl.Series]] = {
    "TS_MOMENTUM": signal_ts_momentum,
    "LAGGED_MOMENTUM": signal_lagged_momentum,
    "MA_CROSSOVER": signal_ma_crossover,
    "DONCHIAN_BREAKOUT": signal_donchian_breakout,
    "BOLLINGER_REVERT": signal_bollinger_revert,
    "BOLLINGER_BREAKOUT": signal_bollinger_breakout,
    "RSI_MEANREVERT": signal_rsi_meanrevert,
    "MACD": signal_macd,
    "VOLTGT_MOMENTUM": signal_voltgt_momentum,
    "ZSCORE_MEANREVERT": signal_zscore_meanrevert,
    "AROON": signal_aroon,
    "STOCHASTIC": signal_stochastic,
    "ROC": signal_roc,
    "CCI": signal_cci,
    "KELTNER_BREAKOUT": signal_keltner_breakout,
    "VOLMANAGED_MOMENTUM": signal_volmanaged_momentum,
    "EWMA_CROSS": signal_ewma_cross,
    "ATR_TRAILING_TREND": signal_atr_trailing_trend,
    "ROLLING_SHARPE_MOM": signal_rolling_sharpe_mom,
    "RANGE_OSCILLATOR": signal_range_oscillator,
    "MOM_SKIP": signal_mom_skip,
}
