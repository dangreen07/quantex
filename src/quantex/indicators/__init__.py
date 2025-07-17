"""quantex.indicators — Technical analysis helpers.

This sub-module provides a lightweight collection of **pure-Pandas**
implementations for the most common technical indicators used in
systematic trading.  Each indicator is exposed as a plain function so
users can simply ::

    from quantex.indicators import sma, rsi

    df["SMA_20"] = sma(df["close"], period=20)

All functions preserve the input index, return a :class:`pandas.Series`
(or :class:`pandas.DataFrame` when multiple outputs are appropriate) and
never mutate the supplied data.

The implementations favour correctness and clarity first; pure-Python
loops are avoided in favour of vectorised NumPy / pandas ops.  For
large-scale back-tests you can later swap these with TA-Lib/Cython
bindings while keeping the same import surface.
"""

from __future__ import annotations
from typing import Final, Tuple
import pandas as pd

__all__: Final[Tuple[str, ...]] = (
    "sma",
    "ema",
    "rsi",
    "bollinger_bands",
)

# ---------------------------------------------------------------------------
# Trend indicators
# ---------------------------------------------------------------------------


def sma(
    series: pd.Series, period: int = 20, *, min_periods: int | None = None
) -> pd.Series:
    """Simple Moving Average (SMA).

    Args:
        series: Input price series (typically `close`).
        period: Look-back window size.
        min_periods: Minimum required observations for a value to be
            computed. Defaults to the *same* as `period`, emitting NaNs
            until the window is fully populated.

    Returns:
        pandas.Series: SMA values aligned with *series.index*.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    return series.rolling(window=period, min_periods=min_periods or period).mean()


def ema(
    series: pd.Series,
    period: int = 20,
    *,
    adjust: bool = False,
) -> pd.Series:
    """Exponential Moving Average (EMA).

    Args:
        series: Input price series.
        period: Smoothing period.
        adjust: Whether to *adjust* the weights as per pandas `.ewm`
            semantics.  The default (`False`) matches most TA libraries.

    Returns:
        pandas.Series: EMA values.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    return series.ewm(span=period, adjust=adjust).mean()


# ---------------------------------------------------------------------------
# Momentum indicators
# ---------------------------------------------------------------------------


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI).

    Implementation follows the classic Wilder smoothing.

    Args:
        series: Price series (usually close).
        period: Look-back period for average gains/losses.

    Returns:
        pandas.Series in the [0, 100] range.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing (alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series


# ---------------------------------------------------------------------------
# Volatility indicators
# ---------------------------------------------------------------------------


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    *,
    std_dev: float = 2.0,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Bollinger Bands (upper, middle, lower).

    Args:
        series: Input price series.
        period: Moving-average window.
        std_dev: Standard-deviation multiplier for the bands.
        min_periods: Forwarded to the underlying SMA calculation.

    Returns:
        DataFrame with columns ``upper``, ``middle`` (SMA), and ``lower``.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    if std_dev <= 0:
        raise ValueError("std_dev must be > 0")

    sma_series = sma(series, period=period, min_periods=min_periods)
    rolling_std = series.rolling(window=period, min_periods=min_periods or period).std()

    upper = sma_series + std_dev * rolling_std
    lower = sma_series - std_dev * rolling_std

    return pd.DataFrame({"upper": upper, "middle": sma_series, "lower": lower})
