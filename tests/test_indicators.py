import numpy as np
import pandas as pd
import pytest

from quantex.indicators import sma, ema, rsi, bollinger_bands


@pytest.fixture(scope="module")
def price_series() -> pd.Series:
    """Generates a deterministic price series for testing."""
    rng = np.random.default_rng(seed=42)
    # Simulate a random walk with slight drift – 5,000 bars for speed
    steps = rng.normal(loc=0.0002, scale=0.01, size=5_000)
    prices = 100 + np.cumsum(steps)
    return pd.Series(prices)


# ---------------------------------------------------------------------------
# SMA / EMA
# ---------------------------------------------------------------------------


def test_sma_matches_manual():
    ser = pd.Series([1, 2, 3, 4, 5])
    result = sma(ser, period=3)
    # Last SMA should be mean of [3,4,5] => 4.0
    assert result.iloc[-1] == pytest.approx(4.0)


def test_sma_equals_pandas(price_series):
    period = 20
    expected = price_series.rolling(window=period, min_periods=period).mean()
    got = sma(price_series, period=period)
    pd.testing.assert_series_equal(got, expected)


def test_ema_equals_pandas(price_series):
    period = 15
    expected = price_series.ewm(span=period, adjust=False).mean()
    got = ema(price_series, period=period)
    pd.testing.assert_series_equal(got, expected)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def test_rsi_bounds(price_series):
    result = rsi(price_series, period=14)
    assert (result.dropna() >= 0).all() and (result.dropna() <= 100).all()


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def test_bollinger_bands_columns(price_series):
    bands = bollinger_bands(price_series, period=20, std_dev=2.0)
    assert set(bands.columns) == {"upper", "middle", "lower"}


def test_bollinger_last_row(price_series):
    period = 20
    std_mult = 2.0
    bands = bollinger_bands(price_series, period=period, std_dev=std_mult)

    sma_series = sma(price_series, period=period)
    rolling_std = price_series.rolling(window=period, min_periods=period).std()

    last_upper_expected = sma_series.iloc[-1] + std_mult * rolling_std.iloc[-1]
    last_lower_expected = sma_series.iloc[-1] - std_mult * rolling_std.iloc[-1]

    assert bands["upper"].iloc[-1] == pytest.approx(last_upper_expected)
    assert bands["lower"].iloc[-1] == pytest.approx(last_lower_expected)
