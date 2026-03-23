import pandas as pd

from quantex.backtester import _infer_periods_per_year, max_drawdown


def test_max_drawdown():
    """Max drawdown should report the peak-to-trough decline as a positive number."""
    equity = pd.Series([100, 110, 105, 95, 100])
    mdd = max_drawdown(equity)
    expected_mdd = (95 - 110) / 110
    assert abs(mdd - abs(expected_mdd)) < 1e-6


def test_infer_periods_per_year():
    """The period inference helper should distinguish daily and empty indexes."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    periods = _infer_periods_per_year(dates)
    assert periods == 252

    periods = _infer_periods_per_year(pd.DatetimeIndex([]))
    assert periods == 252 * 24 * 60