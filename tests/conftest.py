import pandas as pd
import pytest


@pytest.fixture
def ohlcv_data():
    """Deterministic OHLCV data shared across the suite.

    Using fixed values keeps failures reproducible and makes it easier to
    reason about order execution, indicator values, and PnL calculations.
    """
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            "Open": list(range(100, 120)),
            "High": list(range(105, 125)),
            "Low": list(range(95, 115)),
            "Close": list(range(102, 122)),
            "Volume": [1000] * 20,
        },
        index=dates,
    )


@pytest.fixture
def short_ohlcv_data():
    """Smaller deterministic OHLCV sample for focused unit tests."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "Open": list(range(100, 110)),
            "High": list(range(105, 115)),
            "Low": list(range(95, 105)),
            "Close": list(range(102, 112)),
            "Volume": [1000] * 10,
        },
        index=dates,
    )
