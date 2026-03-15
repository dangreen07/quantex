import numpy as np
import pandas as pd
from typing import cast

from quantex.backtester import SimpleBacktester
from quantex.datasource import DataSource
from quantex.strategy import Strategy


class StaticIndicatorStrategy(Strategy):
    def init(self):
        values = np.array([np.nan, np.nan, 3.0, 4.0, 5.0, 6.0], dtype=float)
        self.test_indicator = self.Indicator(values)
        self.seen_values = []

    def next(self):
        if len(self.data["EURUSD"].Close) >= 3:
            self.seen_values.append(float(self.test_indicator[-1]))


class InitDataVisibilityStrategy(Strategy):
    def init(self):
        self.init_close_values = list(self.data["EURUSD"].Close)

    def next(self):
        pass


def test_indicator_values_progress_through_backtest_without_reset_corruption():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5, 6],
            "High": [1, 2, 3, 4, 5, 6],
            "Low": [1, 2, 3, 4, 5, 6],
            "Close": [1, 2, 3, 4, 5, 6],
            "Volume": [100] * 6,
        },
        index=dates,
    )

    original_strategy = StaticIndicatorStrategy()
    original_strategy.add_data(DataSource(df), "EURUSD")

    backtester = SimpleBacktester(original_strategy)
    backtester.run(progress_bar=False)

    strategy = cast(StaticIndicatorStrategy, backtester.strategy)
    assert strategy.seen_values == [3.0, 4.0, 5.0, 6.0]


def test_init_has_access_to_full_datasource_history():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Open": [10, 11, 12, 13],
            "High": [10, 11, 12, 13],
            "Low": [10, 11, 12, 13],
            "Close": [10, 11, 12, 13],
            "Volume": [100] * 4,
        },
        index=dates,
    )

    original_strategy = InitDataVisibilityStrategy()
    original_strategy.add_data(DataSource(df), "EURUSD")

    backtester = SimpleBacktester(original_strategy)
    backtester.run(progress_bar=False)

    strategy = cast(InitDataVisibilityStrategy, backtester.strategy)
    assert strategy.init_close_values == [10, 11, 12, 13]
