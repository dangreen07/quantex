"""Tests for Strategy.price_history and get_lookback_prices helpers."""

import pandas as pd

from quantex.backtest import BacktestRunner
from quantex.sources import BacktestingDataSource
from quantex.models import Bar
from quantex.strategy import Strategy


class DFDataSource(BacktestingDataSource):
    """In-memory OHLCV source backed by a DataFrame (timestamp index)."""

    def __init__(self, df: pd.DataFrame, symbol: str):
        self._df = df
        self.symbol = symbol
        self.index = 0

    # ------------------------------------------------------------------
    # BacktestingDataSource interface
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._df)

    def get_raw_data(self):
        return self._df

    def peek_timestamp(self):
        if self.index < len(self):
            return self._df.index[self.index]
        return None

    def get_current_bar(self):
        row = self._df.iloc[self.index]
        ts = self._df.index[self.index]
        return Bar(ts, row.open, row.high, row.low, row.close, row.volume, self.symbol)

    def get_lookback_data(self, lookback_period):
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1]


class LookbackAssertStrategy(Strategy):
    """Checks that lookback windows are aligned and NaN-free on every bar."""

    def __init__(self, *args, lookback_period: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookback_period = lookback_period
        self.mismatches: int = 0

    def run(self):
        # Skip until enough history has accumulated
        if self.index < self.lookback_period - 1:
            return

        window = self.get_lookback_prices(self.lookback_period)

        # 1. Window should have no NaNs because prices are forward-filled
        if window.isna().any().any():
            self.mismatches += 1

        # 2. Last index must equal current timestamp
        assert window.index[-1] == self.timestamp

        # 3. All expected symbols present
        assert set(window.columns) == set(self.symbols)


def _make_df(prices: list[float], idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Utility to build an OHLCV frame where all OHLC columns equal *price*."""

    template = {
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1000] * len(prices),
    }
    return pd.DataFrame(template, index=idx)


def test_aligned_lookback_helpers():
    # Create mis-aligned indices
    base_idx = pd.date_range("2024-01-01 09:30", periods=4, freq="min", tz="UTC")
    idx_a = base_idx[[0, 2]]  # 09:30, 09:32
    idx_b = base_idx[[0, 1, 3]]  # 09:30, 09:31, 09:33

    df_a = _make_df([100, 102], idx_a)
    df_b = _make_df([200, 201, 203], idx_b)

    ds_a = DFDataSource(df_a, symbol="A")
    ds_b = DFDataSource(df_b, symbol="B")

    strat = LookbackAssertStrategy(symbols=["A", "B"], initial_cash=10_000)
    runner = BacktestRunner(strat, {"a": ds_a, "b": ds_b})
    runner.run()

    # The strategy tracked any mismatches internally
    assert strat.mismatches == 0
