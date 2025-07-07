import pytest
import pandas as pd
from quantex.sources import DataSource, BacktestingDataSource

def test_datasource_methods_raise_notimplemented():
    """Base DataSource methods should raise NotImplementedError when called directly."""
    ds = DataSource()
    with pytest.raises(NotImplementedError):
        ds.get_row(0)
    with pytest.raises(NotImplementedError):
        ds.get_data_before_idx(0, 1)


def test_backtesting_datasource_methods_raise_notimplemented():
    """Base BacktestingDataSource should raise NotImplementedError for all abstract methods."""
    bds = BacktestingDataSource()
    with pytest.raises(NotImplementedError):
        len(bds)
    with pytest.raises(NotImplementedError):
        bds.get_row(0)
    with pytest.raises(NotImplementedError):
        bds.get_data_before_idx(0, 1)


class DummyDataSource(DataSource):
    """Simple concrete implementation of DataSource for unit testing."""

    def __init__(self, data):
        # Store data as a pandas DataFrame for convenience
        self._df = pd.DataFrame({"value": data})

    def get_row(self, idx: int):
        return self._df.iloc[idx]["value"]

    def get_data_before_idx(self, idx: int, lookback_period: int):
        start = max(0, idx - lookback_period + 1)
        return self._df.iloc[start: idx + 1].copy()


class DummyBacktestingDataSource(BacktestingDataSource):
    """Concrete implementation of BacktestingDataSource for unit testing."""

    def __init__(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True)

    def __len__(self):
        return len(self._df)

    def get_row(self, idx: int):
        return self._df.iloc[idx]

    def get_data_before_idx(self, idx: int, lookback_period: int):
        start = max(0, idx - lookback_period + 1)
        return self._df.iloc[start: idx + 1].copy()


def test_dummy_datasource_returns_correct_row_and_window():
    data = [10, 20, 30, 40, 50]
    ds = DummyDataSource(data)

    # Single row
    assert ds.get_row(2) == 30

    # Rolling window (lookback=3 at idx=4 should return 3 values: 30, 40, 50)
    window = ds.get_data_before_idx(4, 3)
    assert list(window["value"]) == [30, 40, 50]


def test_dummy_backtesting_datasource_behaviour():
    df = pd.DataFrame({"price": [100, 101, 102, 103]})
    bds = DummyBacktestingDataSource(df)

    # __len__ implementation
    assert len(bds) == 4

    # Single row
    row = bds.get_row(1)
    assert row["price"] == 101

    # Data window
    window = bds.get_data_before_idx(3, 2)  # expect rows with prices 102, 103
    assert list(window["price"]) == [102, 103]