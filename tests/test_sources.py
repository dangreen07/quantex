import pytest
import pandas as pd
from quantex.sources import DataSource, BacktestingDataSource


def test_datasource_methods_raise_notimplemented():
    """Base DataSource methods should raise NotImplementedError when called directly."""
    ds = DataSource()
    with pytest.raises(NotImplementedError):
        ds.get_current_row()
    with pytest.raises(NotImplementedError):
        ds.get_lookback_data(1)


def test_backtesting_datasource_methods_raise_notimplemented():
    """Base BacktestingDataSource should raise NotImplementedError for all abstract methods."""
    bds = BacktestingDataSource()
    with pytest.raises(NotImplementedError):
        len(bds)
    with pytest.raises(NotImplementedError):
        bds.get_current_row()
    with pytest.raises(NotImplementedError):
        bds.get_lookback_data(1)


class DummyDataSource(DataSource):
    """Simple concrete implementation of DataSource for unit testing."""

    def __init__(self, data):
        # Store data as a pandas DataFrame for convenience
        self._df = pd.DataFrame({"value": data})

    def get_current_row(self):
        return self._df.iloc[self.index]["value"]

    def get_lookback_data(self, lookback_period: int):
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1].copy()


class DummyBacktestingDataSource(BacktestingDataSource):
    """Concrete implementation of BacktestingDataSource for unit testing."""

    def __init__(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True)

    def __len__(self):
        return len(self._df)

    def get_current_row(self):
        return self._df.iloc[self.index]

    def get_lookback_data(self, lookback_period: int):
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1].copy()


def test_dummy_datasource_returns_correct_row_and_window():
    data = [10, 20, 30, 40, 50]
    ds = DummyDataSource(data)

    # Increment index
    ds._increment_index()
    ds._increment_index()

    # Single row
    assert ds.get_current_row() == 30

    # Increment index
    ds._increment_index()
    ds._increment_index()

    # Rolling window (lookback=3 at idx=4 should return 3 values: 30, 40, 50)
    window = ds.get_lookback_data(3)
    assert list(window["value"]) == [30, 40, 50]


def test_dummy_backtesting_datasource_behaviour():
    df = pd.DataFrame({"price": [100, 101, 102, 103]})
    bds = DummyBacktestingDataSource(df)

    # __len__ implementation
    assert len(bds) == 4

    # Increment index
    bds._increment_index()

    # Single row
    row = bds.get_current_row()
    assert row["price"] == 101

    # Increment index
    bds._increment_index()
    bds._increment_index()

    # Data window
    window = bds.get_lookback_data(2)  # expect rows with prices 102, 103
    assert list(window["price"]) == [102, 103]
