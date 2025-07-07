import pytest
import pandas as pd
from io import StringIO
from pathlib import Path

from quantex.sources import DataSource, BacktestingDataSource, CSVDataSource
from quantex.models import Bar


def test_datasource_methods_raise_notimplemented():
    """Abstract DataSource should raise for unimplemented API."""
    ds = DataSource()
    with pytest.raises(NotImplementedError):
        ds.get_current_bar()
    with pytest.raises(NotImplementedError):
        ds.get_lookback_data(1)


def test_backtesting_datasource_methods_raise_notimplemented():
    bds = BacktestingDataSource()
    with pytest.raises(NotImplementedError):
        len(bds)
    with pytest.raises(NotImplementedError):
        bds.get_current_bar()
    with pytest.raises(NotImplementedError):
        bds.get_lookback_data(1)


class DummyDataSource(DataSource):
    """Concrete DataSource that returns sequential integers as close prices."""

    def __init__(self, prices):
        self._df = pd.DataFrame({"close": prices})
        self.index = 0

    def get_current_bar(self):
        ts = pd.Timestamp(self.index, unit="s")
        close = self._df.iloc[self.index]["close"]
        return Bar(
            timestamp=ts, open=close, high=close, low=close, close=close, volume=0
        )

    def get_lookback_data(self, lookback_period):
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1].copy()


class DummyBacktestingDataSource(BacktestingDataSource):
    def __init__(self, df):
        self._df = df.reset_index(drop=True)
        self.index = 0

    def __len__(self):
        return len(self._df)

    def get_current_bar(self):
        row = self._df.iloc[self.index]
        return Bar(
            timestamp=pd.Timestamp(self.index, unit="s"),
            open=row["price"],
            high=row["price"],
            low=row["price"],
            close=row["price"],
            volume=0,
        )

    def get_lookback_data(self, lookback_period):
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1].copy()


def test_dummy_datasource_behaviour():
    prices = [10, 20, 30, 40, 50]
    ds = DummyDataSource(prices)
    ds._increment_index()
    ds._increment_index()
    bar = ds.get_current_bar()
    assert bar.close == 30
    ds._increment_index()
    ds._increment_index()
    window = ds.get_lookback_data(3)
    assert list(window["close"]) == [30, 40, 50]


def test_dummy_backtesting_datasource_behaviour():
    df = pd.DataFrame({"price": [100, 101, 102, 103]})
    bds = DummyBacktestingDataSource(df)
    assert len(bds) == 4
    bds._increment_index()
    assert bds.get_current_bar().close == 101
    bds._increment_index()
    bds._increment_index()
    window = bds.get_lookback_data(2)
    assert list(window["price"]) == [102, 103]


def test_csv_datasource(tmp_path: Path):
    # create temporary CSV file
    csv_content = StringIO(
        """timestamp,open,high,low,close,volume\n2024-01-01 00:00:00,1,2,0.5,1.5,100\n2024-01-01 00:01:00,1.5,2.5,1,2,150\n"""
    )
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(csv_content.getvalue())

    ds = CSVDataSource(csv_path, symbol="TEST")
    assert len(ds) == 2
    bar0 = ds.get_current_bar()
    assert bar0.open == 1 and bar0.close == 1.5 and bar0.symbol == "TEST"
    ds._increment_index()
    bar1 = ds.get_current_bar()
    assert bar1.high == 2.5
    lookback = ds.get_lookback_data(2)
    assert len(lookback) == 2
