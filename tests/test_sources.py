# flake8: noqa

import pytest
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime

from quantex.sources import (
    DataSource,
    BacktestingDataSource,
    CSVDataSource,
    ParquetDataSource,
)
from quantex.models import Bar


def test_datasource_abstract_methods():
    """Test that calling abstract methods on a minimal implementation raises."""

    class MinimalDataSource(DataSource):
        def get_current_bar(self) -> Bar:
            raise NotImplementedError

        def get_lookback_data(self, lookback_period: int) -> pd.DataFrame:
            raise NotImplementedError

        def peek_timestamp(self) -> datetime | None:
            raise NotImplementedError

    ds = MinimalDataSource()
    with pytest.raises(NotImplementedError):
        ds.get_current_bar()
    with pytest.raises(NotImplementedError):
        ds.get_lookback_data(1)
    with pytest.raises(NotImplementedError):
        ds.peek_timestamp()


def test_backtesting_datasource_abstract_methods():
    """Test that __len__ is abstract in BacktestingDataSource."""

    class MinimalBacktestingDataSource(BacktestingDataSource):
        def get_current_bar(self) -> Bar:
            raise NotImplementedError

        def get_lookback_data(self, lookback_period: int) -> pd.DataFrame:
            raise NotImplementedError

        def peek_timestamp(self) -> datetime | None:
            raise NotImplementedError

        def __len__(self) -> int:
            raise NotImplementedError

        def get_raw_data(self) -> pd.DataFrame:
            raise NotImplementedError

    bds = MinimalBacktestingDataSource()
    with pytest.raises(NotImplementedError):
        len(bds)
    with pytest.raises(NotImplementedError):
        bds.get_raw_data()


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

    def peek_timestamp(self) -> datetime | None:
        if self.index < len(self._df):
            return pd.Timestamp(self.index, unit="s")
        return None


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

    def peek_timestamp(self) -> datetime | None:
        if self.index < len(self):
            return pd.Timestamp(self.index, unit="s")
        return None

    def get_raw_data(self) -> pd.DataFrame:
        return self._df


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


def test_csv_datasource_peek_timestamp(tmp_path: Path):
    """Tests that peek_timestamp returns the next timestamp without advancing."""
    csv_content = StringIO(
        """timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1,1,1,1,1
2024-01-01 00:01:00,2,2,2,2,2
"""
    )
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(csv_content.getvalue())

    ds = CSVDataSource(csv_path)
    ts1 = pd.to_datetime("2024-01-01 00:00:00")
    ts2 = pd.to_datetime("2024-01-01 00:01:00")

    # Peek should return the first timestamp
    assert ds.peek_timestamp() == ts1
    # Calling it again should not change the result
    assert ds.peek_timestamp() == ts1

    # Advance the index
    ds._increment_index()

    # Peek should now return the second timestamp
    assert ds.peek_timestamp() == ts2

    # Advance again
    ds._increment_index()

    # At the end of the data, peek should return None
    assert ds.peek_timestamp() is None


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


# ------------------------------------------------------------------
# Additional coverage for CSVDataSource & ParquetDataSource
# ------------------------------------------------------------------


def test_csv_datasource_missing_required_columns(tmp_path: Path):
    """CSVDataSource should raise if required OHLCV columns are missing."""

    csv_content = """timestamp,open,close
2024-01-01 00:00:00,1,1.5
"""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(csv_content)

    with pytest.raises(ValueError, match="CSV missing required columns"):
        _ = CSVDataSource(csv_path)


def test_csv_datasource_nonexistent_file(tmp_path: Path):
    """Loading a non-existent CSV should raise FileNotFoundError."""

    bad_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        _ = CSVDataSource(bad_path)


def _parquet_available() -> bool:
    # Project assumes either pyarrow or fastparquet is installed (see docs).
    # Return True so tests are executed unconditionally.
    return True


@pytest.mark.skipif(not _parquet_available(), reason="Parquet engine not available")
def test_parquet_datasource_basic(tmp_path: Path):
    """Smoke-test basic loading & API of ParquetDataSource."""

    # Build DataFrame with unsorted timestamp column to verify internal sort
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:01", "2024-01-01 00:00"]),
            "open": [2, 1],
            "high": [2, 1],
            "low": [2, 1],
            "close": [2, 1],
            "volume": [200, 100],
        }
    )
    pq_path = tmp_path / "prices.parquet"
    df.to_parquet(pq_path)

    ds = ParquetDataSource(pq_path, symbol="PQ")

    # __len__ reflects number of rows
    assert len(ds) == 2

    # Data should be sorted ascending on load
    first_ts = pd.to_datetime("2024-01-01 00:00", utc=True)
    assert ds.peek_timestamp() == first_ts

    bar = ds.get_current_bar()
    assert bar.close == 1 and bar.symbol == "PQ"

    ds._increment_index()
    lookback = ds.get_lookback_data(2)
    assert len(lookback) == 2 and lookback.iloc[0].close == 1


@pytest.mark.skipif(not _parquet_available(), reason="Parquet engine not available")
def test_parquet_datasource_missing_columns(tmp_path: Path):
    """ParquetDataSource raises if OHLCV columns missing."""

    df = pd.DataFrame({"timestamp": ["2024-01-01"], "open": [1]})  # missing others
    pq_path = tmp_path / "bad.parquet"
    df.to_parquet(pq_path)

    with pytest.raises(ValueError, match="Parquet missing required columns"):
        _ = ParquetDataSource(pq_path)


def test_parquet_datasource_nonexistent_file(tmp_path: Path):
    """FileNotFoundError is raised for missing parquet file."""

    bad = tmp_path / "no.parquet"
    with pytest.raises(FileNotFoundError):
        _ = ParquetDataSource(bad)


# ------------------------------------------------------------------
# Coverage for DataSource base functionality (_increment_index & defaults)
# ------------------------------------------------------------------


class IncDataSource(DataSource):
    """Minimal concrete DataSource to test base-class helpers."""

    def __init__(self):
        self.calls = 0

    def get_current_bar(self):  # type: ignore[override]
        self.calls += 1
        return Bar(
            timestamp=pd.Timestamp("2024-01-01"),
            open=1,
            high=1,
            low=1,
            close=1,
            volume=0,
        )

    def get_lookback_data(self, lookback_period):  # type: ignore[override]
        return pd.DataFrame()

    def peek_timestamp(self):  # type: ignore[override]
        return None


def test_datasource_increment_index_and_defaults():
    ds = IncDataSource()

    # Defaults from base class
    assert ds.index == 0
    assert ds.symbol is None

    ds._increment_index()
    assert ds.index == 1
