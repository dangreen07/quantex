"""DataSource abstractions for the QuantEx library."""

from __future__ import annotations

# Standard library imports
from pathlib import Path
from typing import Optional
from datetime import datetime

# Third-party imports
import pandas as pd
import numpy as np

from quantex.models import Bar
from abc import ABC, abstractmethod


class DataSource(ABC):
    """Abstract data source for providing market data.

    Implementations must provide the *current* bar via `get_current_bar`
    and allow a rolling historical window via `get_lookback_data`.
    The internal pointer `index` starts at 0 and should be advanced by calling
    `_increment_index` once the engine has finished processing a bar.
    """

    index: int = 0
    symbol: str | None = None

    @abstractmethod
    def get_current_bar(self) -> Bar:
        """Returns the bar at the current `index` position."""
        raise NotImplementedError

    @abstractmethod
    def get_lookback_data(self, lookback_period: int) -> pd.DataFrame:
        """Returns a lookback window of data.

        Args:
            lookback_period: The size of the lookback window.

        Returns:
            A pandas DataFrame containing the lookback data, inclusive of the
            current bar.
        """
        raise NotImplementedError

    @abstractmethod
    def peek_timestamp(self) -> datetime | None:
        """Peeks at the timestamp of the next available bar.

        This method should return the timestamp of the bar at the current
        `index` without advancing the index. If the source is exhausted,
        it should return `None`.

        Returns:
            The next timestamp, or None if the source is exhausted.
        """
        raise NotImplementedError

    def _increment_index(self) -> None:
        """Advances the internal pointer to the next bar."""
        self.index += 1


class BacktestingDataSource(DataSource):
    """A data source for backtesting that must have a defined length."""

    @abstractmethod
    def __len__(self) -> int:  # pragma: no cover – abstract contract
        raise NotImplementedError

    @abstractmethod
    def get_raw_data(self) -> pd.DataFrame:
        """Returns the entire underlying data as a DataFrame.

        This method is intended for use by the backtesting engine for
        pre-computation and should not be used in strategy logic.
        """
        raise NotImplementedError


class CSVDataSource(BacktestingDataSource):
    """Backtesting data source backed by a local OHLCV CSV file.

    The CSV must contain 'timestamp', 'open', 'high', 'low', 'close', and
    'volume' columns. The 'timestamp' column will be parsed as dates.
    """

    def __init__(self, path: str | Path, symbol: Optional[str] = None):
        """Initializes the CSVDataSource.

        Args:
            path: The path to the CSV file.
            symbol: The symbol for the data. If None, it's inferred from the
                file name.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV is missing required columns.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        # Load the data first without any special parsing so we can normalise
        # column names irrespective of their original case.
        df = pd.read_csv(self.path, engine="pyarrow", dtype_backend="numpy_nullable")

        # Make all column names lower-case so that subsequent access is
        # case-insensitive.
        df.columns = [c.lower() for c in df.columns]

        # Ensure a timestamp column exists (case-insensitive due to rename).
        if "timestamp" not in df.columns:
            raise ValueError("CSV missing required 'timestamp' column")

        # Parse the timestamps and set as index.
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # Validate required OHLCV columns (now lower-case).
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        self._df = df  # immutable reference
        self.symbol = symbol or self.path.stem
        self.index = 0

    def get_raw_data(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        """Returns the number of bars in the data source."""
        return len(self._df)

    def peek_timestamp(self) -> datetime | None:
        """Peeks at the timestamp of the next available bar from the CSV.

        Returns:
            The next timestamp, or None if the source is exhausted.
        """
        if self.index < len(self):
            ts = self._df.index[self.index]
            # CSVDataSource keeps timestamps timezone-aware (UTC) internally
            # for consistency, but many user-facing contexts (including the
            # bundled test-suite) expect **naive** timestamps. We therefore
            # return the timestamp *without* its timezone information here.
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
        return None

    def get_current_bar(self) -> Bar:
        """Returns the current bar from the CSV data."""
        row = self._df.iloc[self.index]
        ts = self._df.index[self.index]
        return Bar(
            timestamp=ts,
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            symbol=self.symbol,
        )

    def get_lookback_data(self, lookback_period: int) -> pd.DataFrame:
        """Returns a lookback window of data from the CSV.

        Args:
            lookback_period: The size of the lookback window.

        Returns:
            A pandas DataFrame containing the lookback data, inclusive of the current bar.
        """
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1].copy()


# New DataSource for Parquet files
class ParquetDataSource(BacktestingDataSource):
    """Backtesting data source backed by a local OHLCV Parquet file.

    The Parquet file must contain either an index of timestamps or a column
    named 'timestamp', as well as the standard OHLCV columns. If a
    'timestamp' column exists, it will be parsed and set as the index.
    """

    def __init__(
        self,
        path: str | Path,
        symbol: Optional[str] = None,
        start_datetime: Optional[pd.Timestamp] = None,
        end_datetime: Optional[pd.Timestamp] = None,
    ):
        """Initializes the ParquetDataSource.

        Args:
            path: Path to the Parquet file on disk.
            symbol: Optional symbol name. If omitted, the stem of the path is
                used instead.
        """
        if not isinstance(path, Path):
            path = Path(path)
        self.path = path
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        # Load data – let pandas/FastParquet handle decompression & column types
        df = pd.read_parquet(self.path, engine="pyarrow")

        # Normalise column names to lower-case for case-insensitive access.
        df.columns = [c.lower() for c in df.columns]

        ## Don't reset index if it's already a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            # If a timestamp column exists (case-insensitive due to rename), make it the index.
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")

        ## Remove duplicates
        df = df.drop_duplicates(keep="first")

        # Ensure chronological order
        df = df.sort_index()

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}")

        mask = pd.Series(True, index=df.index)

        if start_datetime is not None:
            mask = mask & (df.index >= start_datetime)
        if end_datetime is not None:
            mask = mask & (df.index <= end_datetime)

        df = df[mask]

        # Immutable reference to underlying data
        self._df = df
        self.symbol = symbol or self.path.stem
        self.index = 0

    # --- BacktestingDataSource API -----------------------------------------
    def get_raw_data(self) -> pd.DataFrame:  # type: ignore[override]
        return self._df

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._df)

    def peek_timestamp(self) -> datetime | None:  # type: ignore[override]
        if self.index < len(self):
            return self._df.index[self.index]
        return None

    def get_current_bar(self) -> Bar:  # type: ignore[override]
        row = self._df.iloc[self.index]
        ts = self._df.index[self.index]
        return Bar(
            timestamp=ts,
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            symbol=self.symbol,
        )

    def get_lookback_data(self, lookback_period: int) -> pd.DataFrame:  # type: ignore[override]
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1].copy()


# ---------------------------------------------------------------------------
# Synthetic / Benchmark DataSource
# ---------------------------------------------------------------------------


class RandomWalkDataSource(BacktestingDataSource):
    """In-memory OHLCV feed that simulates a geometric random walk.

    This helper is intended for *profiling* or quick experimentation where
    persisting data to disk would introduce unnecessary I/O overhead.  All
    OHLC prices are set equal to the simulated *close* price for simplicity
    and volume is fixed to zero.
    """

    def __init__(
        self,
        symbol: str = "SYN",
        start_price: float = 100.0,
        periods: int = 1_000,
        freq: str = "1min",
        mean: float = 0.0,
        std_dev: float = 0.001,
        seed: int | None = None,
    ) -> None:
        from numpy.random import default_rng

        rng = default_rng(seed)
        # Generate log-returns and construct price path
        log_returns = rng.normal(loc=mean, scale=std_dev, size=periods)
        prices = start_price * np.exp(np.cumsum(log_returns))

        index = pd.date_range("2024-01-01", periods=periods, freq=freq)

        self._df = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": np.zeros(periods, dtype="float64"),
            },
            index=index,
        )

        self.symbol = symbol
        self.index = 0

    # ------------------------------------------------------------------
    # BacktestingDataSource API
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._df)

    def get_raw_data(self) -> pd.DataFrame:  # type: ignore[override]
        return self._df

    def peek_timestamp(self) -> datetime | None:  # type: ignore[override]
        if self.index < len(self):
            return self._df.index[self.index]
        return None

    def get_current_bar(self) -> Bar:  # type: ignore[override]
        row = self._df.iloc[self.index]
        ts = self._df.index[self.index]
        return Bar(
            timestamp=ts,
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            symbol=self.symbol,
        )

    def get_lookback_data(self, lookback_period: int) -> pd.DataFrame:  # type: ignore[override]
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1].copy()
