"""DataSource abstractions for the QuantEx library."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from quantex.models import Bar
from abc import ABC, abstractmethod


class DataSource(ABC):  # pylint: disable=too-few-public-methods
    """Abstract data source for providing market data.

    Implementations must provide the *current* bar via `get_current_bar`
    and allow a rolling historical window via `get_lookback_data`.
    The internal pointer `index` starts at 0 and should be advanced by calling
    `_increment_index` once the engine has finished processing a bar.
    """

    index: int = 0

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

    def _increment_index(self) -> None:
        """Advances the internal pointer to the next bar."""
        self.index += 1


class BacktestingDataSource(DataSource):
    """A data source for backtesting that must have a defined length."""

    @abstractmethod
    def __len__(self) -> int:  # pragma: no cover – abstract contract
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

        df = pd.read_csv(self.path, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        df = df.sort_index()
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            missing = required_cols.difference(df.columns)
            raise ValueError(f"CSV missing required columns: {missing}")

        self._df = df  # immutable reference
        self.symbol = symbol or self.path.stem
        self.index = 0

    def __len__(self) -> int:
        """Returns the number of bars in the data source."""
        return len(self._df)

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
