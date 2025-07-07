"""DataSource abstractions used by the QuantEx library.

`DataSource` is the most generic interface; concrete back-testing or live
implementations inherit from it. `BacktestingDataSource` adds `__len__` so the
engine can iterate deterministically through historical data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from quantex.models import Bar


class DataSource:  # pylint: disable=too-few-public-methods
    """Abstract data source.

    Implementations must provide the *current* bar via :meth:`get_current_bar`
    and allow a rolling historical window via :meth:`get_lookback_data`.
    The internal pointer *index* starts at 0 and should be advanced by calling
    :meth:`_increment_index` once the engine has finished processing a bar.
    """

    index: int = 0

    # ---------------------------------------------------------------------
    # Public API expected by the engine / strategy
    # ---------------------------------------------------------------------
    def get_current_bar(self) -> Bar:
        """Return the bar at the current *index* position."""
        raise NotImplementedError

    def get_lookback_data(self, lookback_period: int) -> pd.DataFrame:
        """Return a *lookback_period* window (inclusive) ending at *index*."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers for iteration control
    # ------------------------------------------------------------------
    def _increment_index(self) -> None:
        """Advance the internal pointer to the next bar."""
        self.index += 1


class BacktestingDataSource(DataSource):
    """Concrete back-testing sources must implement :pycode{__len__}."""

    def __len__(self) -> int:  # pragma: no cover – abstract contract
        raise NotImplementedError


class CSVDataSource(BacktestingDataSource):
    """Back-testing data source backed by a local OHLCV CSV file.

    The CSV must contain the standard columns: ``timestamp``, ``open``,
    ``high``, ``low``, ``close``, ``volume``. Additional columns are ignored.
    The *timestamp* column will be parsed to :class:`pandas.DatetimeIndex`.
    """

    def __init__(self, path: str | Path, symbol: Optional[str] = None):
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

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401 – simple method
        return len(self._df)

    def get_current_bar(self) -> Bar:
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

    def get_lookback_data(self, lookback_period: int) -> pd.DataFrame:  # noqa: D401
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1].copy()
