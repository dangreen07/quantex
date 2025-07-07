"""DataSource abstractions used by the QuantEx library.

`DataSource` is the most generic interface; concrete back-testing or live
implementations inherit from it.  `BacktestingDataSource` simply adds the
`__len__` protocol so that the engine can iterate through historical data.
"""

import pandas as pd


class DataSource:
    """A data source class that will be inherited by all data sources."""

    index: int = 0

    def get_current_row(self) -> tuple | pd.Series:
        """Get the current row of data from the data source."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_lookback_data(self, lookback_period: int) -> pd.DataFrame:
        """Get all data before and including the current row from the data source."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _increment_index(self) -> None:
        """Increment the index of the data source."""
        self.index += 1


class BacktestingDataSource(DataSource):
    """DataSource that supports the Python `len()` protocol for backtests."""

    def __len__(self) -> int:
        """Return the number of data points available in the source."""
        raise NotImplementedError("Subclasses must implement this method.")
