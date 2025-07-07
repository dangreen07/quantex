"""DataSource abstractions used by the QuantEx library.

`DataSource` is the most generic interface; concrete back-testing or live
implementations inherit from it.  `BacktestingDataSource` simply adds the
`__len__` protocol so that the engine can iterate through historical data.
"""

import pandas as pd


class DataSource:
    """A data source class that will be inherited by all data sources."""

    def get_row(self, idx: int) -> tuple | pd.Series:
        """Get a row of data from the data source."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_data_before_idx(self, idx: int, lookback_period: int) -> pd.DataFrame:
        """Get all data before and including the given index from the data source."""
        raise NotImplementedError("Subclasses must implement this method.")


class BacktestingDataSource(DataSource):
    """DataSource that supports the Python `len()` protocol for backtests."""

    def __len__(self) -> int:
        """Return the number of data points available in the source."""
        raise NotImplementedError("Subclasses must implement this method.")
