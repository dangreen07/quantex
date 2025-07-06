import pandas as pd

## This is a primitive class for data sources.
class DataSource:
    """A data source class that will be inherited by all data sources."""
    def get_row(self, idx: int) -> tuple | pd.Series:
        """Get a row of data from the data source."""
        raise NotImplementedError("Subclasses must implement this method.")
    def get_data_before_idx(self, idx: int, lookback_period: int) -> pd.DataFrame:
        """Get all data before and including the given index from the data source."""
        raise NotImplementedError("Subclasses must implement this method.")

## This is a primitive class for backtesting data sources.
class BacktestingDataSource(DataSource):
    """A data source for backtesting."""
    def __len__(self) -> int:
        """Get the length of the data source."""
        raise NotImplementedError("Subclasses must implement this method.")