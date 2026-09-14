from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf
import numpy as np


class DataSource:
    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, name: str, df: pd.DataFrame | None = None):
        """
        DataSource is a class that provides access to data.

        Parameters:
            name: The name of the data source.
            df: The data frame to use as the data source. If None, an empty data source will be created.
        """
        self.name = name
        if df is None:
            return
        self.__init_df__(df)

    def __init_df__(self, df: pd.DataFrame):
        self.__data: dict[str, np.ndarray] = {}
        self.__data["timestamp"] = df.index.to_numpy(copy=True)
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                print(f"Column {col} not found in DataFrame")
                break
            else:
                self.__data[col] = df[col].to_numpy(copy=True)
        self._current = self.__data["timestamp"].shape[0]

    @property
    def Timestamp(self):
        """
        The timestamps of the data.
        """
        return self.__data["timestamp"][: self._current]

    @property
    def Open(self):
        """
        The open prices of the data.
        """
        return self.__data["Open"][: self._current]

    @property
    def High(self):
        """
        The high prices of the data.
        """
        return self.__data["High"][: self._current]

    @property
    def Low(self):
        """
        The low prices of the data.
        """
        return self.__data["Low"][: self._current]

    @property
    def Close(self):
        """
        The close prices of the data.
        """
        return self.__data["Close"][: self._current]

    @property
    def Volume(self):
        """
        The volume of the data.
        """
        return self.__data["Volume"][: self._current]

    def __len__(self):
        return self._current


class YahooDataSource(DataSource):
    def __init__(
        self,
        ticker: str,
        period: str = "1y",
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
    ):
        """
        YahooDataSource is a class that provides access to Yahoo Finance data.

        Parameters:
            ticker: The ticker symbol of the stock to retrieve data for.
            period: The period of the data to retrieve. Can be "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max".
            start: The start date of the data to retrieve. If None, the start date will be the first date available.
            end: The end date of the data to retrieve. If None, the end date will be the last date available.
            interval: The interval of the data to retrieve. Can be "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max".
        """
        super().__init__(name=ticker)
        if start is not None:
            data = yf.Ticker(ticker).history(start=start, end=end, interval=interval)
            data = data[self.REQUIRED_COLUMNS]
            data["timestamp"] = pd.to_datetime(data.index, utc=True)
            data.set_index("timestamp", inplace=True)
            self.__init_df__(data)
        elif period is not None:
            data = yf.Ticker(ticker).history(period=period, interval=interval)
            data = data[["Open", "High", "Low", "Close", "Volume"]]
            data["timestamp"] = pd.to_datetime(data.index, utc=True)
            data.set_index("timestamp", inplace=True)
            self.__init_df__(data)


@dataclass
class PricingData:
    index: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    datas: dict[str, DataSource] = field(default_factory=dict)

    @property
    def data(self) -> "DataSource":
        if not self.datas:
            raise RuntimeError("No data available")
        return self.datas[list(self.datas.keys())[0]]

    def add_data(
        self,
        data: "DataSource",
        name: str | None = None,
    ) -> None:
        if name is None:
            name = data.name

        self.datas[name] = data
        self.index = self.index.union(data.Timestamp)  # type: ignore
