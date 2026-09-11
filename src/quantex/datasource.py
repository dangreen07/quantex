from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf
import numpy as np


class DataSource:
    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
    __data: dict[str, np.ndarray]

    def __init__(self, name: str, df: pd.DataFrame | None = None):
        self.name = name
        if df is None:
            return
        self._init_df(df)

    def _init_df(self, df: pd.DataFrame):
        self.__data = {}
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
        return self.__data["timestamp"][: self._current]

    @property
    def Open(self):
        return self.__data["Open"][: self._current]

    @property
    def High(self):
        return self.__data["High"][: self._current]

    @property
    def Low(self):
        return self.__data["Low"][: self._current]

    @property
    def Close(self):
        return self.__data["Close"][: self._current]

    @property
    def Volume(self):
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
        super().__init__(name=ticker)
        if start is not None:
            data = yf.Ticker(ticker).history(start=start, end=end, interval=interval)
            data = data[self.REQUIRED_COLUMNS]
            data["timestamp"] = pd.to_datetime(data.index, utc=True)
            data.set_index("timestamp", inplace=True)
            self._init_df(data)
        elif period is not None:
            data = yf.Ticker(ticker).history(period=period, interval=interval)
            data = data[["Open", "High", "Low", "Close", "Volume"]]
            data["timestamp"] = pd.to_datetime(data.index, utc=True)
            data.set_index("timestamp", inplace=True)
            self._init_df(data)


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
            name = str(len(self.datas))

        self.datas[name] = data
        self.index = self.index.union(data.Timestamp)  # type: ignore
