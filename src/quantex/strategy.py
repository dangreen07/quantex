from quantex.broker import Broker
from quantex.datasource import DataSource, PricingData
from abc import abstractmethod
import pandas as pd
import numpy as np


class Indicator:
    def __init__(self, data: np.ndarray):
        self.__data__ = data
        self.__current__ = 1

    def __getitem__(self, key):
        return self.__data__[: self.__current__][key]

    def __str__(self):
        return str(self.__data__[: self.__current__])

    def __len__(self):
        return self.__current__


class Strategy:
    def __init__(self, context: PricingData | None = None, cash: float = 10_000):
        self.__context__ = context or PricingData()
        self.broker = Broker(self.__context__, cash)

    @property
    def datas(self):
        return self.__context__.datas

    @property
    def data(self):
        return self.datas[list(self.datas.keys())[0]]

    def add_data(
        self,
        data: "DataSource",
        name: str | None = None,
    ) -> None:
        self.__context__.add_data(data, name)

    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def next(self) -> None:
        pass

    def Indicator(
        self, data: np.ndarray | pd.Series, ffill: bool = False, bfill: bool = False
    ):
        if isinstance(data, pd.Series):
            data = data.reindex(self.__context__.index)
            if ffill:
                data = data.ffill()
            if bfill:
                data = data.bfill()
            data = data.to_numpy()
        return Indicator(data)
