from quantex.broker import Broker
from quantex.datasource import DataSource, PricingData
from abc import abstractmethod
import pandas as pd
import numpy as np


class Indicator:
    def __init__(self, data: np.ndarray):
        self.__data = data
        self._current = 1

    def __getitem__(self, key):
        return self.__data[: self._current][key]

    def __str__(self):
        return str(self.__data[: self._current])

    def __len__(self):
        return self._current


class Strategy:
    def __init__(self, context: PricingData | None = None, cash: float = 10_000):
        """
        Strategy is the base class for all strategies.

        Parameters:
            context: The pricing data to use. If None, a new pricing data object will be created.
            cash: The starting cash to use. If None, the starting cash will be 10,000.
        """
        self.__context__ = context or PricingData()
        self.broker = Broker(self.__context__, cash)

    @property
    def datas(self) -> dict[str, DataSource]:
        """
        The data sources used by the strategy.
        """
        return self.__context__.datas

    @property
    def data(self) -> DataSource:
        """
        The first data source used by the strategy.
        """
        return self.datas[list(self.datas.keys())[0]]

    def add_data(
        self,
        data: "DataSource",
        name: str | None = None,
    ) -> None:
        """
        Adds a data source to the strategy.

        Parameters:
            data: The data source to add.
            name: The name of the data source. If None, the name will be the same as the data source.
        """
        self.__context__.add_data(data, name)

    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def next(self) -> None:
        pass

    def Indicator(
        self, data: np.ndarray | pd.Series, ffill: bool = False, bfill: bool = False
    ) -> Indicator:
        """
        Creates an indicator for the strategy.

        Parameters:
            data: The data to create the indicator from.
            ffill: Whether to forward fill the data.
            bfill: Whether to backward fill the data.

        Returns:
            The indicator.
        """
        if isinstance(data, pd.Series):
            data = data.reindex(self.__context__.index)
            if ffill:
                data = data.ffill()
            if bfill:
                data = data.bfill()
            data = data.to_numpy()
        return Indicator(data)
