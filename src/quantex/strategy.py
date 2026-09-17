from quantex.datasource import DataSource, PricingData
from quantex.commission import Commission
from quantex.broker import Broker
from abc import abstractmethod
import pandas as pd
import numpy as np

from quantex.margin import Margin


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
    def __init__(
        self,
        commission: Commission | None = None,
        context: PricingData | None = None,
        cash: float = 10_000,
        margin: Margin | None = None,
        multiplier: float = 1,
    ):
        """
        Strategy is the base class for all strategies.

        Parameters:
            commission: The commission object to use to calculate commissions.
            context: The pricing data to use. If None, a new pricing data object will be created.
            cash: The starting cash to use. If None, the starting cash will be 10,000.
            margin: The margin object to use. If None, a new margin object will be created.
            multiplier: The multiplier to the orders in the backtest.
        """
        self.__context__ = context or PricingData()
        if commission is None:
            commission = Commission()
        if margin is None:
            margin = Margin()
        self.broker = Broker(
            self.__context__, commission, margin, cash=cash, multiplier=multiplier
        )

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

    @property
    def index(self) -> pd.DatetimeIndex:
        """
        The index of the data used by the strategy.
        """
        return self.__context__.index

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
