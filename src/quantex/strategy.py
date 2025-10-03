from abc import ABC, abstractmethod
from typing import final
import numpy as np
from .broker import Broker
from .datasource import DataSource
from .helpers import TimeNDArray


class Strategy(ABC):
    def __init__(self):
        self.positions: dict[str, Broker] = {}
        self.data: dict[str, DataSource] = {}
        self.indicators: list[TimeNDArray] = []
    
    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @final
    def add_data(self, source: DataSource, symbol: str):
        self.data[symbol] = source
        self.positions[symbol] = Broker(source)
    
    @final
    def Indicator(self, arr: np.typing.NDArray):
        data = TimeNDArray.from_array(arr)
        self.indicators.append(data)
        return data