from enum import Enum
import numpy as np
from .strategy import Strategy

class CommissionType(Enum):
    PERCENTAGE = 0
    CASH = 1

class SimpleBacktester():
    def __init__(self, 
                 strategy: Strategy,
                cash: float = 10_000, 
                commision: float = 0.002, 
                commision_type: CommissionType = CommissionType.PERCENTAGE
                ):
        self.strategy = strategy
        self.cash = cash
        self.commision = commision
        self.comission_type = commision_type
    def run(self):
        for key in self.strategy.positions.keys():
            self.strategy.positions[key].cash = np.float64(self.cash)
        self.strategy.init()
        ## Simple backtesting loop
        for i in range(1, max([len(i) for i in self.strategy.data.values()])):
            for val in self.strategy.data.values():
                val.current_index = i
            for val in self.strategy.positions.values():
                val._iterate(i)
            for item in self.strategy.indicators:
                item._i = i
            self.strategy.next()
        self.strategy.positions['EURUSD'].close()
