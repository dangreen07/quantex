import numpy as np

from quantex.datasource import DataSource, PricingData
from quantex.strategy import Indicator, Strategy


class Backtester:
    strategy: type[Strategy]
    data: PricingData

    def __init__(self, strategy: type[Strategy], cash: float = 10_000):
        self.strategy = strategy
        self.data = PricingData()
        self.cash = cash

    def add_data(self, data: DataSource, name: str | None = None):
        self.data.add_data(data, name)

    def run(self):
        for name in self.data.datas.keys():
            self.data.datas[name]._current = 0  ## Reset the current index
        strat = self.strategy(self.data, cash=self.cash)
        strat.init()
        indicators = [
            i for i in strat.__dict__.keys() if isinstance(strat.__dict__[i], Indicator)
        ]
        for indicator in indicators:
            strat.__dict__[indicator].__current__ = 0
        for i in range(len(self.data.index)):
            for name in self.data.datas.keys():
                self.data.datas[
                    name
                ]._current += (
                    1  ## TODO: Handle multiple data sources with different indexes
                )
            skip_nan = False
            for indicator in indicators:
                strat.__dict__[
                    indicator
                ].__current__ += 1  ## Shouldn't have to handle multiple indexers as they should all be the same
                ind: Indicator = strat.__dict__[indicator]
                if ind[-1] == np.nan:
                    skip_nan = True
            if skip_nan:
                continue
            print(len(strat.data.Close))
            print(strat.__dict__[indicators[0]])
            ## TODO: strat.next()
            if i > 15:
                break
