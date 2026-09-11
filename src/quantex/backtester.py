from quantex.datasource import DataSource, PricingData
from quantex.strategy import Indicator, Strategy
import numpy as np


class Result:
    equity: np.ndarray
    starting_cash: float
    run_strategy: Strategy
    total_trades: int

    def __init__(
        self,
        equity: np.ndarray,
        starting_cash: float,
        run_strategy: Strategy,
        total_trades: int = 0,
    ):
        self.equity = equity
        self.starting_cash = starting_cash
        self.run_strategy = run_strategy
        self.total_trades = total_trades

    def total_return(self) -> float:
        return (self.equity[-1] - self.starting_cash) / self.starting_cash


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
        strat = self.strategy(self.data, cash=self.cash)
        strat.init()
        indicators = [
            i for i in strat.__dict__.keys() if isinstance(strat.__dict__[i], Indicator)
        ]
        for name in self.data.datas.keys():
            self.data.datas[name]._current = 0  ## Reset the current index
        for indicator in indicators:
            strat.__dict__[indicator]._current = 0
        initial_cash = strat.broker.cash
        equity = np.full(len(self.data.index), initial_cash, dtype=np.float64)
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
                ]._current += 1  ## Shouldn't have to handle multiple indexers as they should all be the same
                ind: Indicator = strat.__dict__[indicator]
                if np.isnan(ind[-1]):
                    skip_nan = True
            if skip_nan:
                continue
            strat.next()
            strat.broker.__process_orders__()
            equity[i] = strat.broker.equity()
        self.result = Result(equity, initial_cash, strat, strat.broker.total_trades)
        return self.result
