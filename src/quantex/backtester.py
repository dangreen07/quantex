from quantex.datasource import DataSource, PricingData
from quantex.strategy import Indicator, Strategy
import numpy as np
import pandas as pd


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

    @property
    def total_return(self) -> float:
        return (self.equity[-1] - self.starting_cash) / self.starting_cash

    def sharpe_ratio(self, risk_free_rate: float = 0.04):
        total_time = (
            self.run_strategy.data.Timestamp[-1] - self.run_strategy.data.Timestamp[0]
        )
        years = total_time.days / 365.25
        periods_per_year = len(self.equity) / years
        returns = pd.Series(
            self.equity, index=self.run_strategy.data.Timestamp
        ).pct_change()
        risk_free = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        std = returns.std()
        if std == 0:
            return 0
        sharpe = (returns.mean() - risk_free) / std * np.sqrt(periods_per_year)
        return sharpe

    @property
    def annualized_return(self):
        total_time = (
            self.run_strategy.data.Timestamp[-1] - self.run_strategy.data.Timestamp[0]
        )
        years = total_time.days / (365.25)
        return (1 + self.total_return) ** (1 / years) - 1

    @property
    def max_drawdown(self):
        peak = self.equity[0]
        max_drawdown_dollars = 0
        max_drawdown_percent = 0
        for value in self.equity:
            drawdown_dollars = value - peak
            drawdown_percent = (value - peak) / peak
            max_drawdown_dollars = min(
                max_drawdown_dollars,
                drawdown_dollars,
            )
            max_drawdown_percent = min(
                max_drawdown_percent,
                drawdown_percent,
            )
        return abs(max_drawdown_dollars), abs(max_drawdown_percent)


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
            strat.broker.__process_orders__()
            strat.next()
            equity[i] = strat.broker.equity()
        self.result = Result(equity, initial_cash, strat, strat.broker.total_trades)
        return self.result
