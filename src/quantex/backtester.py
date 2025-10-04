from dataclasses import dataclass
import numpy as np
import pandas as pd

from quantex.src.quantex.broker import Order
from .strategy import Strategy
from .enums import CommissionType
import copy

def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    return float(abs(max_dd))  # return as positive percentage

@dataclass
class BacktestReport:
    starting_cash: np.float64
    final_cash: np.float64
    PnlRecord: pd.Series
    orders: list[Order]

    def __str__(self) -> str:
        returns = self.PnlRecord.pct_change().dropna()
        mean_return = returns.mean()
        volatility = returns.std()
        periods_per_year = 252 * 24 * 60
        risk_free_rate = 0.04 / periods_per_year  # daily Rf if 4% annual
        sharpe = ((mean_return - risk_free_rate) / volatility) * np.sqrt(periods_per_year)
        sharpe_error = np.sqrt((1 + sharpe) / len(self.PnlRecord))
        ## 95% confidence interval
        sharpe_lower = sharpe - 1.96 * sharpe_error
        sharpe_upper = sharpe + 1.96 * sharpe_error
        drawdown = max_drawdown(self.PnlRecord)
        tot_return = self.final_cash / self.starting_cash - 1
        tot_orders = len(self.orders)
        output = f'''Starting Cash: ${self.starting_cash:,.2f}
Final Cash: ${self.final_cash:,.2f}
Total Return: {tot_return:.2%}
Sharpe Ratio: {sharpe:.2f}
Sharpe Confidence Interval: {sharpe_lower:.4f} - {sharpe_upper:.4f}
Max Drawdown: {drawdown:.2%}
Total Trades: {tot_orders:,}'''
        return output

class SimpleBacktester():
    def __init__(self, 
                 strategy: Strategy,
                cash: float = 10_000, 
                commision: float = 0.002, 
                commision_type: CommissionType = CommissionType.PERCENTAGE
                ):
        self.strategy = copy.deepcopy(strategy)
        self.cash = cash
        self.commision = commision
        self.comission_type = commision_type
        source = self.strategy.positions[list(self.strategy.positions.keys())[0]].source
        self.PnLRecord = pd.Series([0] * len(source.data['Close']), index=source.data['Close'].index, dtype=np.float64)
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
        orders: list[Order] = []
        for val in self.strategy.positions.values():
            val.close()
            self.PnLRecord = self.PnLRecord.add(val.PnLRecord) # type: ignore
            orders.extend(val.complete_orders)
        
        return BacktestReport(
            starting_cash=np.float64(self.cash), 
            final_cash=self.PnLRecord.iloc[-1], 
            PnlRecord=self.PnLRecord,
            orders=orders)
