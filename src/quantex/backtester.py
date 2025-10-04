from dataclasses import dataclass
import numpy as np
import pandas as pd

from .broker import Order
from .strategy import Strategy
from .enums import CommissionType
import copy

def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    return float(abs(max_dd))  # return as positive percentage

def _infer_periods_per_year(index: pd.Index, default: int = 252 * 24 * 60) -> int:
    # Simple inference; falls back to minute trading year if uncertain
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return default
    dt = np.diff(index.values).astype("timedelta64[s]").astype(float)
    if not np.isfinite(dt).any():
        return default
    med_sec = np.median(dt[dt > 0])
    if not np.isfinite(med_sec) or med_sec <= 0:
        return default
    periods_per_day = 86400.0 / med_sec
    # Assume 252 trading days/year
    return int(round(252 * periods_per_day))

@dataclass
class BacktestReport:
    starting_cash: np.float64
    final_cash: np.float64
    PnlRecord: pd.Series
    orders: list[Order]

    def __str__(self) -> str:
        equity = self.PnlRecord.astype(float)
        returns = equity.pct_change().dropna()

        # Infer frequency automatically (fallback to minute-level)
        periods_per_year = _infer_periods_per_year(equity.index, 252 * 24 * 60)

        # Risk-free per period from an annual rate
        annual_rf = 0.04
        rf_per_period = annual_rf / periods_per_year

        if len(returns) < 2 or returns.std(ddof=1) == 0:
            sharpe = np.nan
            lo = np.nan
            hi = np.nan
        else:
            excess = returns - rf_per_period
            mean = excess.mean()
            vol = excess.std(ddof=1)
            sharpe = (mean / vol) * np.sqrt(periods_per_year)

            # Standard error of Sharpe (i.i.d. normal approx)
            n = len(excess)
            se = np.sqrt((1 + 0.5 * sharpe**2) / n)
            z = 1.96  # 95% CI
            lo = sharpe - z * se
            hi = sharpe + z * se

        # Max drawdown on equity curve
        running_max = equity.cummax()
        drawdown = ((equity - running_max) / running_max).min()
        mdd = float(abs(drawdown))

        tot_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        tot_orders = len(self.orders)

        return (
            f"Starting Cash: ${self.starting_cash:,.2f}\n"
            f"Final Cash: ${self.final_cash:,.2f}\n"
            f"Total Return: {tot_return:.2%}\n"
            f"Sharpe Ratio: {sharpe:.2f}" if np.isfinite(sharpe) else
            f"Sharpe Ratio: nan"
        ) + (
            f"\nSharpe Confidence Interval: {lo:.4f} - {hi:.4f}"
            if np.isfinite(sharpe) else "\nSharpe Confidence Interval: nan - nan"
        ) + (
            f"\nMax Drawdown: {mdd:.2%}\n"
            f"Total Trades: {tot_orders:,}"
        )

class SimpleBacktester():
    def __init__(self, 
                 strategy: Strategy,
                cash: float = 10_000, 
                commision: float = 0.002, 
                commision_type: CommissionType = CommissionType.PERCENTAGE,
                lot_size: int = 1
                ):
        self.strategy = copy.deepcopy(strategy)
        self.cash = cash
        self.commision = commision
        self.commision_type = commision_type
        self.lot_size = lot_size
        source = self.strategy.positions[list(self.strategy.positions.keys())[0]].source
        self.PnLRecord = pd.Series([0] * len(source.data['Close']), index=source.data['Close'].index, dtype=np.float64)
    def run(self):
        for key in self.strategy.positions.keys():
            self.strategy.positions[key].cash = np.float64(self.cash)
            self.strategy.positions[key].lot_size = self.lot_size
            self.strategy.positions[key].commision = np.float64(self.commision)
            self.strategy.positions[key].commision_type = self.commision_type

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
