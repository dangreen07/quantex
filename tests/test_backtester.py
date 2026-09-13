from quantex.backtester import Backtester
from quantex.datasource import DataSource, YahooDataSource
from quantex.strategy import Strategy
import pandas as pd
import pytest
import talib


class SMACrossover(Strategy):
    fast_period = 5
    slow_period = 20
    trade_size = 0.25

    def init(self):
        self.fast_ma = self.Indicator(
            talib.SMA(self.data.Close, timeperiod=self.fast_period)
        )
        self.slow_ma = self.Indicator(
            talib.SMA(self.data.Close, timeperiod=self.slow_period)
        )

    def next(self):
        if (
            self.fast_ma[-1] > self.slow_ma[-1]
            and self.fast_ma[-2] < self.slow_ma[-2]
            and (self.broker.is_closed() or self.broker.is_short())
        ):
            self.broker.close()
            amount = self.broker.equity() * self.trade_size / self.data.Close[-1]
            self.broker.buy(
                amount=amount,
                stop_loss=0.975 * self.data.Close[-1],
                take_profit=1.1 * self.data.Close[-1],
            )
        elif (
            self.fast_ma[-1] < self.slow_ma[-1]
            and self.fast_ma[-2] > self.slow_ma[-2]
            and (self.broker.is_closed() or self.broker.is_long())
        ):
            self.broker.close()
            amount = self.broker.equity() * self.trade_size / self.data.Close[-1]
            self.broker.sell(
                amount=amount,
                stop_loss=1.025 * self.data.Close[-1],
                take_profit=0.9 * self.data.Close[-1],
            )


class BuyAndHold(Strategy):
    def init(self):
        pass

    def next(self):
        if self.broker.is_closed():
            self.broker.buy(amount=50)


def test_backtester():
    bt = Backtester(SMACrossover)
    source = YahooDataSource("NVDA", start="2020-01-01", end="2025-12-31")
    bt.add_data(source, "NVDA")
    result = bt.run()
    assert result.total_return == pytest.approx(0.6971, rel=1e-2)
    assert result.total_trades == 102
    assert result.sharpe_ratio() == pytest.approx(0.47, rel=1e-2)
    assert result.annualized_return == pytest.approx(0.0923, rel=1e-2)
    assert result.max_drawdown == pytest.approx((523.98, 0.0524), rel=1e-2)


def test_basic_backtester():
    bt = Backtester(BuyAndHold)
    source = DataSource(
        "STOCK",
        pd.DataFrame(
            data={
                "Open": [10, 15, 20, 25],
                "High": [15, 20, 25, 30],
                "Low": [5, 10, 15, 20],
                "Close": [15, 20, 25, 30],
                "Volume": [100, 100, 100, 100],
            }
        ),
    )
    bt.add_data(source)
    expected_return = 50 * (30 - 20) / 10_000
    result = bt.run()
    assert result.total_return == pytest.approx(expected_return)
