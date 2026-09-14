from quantex.backtester import Backtester, SearchType
from quantex.datasource import DataSource
from quantex.strategy import Strategy
import pandas as pd
import numpy as np
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
    data = pd.read_parquet("tests/data/NVDA.parquet")
    source = DataSource("NVDA", data)
    bt.add_data(source, "NVDA")
    result = bt.run()
    assert result.total_return == pytest.approx(0.6971, rel=1e-2)
    assert result.total_trades == 102
    assert result.sharpe_ratio() == pytest.approx(0.47, rel=1e-2)
    assert result.annualized_return == pytest.approx(0.0923, rel=1e-2)
    assert result.max_drawdown == pytest.approx((523.98, 0.0524), rel=1e-2)


def test_optimize_grid():
    bt = Backtester(SMACrossover)
    data = pd.read_parquet("tests/data/NVDA.parquet")
    source = DataSource("NVDA", data)
    bt.add_data(source, "NVDA")
    max_sharpe, best_trial = bt.optimize(
        {
            "fast_period": [1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
            "slow_period": [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100],
            "trade_size": np.arange(0.05, 1, 0.05).tolist(),
        },
        constraint=lambda params: params["fast_period"] < params["slow_period"],
        max_trials=500,
    )
    assert max_sharpe == pytest.approx(1.06, rel=1e-2)
    assert best_trial["fast_period"] == 1
    assert best_trial["slow_period"] == 25
    assert best_trial["trade_size"] == pytest.approx(0.5)


def test_optimize_optuna():
    bt = Backtester(SMACrossover)
    data = pd.read_parquet("tests/data/NVDA.parquet")
    source = DataSource("NVDA", data)
    bt.add_data(source, "NVDA")
    max_sharpe, best_trial = bt.optimize(
        {
            "fast_period": [1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
            "slow_period": [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100],
            "trade_size": np.arange(0.05, 1, 0.05).tolist(),
        },
        constraint=lambda params: params["fast_period"] < params["slow_period"],
        max_trials=500,
        search_type=SearchType.OPTUNA,
    )
    assert max_sharpe == pytest.approx(1.21, rel=1e-2)
    assert best_trial["fast_period"] == 20
    assert best_trial["slow_period"] == 100
    assert best_trial["trade_size"] == pytest.approx(0.95)


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
