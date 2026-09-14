from quantex.backtester import Backtester, SearchType
from quantex.datasource import DataSource
from quantex.strategy import Strategy
import pandas as pd
import numpy as np
import pytest
import talib


class SMACrossover(Strategy):
    fast_period = 5
    slow_period = 10
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
                stop_loss=0.9 * self.data.Close[-1],  # type: ignore
                take_profit=1.2 * self.data.Close[-1],  # type: ignore
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
                stop_loss=1.1 * self.data.Close[-1],  # type: ignore
                take_profit=0.8 * self.data.Close[-1],  # type: ignore
            )


class BuyAndHold(Strategy):
    def init(self):
        pass

    def next(self):
        if self.broker.is_closed():
            self.broker.buy(amount=50)


class PartialFill(Strategy):
    def init(self):
        self.signal = self.Indicator(np.array([1, 0, 0, 0, 0, -0.5, 0, 0, 0, 0]))

    def next(self):
        sig = self.signal[-1]
        if sig > 0:
            self.broker.buy(amount=abs(sig) * 2)  # type: ignore
        elif sig < 0:
            self.broker.sell(amount=abs(sig) * 2)  # type: ignore


class StopLossTest(Strategy):
    def init(self):
        pass

    def next(self):
        if self.broker.is_closed() and len(self.broker.processedOrders["TEST"]) == 0:
            self.broker.sell(amount=1, stop_loss=15)


class TakeProfitTest(Strategy):
    def init(self):
        pass

    def next(self):
        if self.broker.is_closed() and len(self.broker.processedOrders["TEST"]) == 0:
            self.broker.buy(amount=1, take_profit=15)


class StopAndCloseSameBar(Strategy):
    def init(self):
        pass

    def next(self):
        if self.broker.is_closed() and len(self.broker.processedOrders["TEST"]) == 0:
            self.broker.buy(amount=1, stop_loss=8)
        elif self.broker.is_long():
            self.broker.close()


class MultipleStopLoss(Strategy):
    def init(self):
        data = np.full(30, 0)
        data[0] = 1
        data[4] = -1
        self.signal = self.Indicator(data)

    def next(self):
        sig = self.signal[-1]
        if sig > 0:
            self.broker.buy(amount=5, stop_loss=4)
        elif sig < 0:
            self.broker.close()
            self.broker.sell(amount=5, stop_loss=15)


def test_backtester():
    bt = Backtester(SMACrossover)
    data = pd.read_parquet("tests/data/NVDA.parquet")
    source = DataSource("NVDA", data)
    bt.add_data(source, "NVDA")
    result = bt.run()
    print(f"Total return: {result.total_return:.2%}")
    print(f"Total trades: {result.total_trades}")
    print(f"Sharpe ratio: {result.sharpe_ratio():.2f}")
    print(f"Annualized return: {result.annualized_return:.2%}")
    print(f"Max drawdown: {result.max_drawdown}")
    assert result.total_return == pytest.approx(0.6169, rel=1e-2)
    assert result.total_trades == 157
    assert result.sharpe_ratio() == pytest.approx(0.4155, rel=1e-2)
    assert result.annualized_return == pytest.approx(0.0835, rel=1e-2)
    assert result.max_drawdown == pytest.approx((233.0648, 0.023306), rel=1e-2)


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
    assert max_sharpe == pytest.approx(0.4884, rel=1e-2)
    assert best_trial["fast_period"] == 3
    assert best_trial["slow_period"] == 15
    assert best_trial["trade_size"] == pytest.approx(0.95)


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
    assert max_sharpe == pytest.approx(0.6695, rel=1e-2)
    assert best_trial["fast_period"] == 5
    assert best_trial["slow_period"] == 10
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
    expected_return = 50 * (30 - 15) / 10_000
    result = bt.run()
    assert result.total_return == pytest.approx(expected_return)


def test_partial_fill():
    data = np.arange(10, 20, 1).tolist()
    data = np.array([data] * len(DataSource.REQUIRED_COLUMNS)).T
    idx = pd.date_range("2022-01-01", "2022-01-10", freq="D")
    df = pd.DataFrame(data, index=idx, columns=DataSource.REQUIRED_COLUMNS)
    source = DataSource("TEST", df)
    expected_return = (1 + 10 / 10_000) * (1 + 3 / 10_000) - 1
    bt = Backtester(PartialFill)
    bt.add_data(source)
    result = bt.run()
    assert result.total_return == pytest.approx(expected_return, rel=1e-2)


def test_single_order_stop_loss():
    data = np.arange(10, 20, 1).tolist()
    data = np.array([data] * len(DataSource.REQUIRED_COLUMNS)).T
    idx = pd.date_range("2022-01-01", "2022-01-10", freq="D")
    df = pd.DataFrame(data, index=idx, columns=DataSource.REQUIRED_COLUMNS)
    source = DataSource("TEST", df)
    expected_return = -4 / 10_000
    bt = Backtester(StopLossTest)
    bt.add_data(source)
    result = bt.run()
    assert result.total_return == pytest.approx(expected_return)


def test_single_order_take_profit():
    data = np.arange(10, 20, 1).tolist()
    data = np.array([data] * len(DataSource.REQUIRED_COLUMNS)).T
    idx = pd.date_range("2022-01-01", "2022-01-10", freq="D")
    df = pd.DataFrame(data, index=idx, columns=DataSource.REQUIRED_COLUMNS)
    source = DataSource("TEST", df)
    expected_return = 4 / 10_000
    bt = Backtester(TakeProfitTest)
    bt.add_data(source)
    result = bt.run()
    assert result.total_return == pytest.approx(expected_return)


def test_stop_loss_close_race_condition():
    data = np.array([10, 10, 8, 6, 5, 10])
    data = np.array([data] * len(DataSource.REQUIRED_COLUMNS)).T
    idx = pd.date_range("2022-01-01", periods=len(data), freq="D")
    df = pd.DataFrame(data, index=idx, columns=DataSource.REQUIRED_COLUMNS)
    source = DataSource("TEST", df)
    bt = Backtester(StopAndCloseSameBar)
    bt.add_data(source)

    result = bt.run()

    assert result.run_strategy.broker.is_closed()


def test_multiple_stop_loss():
    arr1 = np.arange(10, 5, -1)
    arr2 = np.arange(5, 8, 1)
    arr3 = np.arange(8, 3, -1)
    arr4 = np.arange(3, 20, 1)
    arr = np.concatenate((arr1, arr2, arr3, arr4))
    data = np.array([arr] * len(DataSource.REQUIRED_COLUMNS)).T
    idx = pd.date_range("2022-01-01", "2022-01-30", freq="D")
    df = pd.DataFrame(data, index=idx, columns=DataSource.REQUIRED_COLUMNS)
    source = DataSource("TEST", df)
    expected_return = (1 + 5 * (arr[4] - arr[1]) / 10_000) * (
        1 + 5 * (arr[5] - arr[26]) / 10_000
    ) - 1
    bt = Backtester(MultipleStopLoss)
    bt.add_data(source)
    result = bt.run()
    assert result.total_return == pytest.approx(expected_return, rel=1e-2)
