from quantex.backtester import Backtester, SearchType
from quantex.broker import Order, OrderDirection
from quantex.datasource import DataSource
import statsmodels.api as sm
from quantex.strategy import Strategy
from typing import Iterable, cast
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
    amount = 50.0

    def init(self):
        pass

    def next(self):
        if self.broker.is_closed():
            self.broker.buy(amount=self.amount)


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


class MultipleDataSources(Strategy):
    """
    Simple cointegration strategy with fixed hedge ratio and constant.
    """

    window = 60  ## 2 months
    hedge_ratio = 1
    hedge_constant = 0.5
    entry_threshold = 2  ## Enter when the |z-score| is above 2
    exit_threshold = 0.5  ## Exit when the |z-score| is below 0.5

    def init(self):
        self.log1 = np.log(self.datas["NVDA"].Close)
        self.log2 = np.log(self.datas["MSFT"].Close)
        residuals = pd.Series(
            self.log2 - (self.hedge_constant + self.hedge_ratio * self.log1),
            index=self.index,
        )
        res_mean = residuals.rolling(self.window).mean()
        res_std = residuals.rolling(self.window).std()
        z_score = (residuals - res_mean) / res_std
        self.z_score = self.Indicator(z_score)

    def next(self):
        if abs(self.z_score[-1]) > self.entry_threshold and self.broker.is_closed():
            amount = 50
            nvda_amount = (amount * self.datas["MSFT"].Close[-1]) / self.datas[
                "NVDA"
            ].Close[-1]
            if self.z_score[-1] > 0:
                self.broker.sell("MSFT", amount=amount)
                self.broker.buy("NVDA", amount=nvda_amount)
            elif self.z_score[-1] < 0:
                self.broker.buy("MSFT", amount=amount)
                self.broker.sell("NVDA", amount=nvda_amount)
        elif abs(self.z_score[-1]) < self.exit_threshold:
            self.broker.close()


def test_backtester():
    bt = Backtester(SMACrossover)
    data = pd.read_parquet("tests/data/NVDA.parquet")
    source = DataSource("NVDA", data)
    bt.add_data(source, "NVDA")
    result = bt.run()
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
            },
            index=pd.date_range("2022-01-01", periods=4, freq="D"),
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


def calculate_total_pnl(
    orders: Iterable[Order],
    prices: pd.Series | pd.DataFrame,
) -> float:
    """
    Return total mark-to-market P&L as a cash amount.

    Buys reduce cash and increase the position. Sells increase cash and
    reduce the position. Any remaining position is valued at the final
    available market price.

    `Order.price` is used as the fill price when present. Otherwise, the
    latest price at or before `fill_timestamp` is used.
    """
    if prices.empty:
        raise ValueError("prices must not be empty")

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().dropna()

    if isinstance(prices, pd.DataFrame):
        execution_prices = prices["Open"]
        closing_prices = prices["Close"]
    else:
        execution_prices = prices
        closing_prices = prices

    if prices.index.has_duplicates:
        prices = prices[~prices.index.duplicated(keep="last")]

    cash = 0.0
    position = 0.0

    sorted_orders = sorted(
        orders,
        key=lambda order: pd.Timestamp(order.fill_timestamp),
    )

    for order in sorted_orders:
        # Broker mutates a fully closed entry order to amount_filled == 0.
        filled_amount = float(order.amount_filled or order.amount)

        if filled_amount < 0:
            raise ValueError(f"Order {order.id} has a negative amount_filled")

        fill_timestamp = pd.Timestamp(order.fill_timestamp)

        if order.price is not None:
            execution_price = float(order.price)
        else:
            execution_price = cast(float, execution_prices.asof(fill_timestamp))

            if pd.isna(execution_price):
                raise ValueError(
                    f"No market price available for order {order.id} at "
                    f"{fill_timestamp}"
                )

        if execution_price <= 0:
            raise ValueError(f"Order {order.id} has a non-positive execution price")

        notional = filled_amount * execution_price

        if order.direction == OrderDirection.BUY:
            position += filled_amount
            cash -= notional

        elif order.direction == OrderDirection.SELL:
            position -= filled_amount
            cash += notional

        else:
            raise ValueError(f"Unsupported order direction: {order.direction!r}")

    final_price = float(closing_prices.iloc[-1])
    market_value = position * final_price

    return cash + market_value


def test_multiple_data_sources():
    data1 = pd.read_parquet("tests/data/NVDA.parquet")
    data2 = pd.read_parquet("tests/data/MSFT-2.parquet")
    x = np.log(data1["Close"][data1.index < "2020-12-31"])  ## training data
    y = np.log(data2["Close"][data2.index < "2020-12-31"])  ## training data
    data1 = data1[data1.index >= "2020-12-31"]  ## strategy data
    data2 = data2[data2.index >= "2020-12-31"]  ## strategy data
    ## Calculate the hedge ratio
    x = sm.add_constant(x)
    result = sm.OLS(y, x).fit()
    hedge_constant = result.params.iloc[0]
    hedge_ratio = result.params.iloc[1]
    source1 = DataSource("NVDA", data1)
    source2 = DataSource("MSFT", data2)
    bt = Backtester(MultipleDataSources)
    bt.add_data(source1, "NVDA")
    bt.add_data(source2, "MSFT")
    result = bt.run({"hedge_ratio": hedge_ratio, "hedge_constant": hedge_constant})
    assert result.total_return == pytest.approx(2.0427, rel=1e-2)
    assert result.total_trades == 52
    assert result.sharpe_ratio() == pytest.approx(0.7165, rel=1e-2)
    assert result.annualized_return == pytest.approx(0.249450, rel=1e-2)
    assert result.max_drawdown == pytest.approx((2008.65425, 0.200865), rel=1e-2)
    total_return = 0.0
    for name in ("NVDA", "MSFT"):
        broker = result.run_strategy.broker
        orders = broker.processedOrders[name] + broker.openPositions[name]
        data = result.run_strategy.datas[name]
        prices = pd.DataFrame(
            {"Open": data.Open, "Close": data.Close},
            index=result.run_strategy.index,
        )
        total_return += calculate_total_pnl(orders, prices)
    total_return /= 10_000
    assert total_return == pytest.approx(result.total_return, rel=1e-2)


class MultiplierTrading(Strategy):
    def init(self):
        signal = np.full(len(self.data.Close), 0)
        signal[0] = 1
        signal[9] = -1
        self.signal = self.Indicator(signal)

    def next(self):
        if self.signal[-1] == 1:
            self.broker.buy(amount=1)
        elif self.signal[-1] == -1:
            self.broker.sell(amount=1)


def test_multiplier_trading():
    data = np.arange(10, 30, 1)
    data = np.array([data] * len(DataSource.REQUIRED_COLUMNS)).T
    idx = pd.date_range("2022-01-01", periods=len(data), freq="D")
    df = pd.DataFrame(data, index=idx, columns=DataSource.REQUIRED_COLUMNS)
    source = DataSource("STOCK", df)
    price_change = source.Close[10] - source.Close[1]
    multiplier = 50
    expected_return = (price_change * multiplier) / 10_000
    bt = Backtester(MultiplierTrading, multiplier=multiplier)
    bt.add_data(source)
    result = bt.run()
    print(result.total_return, expected_return)


def test_trade_more_than_account():
    data = np.arange(10, 20, 1).tolist()
    data = np.array([data] * len(DataSource.REQUIRED_COLUMNS)).T
    idx = pd.date_range("2022-01-01", periods=len(data), freq="D")
    df = pd.DataFrame(data, index=idx, columns=DataSource.REQUIRED_COLUMNS)
    source = DataSource("TEST", df)
    bt = Backtester(BuyAndHold)
    bt.add_data(source)
    result = bt.run(params={"amount": 2_000})
    assert result.total_return == pytest.approx(0.0)
    assert result.total_trades == 0
