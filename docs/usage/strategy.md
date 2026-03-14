# Strategy Guide

This guide explains how to write a trading strategy in Quantex from the ground up.

If you are new to the library, the most important idea is this: a strategy is just a Python class that decides what to do on each bar of historical market data.

In Quantex, strategy classes inherit from [`Strategy`](../../src/quantex/strategy.py:9).

## What a strategy is responsible for

Your strategy does three things:

1. load or attach data in [`Strategy.init()`](../../src/quantex/strategy.py:52)
2. optionally create indicator arrays with [`Strategy.Indicator()`](../../src/quantex/strategy.py:126)
3. place orders in [`Strategy.next()`](../../src/quantex/strategy.py:71)

The backtester does the rest. It advances time, updates data visibility, asks brokers to process orders, and records the equity curve through [`SimpleBacktester.run()`](../../src/quantex/backtester.py:414).

## The two required methods

Every concrete strategy must implement:

- [`Strategy.init()`](../../src/quantex/strategy.py:52)
- [`Strategy.next()`](../../src/quantex/strategy.py:71)

### [`Strategy.init()`](../../src/quantex/strategy.py:52)

This method runs once before the backtest loop starts.

Use it to:

- attach one or more data sources with [`Strategy.add_data()`](../../src/quantex/strategy.py:98)
- precompute arrays
- create time-aware indicators with [`Strategy.Indicator()`](../../src/quantex/strategy.py:126)
- set internal state variables

### [`Strategy.next()`](../../src/quantex/strategy.py:71)

This method runs once per backtest step.

Use it to:

- read the current market state from [`Strategy.data`](../../src/quantex/strategy.py:48)
- inspect position state from [`Strategy.positions`](../../src/quantex/strategy.py:47)
- place orders with [`Broker.buy()`](../../src/quantex/broker.py:159), [`Broker.sell()`](../../src/quantex/broker.py:235), or [`Broker.close()`](../../src/quantex/broker.py:307)

## First complete example

```python
from quantex import Strategy, CSVDataSource, SimpleBacktester
import pandas as pd


class MovingAverageCross(Strategy):
    def __init__(self, fast_period=5, slow_period=20):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def init(self):
        self.add_data(CSVDataSource("data.csv"), "TEST")

        close = self.data["TEST"].Close
        self.fast_ma = self.Indicator(
            pd.Series(close).rolling(window=self.fast_period).mean().to_numpy()
        )
        self.slow_ma = self.Indicator(
            pd.Series(close).rolling(window=self.slow_period).mean().to_numpy()
        )

    def next(self):
        if len(self.fast_ma) < 2 or len(self.slow_ma) < 2:
            return

        broker = self.positions["TEST"]

        crossed_up = self.fast_ma[-2] <= self.slow_ma[-2] and self.fast_ma[-1] > self.slow_ma[-1]
        crossed_down = self.fast_ma[-2] >= self.slow_ma[-2] and self.fast_ma[-1] < self.slow_ma[-1]

        if crossed_up and broker.is_closed():
            broker.buy(quantity=1.0)
        elif crossed_down and broker.is_long():
            broker.close()


strategy = MovingAverageCross()
report = SimpleBacktester(strategy, cash=10_000).run()
print(report)
```

## Understanding the objects available inside a strategy

### [`self.data`](../../src/quantex/strategy.py:48)

[`self.data`](../../src/quantex/strategy.py:48) is a dictionary whose keys are the symbol names you used with [`Strategy.add_data()`](../../src/quantex/strategy.py:98).

Example:

```python
def init(self):
    self.add_data(CSVDataSource("eurusd.csv"), "EURUSD")
    self.add_data(CSVDataSource("gbpusd.csv"), "GBPUSD")

def next(self):
    eurusd_close = self.data["EURUSD"].CClose
    gbpusd_close = self.data["GBPUSD"].CClose
```

Each value is a [`DataSource`](../../src/quantex/datasource.py:6) object.

### [`self.positions`](../../src/quantex/strategy.py:47)

[`self.positions`](../../src/quantex/strategy.py:47) is another dictionary keyed by the same symbol names.

Each value is a [`Broker`](../../src/quantex/broker.py:113) created automatically when you call [`Strategy.add_data()`](../../src/quantex/strategy.py:98).

Example:

```python
def next(self):
    broker = self.positions["EURUSD"]

    if broker.is_closed():
        broker.buy(quantity=0.5)
```

### [`self.indicators`](../../src/quantex/strategy.py:49)

[`self.indicators`](../../src/quantex/strategy.py:49) stores indicator arrays created through [`Strategy.Indicator()`](../../src/quantex/strategy.py:126).

In most strategies you will also keep direct references such as `self.fast_ma` or `self.rsi`, because those are easier to read than indexing the list directly.

## Reading market data

The most common values you will read are:

- [`DataSource.COpen`](../../src/quantex/datasource.py:145)
- [`DataSource.CHigh`](../../src/quantex/datasource.py:155)
- [`DataSource.CLow`](../../src/quantex/datasource.py:165)
- [`DataSource.CClose`](../../src/quantex/datasource.py:175)
- [`DataSource.CVolume`](../../src/quantex/datasource.py:185)

These all mean “the current visible bar”.

You can also access visible history with:

- [`DataSource.Open`](../../src/quantex/datasource.py:90)
- [`DataSource.High`](../../src/quantex/datasource.py:101)
- [`DataSource.Low`](../../src/quantex/datasource.py:112)
- [`DataSource.Close`](../../src/quantex/datasource.py:123)
- [`DataSource.Volume`](../../src/quantex/datasource.py:134)

Example:

```python
def next(self):
    close_now = self.data["TEST"].CClose
    previous_close = self.data["TEST"].Close[-2]
    recent_closes = self.data["TEST"].Close[-10:]
```

## Creating indicators

Quantex does not provide a built-in indicator catalogue. Instead, you calculate arrays yourself and register them as time-aware indicators.

Example:

```python
import pandas as pd


def init(self):
    self.add_data(CSVDataSource("data.csv"), "TEST")
    close = self.data["TEST"].Close

    sma_20 = pd.Series(close).rolling(window=20).mean().to_numpy()
    sma_50 = pd.Series(close).rolling(window=50).mean().to_numpy()

    self.sma_20 = self.Indicator(sma_20)
    self.sma_50 = self.Indicator(sma_50)
```

The returned object is a [`TimeNDArray`](../../src/quantex/helpers.py:7), which exposes only data up to the current backtest step.

That lets you write logic such as:

```python
def next(self):
    if len(self.sma_20) < 2 or len(self.sma_50) < 2:
        return

    if self.sma_20[-2] <= self.sma_50[-2] and self.sma_20[-1] > self.sma_50[-1]:
        self.positions["TEST"].buy(quantity=0.5)
```

## Placing orders

The current public order methods are:

- [`Broker.buy()`](../../src/quantex/broker.py:159)
- [`Broker.sell()`](../../src/quantex/broker.py:235)
- [`Broker.close()`](../../src/quantex/broker.py:307)

### Buy orders

```python
def next(self):
    broker = self.positions["TEST"]

    broker.buy(quantity=0.25)
    broker.buy(quantity=0.25, limit=99.5)
    broker.buy(quantity=0.25, stop_loss=95.0, take_profit=110.0)
```

In the current implementation of [`Broker.buy()`](../../src/quantex/broker.py:159), `quantity` is interpreted as a fraction of broker cash unless `amount` is supplied.

### Sell orders

```python
def next(self):
    broker = self.positions["TEST"]

    broker.sell(quantity=0.25)
    broker.sell(quantity=0.25, limit=105.0)
```

[`Broker.sell()`](../../src/quantex/broker.py:235) can reduce a long position or open/increase a short position, depending on the current broker state.

### Closing a position

```python
def next(self):
    broker = self.positions["TEST"]

    if broker.is_long():
        broker.close()
```

[`Broker.close()`](../../src/quantex/broker.py:307) submits a market order that offsets the current position.

## Position state

Useful broker attributes and helpers include:

- [`Broker.position`](../../src/quantex/broker.py:143)
- [`Broker.position_avg_price`](../../src/quantex/broker.py:144)
- [`Broker.cash`](../../src/quantex/broker.py:145)
- [`Broker.orders`](../../src/quantex/broker.py:151)
- [`Broker.complete_orders`](../../src/quantex/broker.py:152)
- [`Broker.is_long()`](../../src/quantex/broker.py:359)
- [`Broker.is_short()`](../../src/quantex/broker.py:373)
- [`Broker.is_closed()`](../../src/quantex/broker.py:387)

Example:

```python
def next(self):
    broker = self.positions["TEST"]

    print("position", broker.position)
    print("avg price", broker.position_avg_price)
    print("cash", broker.cash)
    print("pending orders", len(broker.orders))
    print("completed orders", len(broker.complete_orders))
```

## Multi-symbol strategies

Quantex can attach multiple symbols to one strategy.

```python
from quantex import Strategy, CSVDataSource


class MultiSymbolStrategy(Strategy):
    def init(self):
        self.add_data(CSVDataSource("eurusd.csv"), "EURUSD")
        self.add_data(CSVDataSource("gbpusd.csv"), "GBPUSD")

    def next(self):
        if self.data["EURUSD"].CClose > self.data["GBPUSD"].CClose:
            self.positions["EURUSD"].buy(quantity=0.2)
```

Important detail: [`SimpleBacktester.run()`](../../src/quantex/backtester.py:443) splits starting cash evenly across all attached brokers.

That means the cash available to each symbol-specific broker is a fraction of the total starting cash.

## Common mistakes to avoid

### 1. Forgetting to attach data

If you never call [`Strategy.add_data()`](../../src/quantex/strategy.py:98), the strategy has no symbol data and no broker to trade through.

### 2. Using indicator values too early

Rolling calculations often contain `NaN` values at the start. Guard with length checks before using recent values.

### 3. Assuming built-in indicators exist

The library gives you [`Strategy.Indicator()`](../../src/quantex/strategy.py:126), but not built-in SMA, EMA, or RSI functions.

### 4. Assuming orders fill at close

The current broker logic in [`Broker._iterate()`](../../src/quantex/broker.py:483) executes market orders at the current open, not the current close.

## Minimal debugging approach

One easy way to understand a strategy is to print values inside [`Strategy.next()`](../../src/quantex/strategy.py:71):

```python
def next(self):
    print("time", self.data["TEST"].Index[self.data["TEST"].current_index])
    print("open", self.data["TEST"].COpen)
    print("close", self.data["TEST"].CClose)
    print("position", self.positions["TEST"].position)
```

## Running the strategy

Once the class is written, execution is always the same pattern:

```python
strategy = MovingAverageCross(fast_period=5, slow_period=20)
backtester = SimpleBacktester(strategy, cash=10_000)
report = backtester.run()

print(report)
```

For a deeper explanation of reports and performance metrics, see [Backtesting guide](./backtesting.md).

