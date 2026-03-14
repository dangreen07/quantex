# Indicators Guide

This guide explains how indicators work in Quantex.

The most important correction to earlier documentation is simple: Quantex does not currently ship a built-in library of indicator functions. Instead, it provides a mechanism for making your own arrays time-aware during backtesting.

That mechanism is [`Strategy.Indicator()`](../../src/quantex/strategy.py:126), which wraps a NumPy array in [`TimeNDArray`](../../src/quantex/helpers.py:7).

## What [`Strategy.Indicator()`](../../src/quantex/strategy.py:126) does

When you call [`Strategy.Indicator()`](../../src/quantex/strategy.py:126):

1. your input array is converted to [`TimeNDArray`](../../src/quantex/helpers.py:7)
2. the time-aware array is appended to [`Strategy.indicators`](../../src/quantex/strategy.py:49)
3. the same object is returned so you can store it on `self`

Example:

```python
import pandas as pd


def init(self):
    close = self.data["TEST"].Close
    sma = pd.Series(close).rolling(window=20).mean().to_numpy()
    self.sma_20 = self.Indicator(sma)
```

## Why time-aware arrays matter

During a backtest, future bars should not be visible to the strategy.

[`TimeNDArray`](../../src/quantex/helpers.py:7) enforces this by keeping an internal visibility index. On each backtest step, [`SimpleBacktester.run()`](../../src/quantex/backtester.py:461) advances that visible window for every registered indicator.

That means:

- `indicator[-1]` means the latest currently visible value
- `indicator[-2]` means the previous visible value
- slices such as `indicator[-10:]` only expose visible history

## First complete example

```python
from quantex import Strategy, CSVDataSource
import pandas as pd


class SmaStrategy(Strategy):
    def init(self):
        self.add_data(CSVDataSource("data.csv"), "TEST")

        close = self.data["TEST"].Close
        self.sma_10 = self.Indicator(
            pd.Series(close).rolling(window=10).mean().to_numpy()
        )
        self.sma_20 = self.Indicator(
            pd.Series(close).rolling(window=20).mean().to_numpy()
        )

    def next(self):
        if len(self.sma_10) < 2 or len(self.sma_20) < 2:
            return

        if self.sma_10[-2] <= self.sma_20[-2] and self.sma_10[-1] > self.sma_20[-1]:
            self.positions["TEST"].buy(quantity=0.5)
```

## Building indicators with pandas or NumPy

The usual pattern is:

1. take a visible historical array such as [`DataSource.Close`](../../src/quantex/datasource.py:123)
2. compute an indicator array with pandas or NumPy
3. register it with [`Strategy.Indicator()`](../../src/quantex/strategy.py:126)

### Simple moving average

```python
close = self.data["TEST"].Close
self.sma_20 = self.Indicator(
    pd.Series(close).rolling(window=20).mean().to_numpy()
)
```

### Exponential moving average

```python
close = self.data["TEST"].Close
self.ema_20 = self.Indicator(
    pd.Series(close).ewm(span=20).mean().to_numpy()
)
```

### RSI

```python
close = self.data["TEST"].Close
delta = pd.Series(close).diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss

self.rsi_14 = self.Indicator((100 - (100 / (1 + rs))).to_numpy())
```

## Accessing indicator values

The returned object behaves like an array with time-aware bounds.

Example:

```python
def next(self):
    current_value = self.sma_20[-1]
    previous_value = self.sma_20[-2]
    recent_values = self.sma_20[-5:]
```

This behavior comes from [`TimeNDArray.__getitem__()`](../../src/quantex/helpers.py:180) and [`TimeNDArray.__len__()`](../../src/quantex/helpers.py:118).

## Avoiding common mistakes

### 1. Reading too early

Many rolling indicators begin with `NaN` values. Guard with length checks before using the newest values.

```python
def next(self):
    if len(self.sma_20) < 2:
        return
```

### 2. Assuming indicators are updated automatically from formulas

The array visibility is updated automatically, but the numerical values are whatever you computed up front. If you want a different indicator formula, you must create that array yourself.

### 3. Assuming Quantex provides built-in functions such as `self.sma()`

The current [`Strategy`](../../src/quantex/strategy.py:9) base class does not define `sma`, `ema`, `rsi`, or similar helpers. If you use those names, they must be your own methods or external functions.

## Multi-indicator example

```python
from quantex import Strategy, CSVDataSource
import pandas as pd


class MultiIndicatorStrategy(Strategy):
    def init(self):
        self.add_data(CSVDataSource("data.csv"), "TEST")
        close = self.data["TEST"].Close

        self.sma_10 = self.Indicator(pd.Series(close).rolling(10).mean().to_numpy())
        self.sma_30 = self.Indicator(pd.Series(close).rolling(30).mean().to_numpy())
        self.volatility = self.Indicator(pd.Series(close).pct_change().rolling(20).std().to_numpy())

    def next(self):
        if len(self.sma_10) < 2 or len(self.sma_30) < 2:
            return

        broker = self.positions["TEST"]

        bullish_cross = self.sma_10[-2] <= self.sma_30[-2] and self.sma_10[-1] > self.sma_30[-1]
        bearish_cross = self.sma_10[-2] >= self.sma_30[-2] and self.sma_10[-1] < self.sma_30[-1]

        if bullish_cross and broker.is_closed():
            broker.buy(quantity=0.25)
        elif bearish_cross and broker.is_long():
            broker.close()
```

## Using indicators with multiple symbols

```python
def init(self):
    self.add_data(CSVDataSource("eurusd.csv"), "EURUSD")
    self.add_data(CSVDataSource("gbpusd.csv"), "GBPUSD")

    self.eurusd_sma = self.Indicator(
        pd.Series(self.data["EURUSD"].Close).rolling(20).mean().to_numpy()
    )
    self.gbpusd_sma = self.Indicator(
        pd.Series(self.data["GBPUSD"].Close).rolling(20).mean().to_numpy()
    )
```

## Using indicators with multi-timeframe data

Quantex lets you attach multiple sources, so you can compute indicators from different files or timeframes.

```python
def init(self):
    self.add_data(CSVDataSource("eurusd_m1.csv"), "EURUSD_M1")
    self.add_data(CSVDataSource("eurusd_h1.csv"), "EURUSD_H1")

    self.m1_sma = self.Indicator(
        pd.Series(self.data["EURUSD_M1"].Close).rolling(20).mean().to_numpy()
    )
    self.h1_sma = self.Indicator(
        pd.Series(self.data["EURUSD_H1"].Close).rolling(50).mean().to_numpy()
    )
```

Be careful here: Quantex does not provide special synchronization logic beyond advancing each source index during the backtest loop. If your files have different lengths or timestamp structures, strategy logic must account for that.

## Indicator debugging

Printing a few values from inside [`Strategy.next()`](../../src/quantex/strategy.py:71) is often enough to confirm indicator behavior.

```python
def next(self):
    if len(self.sma_20) < 2:
        return

    print(self.data["TEST"].Index[self.data["TEST"].current_index])
    print("price", self.data["TEST"].CClose)
    print("sma", self.sma_20[-1])
```

## What the current indicator system does not include

The current codebase does **not** include:

- a built-in technical indicator module
- automatic plotting helpers
- automatic indicator parameter optimization helpers
- a dictionary-based indicator manager on the base [`Strategy`](../../src/quantex/strategy.py:9)

Earlier documentation included many conceptual examples of those ideas, but they are not part of the current implementation.

## Summary

Use indicators in Quantex like this:

1. compute a NumPy array with pandas or NumPy
2. register it with [`Strategy.Indicator()`](../../src/quantex/strategy.py:126)
3. access only the visible portion during [`Strategy.next()`](../../src/quantex/strategy.py:71)

For strategy structure, see [Strategy guide](./strategy.md). For parameter search, see [Optimization guide](./optimizer.md).

