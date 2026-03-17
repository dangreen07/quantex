# QuantEx Documentation

QuantEx is a Python backtesting library for people who want to write trading rules and run them on historical OHLCV market data.

This documentation is written to match the current codebase, not an imagined future version. Where the library is intentionally small or has limitations, those limitations are stated explicitly.

## What QuantEx is

At its core, QuantEx provides four main concepts:

1. [`Strategy`](../src/quantex/strategy.py:9) defines your trading logic.
2. [`DataSource`](../src/quantex/datasource.py:6) provides time-indexed market data.
3. [`Broker`](../src/quantex/broker.py:113) manages orders, positions, and cash for each attached symbol.
4. [`SimpleBacktester`](../src/quantex/backtester.py:356) runs the bar-by-bar simulation and can search parameter combinations.

If you are new to the library, the usual workflow is:

- load a CSV or Parquet file into a data source
- attach that data source to a strategy
- create indicator arrays if needed
- place buy, sell, or close orders inside your strategy logic
- run the strategy through the backtester
- inspect the resulting [`BacktestReport`](../src/quantex/backtester.py:188)

## What QuantEx is not

The current codebase does **not** try to be all of the following:

- a live-trading framework
- a full portfolio accounting platform
- a live execution stack or portfolio analytics platform beyond backtesting
- a feature-complete execution simulator with slippage, partial fills, or order cancellation

Those distinctions matter because some older documentation implied broader functionality than the code currently implements.

## Package contents

The public package exports are currently defined in [`src/quantex/__init__.py`](../src/quantex/__init__.py):

- [`CSVDataSource`](../src/quantex/__init__.py:1)
- [`ParquetDataSource`](../src/quantex/__init__.py:1)
- [`DataSource`](../src/quantex/__init__.py:1)
- [`Strategy`](../src/quantex/__init__.py:2)
- [`SimpleBacktester`](../src/quantex/__init__.py:3)
- [`CommissionType`](../src/quantex/__init__.py:4)
- [`indicators`](../src/quantex/__init__.py:5)

## Quick start

### Installation

```bash
pip install quantex
```

Python version support is declared in [`pyproject.toml`](../pyproject.toml:9) as Python 3.10 or newer.

### Smallest useful example

```python
from quantex import Strategy, CSVDataSource, SimpleBacktester
import pandas as pd


class BuyAndHold(Strategy):
    def __init__(self):
        super().__init__()
        self.entered = False

    def init(self):
        self.add_data(CSVDataSource("data.csv"), "TEST")

    def next(self):
        if not self.entered and len(self.data["TEST"].Close) >= 2:
            self.positions["TEST"].buy(quantity=1.0)
            self.entered = True


strategy = BuyAndHold()
backtester = SimpleBacktester(strategy, cash=10_000)
report = backtester.run()

print(report)
```

This example works as follows:

- [`CSVDataSource`](../src/quantex/datasource.py:194) loads a CSV file into a [`DataSource`](../src/quantex/datasource.py:6)
- [`Strategy.add_data()`](../src/quantex/strategy.py:98) stores the data source and creates a matching [`Broker`](../src/quantex/broker.py:113)
- [`SimpleBacktester.run()`](../src/quantex/backtester.py:414) advances the simulation one bar at a time
- [`Broker.buy()`](../src/quantex/broker.py:159) places an order that the broker processes on the next iteration step

## How the backtest loop works

Understanding the backtest loop makes the rest of the library easier to use.

When you create [`SimpleBacktester`](../src/quantex/backtester.py:356), the provided strategy is deep-copied in [`SimpleBacktester.__init__()`](../src/quantex/backtester.py:380). When you call [`SimpleBacktester.run()`](../src/quantex/backtester.py:414), the following happens:

1. Starting cash is split evenly across all registered brokers in [`SimpleBacktester.run()`](../src/quantex/backtester.py:443).
2. Every data source receives the current bar index by updating [`DataSource.current_index`](../src/quantex/datasource.py:62).
3. Every broker processes pending and active orders through [`Broker._iterate()`](../src/quantex/broker.py:483).
4. Every time-aware indicator is advanced by updating its visibility window in [`SimpleBacktester.run()`](../src/quantex/backtester.py:461).
5. Your [`Strategy.next()`](../src/quantex/strategy.py:71) method is called.
6. At the end, all brokers are asked to close open positions via [`Broker.close()`](../src/quantex/broker.py:307), and the combined equity curve is returned inside [`BacktestReport`](../src/quantex/backtester.py:188).

## Data model

### Required input data

[`DataSource`](../src/quantex/datasource.py:6) requires a pandas DataFrame with these columns:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

That validation happens in [`DataSource.__init__()`](../src/quantex/datasource.py:32).

### Current bar vs historical arrays

The library distinguishes between:

- full visible history, such as [`DataSource.Close`](../src/quantex/datasource.py:123)
- current-bar values, such as [`DataSource.CClose`](../src/quantex/datasource.py:175)

This is important because strategies normally make decisions using the current bar and recent history, not the entire future dataset.

## Orders and execution behavior

The order system is defined around [`Order`](../src/quantex/broker.py:52), [`OrderSide`](../src/quantex/broker.py:11), [`OrderType`](../src/quantex/broker.py:24), and [`OrderStatus`](../src/quantex/broker.py:37).

The current implementation supports:

- market orders
- limit orders
- optional stop-loss and take-profit levels attached to an order
- long and short positions
- percentage commissions and cash-per-lot commissions through [`CommissionType`](../src/quantex/enums.py:4)

Important implementation details from [`Broker._iterate()`](../src/quantex/broker.py:483):

- market orders execute using the current bar open, via [`DataSource.COpen`](../src/quantex/datasource.py:145)
- buy limit orders trigger when the current open is less than or equal to the limit price
- sell limit orders trigger when the current open is greater than or equal to the limit price
- stop-loss and take-profit monitoring also uses the current open
- slippage is **not** modeled in the current code
- there is no public order cancellation API

## Indicators

QuantEx ships a built-in indicator catalogue through [`quantex.indicators`](../src/quantex/indicators.py) and the strategy-level convenience handle `self.ta` created in [`Strategy.__init__()`](../src/quantex/strategy.py:40).

The indicator module includes common stock-trading studies such as moving averages, RSI, MACD, Bollinger Bands, ATR, stochastic oscillator, CCI, Williams %R, OBV, MFI, ADX, Keltner Channels, Donchian Channels, Aroon, and Vortex, plus more advanced tools such as Ichimoku Cloud, KAMA, TRIX, Fisher Transform, Hurst exponent, rolling Sharpe/Sortino ratios, Z-score, and linear regression slope.

Use [`Strategy.Indicator()`](../src/quantex/strategy.py:126) to wrap any returned NumPy array in [`TimeNDArray`](../src/quantex/helpers.py:7). The wrapper makes only the currently visible portion of the array accessible during the backtest.

## Optimization

QuantEx includes two optimization entry points:

- [`SimpleBacktester.optimize()`](../src/quantex/backtester.py:485)
- [`SimpleBacktester.optimize_parallel()`](../src/quantex/backtester.py:659)

Both perform grid search over candidate parameter values. The parallel version distributes combinations across worker processes and then reruns the best result locally to produce the full report.

## Guides

- [Strategy guide](./usage/strategy.md)
- [Data sources guide](./usage/data-sources.md)
- [Backtesting guide](./usage/backtesting.md)
- [Execution guide](./usage/execution.md)
- [Indicators guide](./usage/indicators.md)
- [Optimization guide](./usage/optimizer.md)

## API reference

Reference pages live under [`docs/reference/`](./reference/).

## Project structure

The main source files in the current codebase are:

- [`src/quantex/strategy.py`](../src/quantex/strategy.py)
- [`src/quantex/datasource.py`](../src/quantex/datasource.py)
- [`src/quantex/broker.py`](../src/quantex/broker.py)
- [`src/quantex/backtester.py`](../src/quantex/backtester.py)
- [`src/quantex/helpers.py`](../src/quantex/helpers.py)
- [`src/quantex/enums.py`](../src/quantex/enums.py)

The test suite is also a useful source of truth for documented behavior:

- [`tests/test_strategy.py`](../tests/test_strategy.py)
- [`tests/test_datasource.py`](../tests/test_datasource.py)
- [`tests/test_broker.py`](../tests/test_broker.py)
- [`tests/test_backtester.py`](../tests/test_backtester.py)

## License

See [`LICENSE.md`](../LICENSE.md).
