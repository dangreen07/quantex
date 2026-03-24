# Backtesting Guide

This guide explains what Quantex actually does when you run a backtest and how to interpret the results.

If you are new to the library, a backtest in Quantex means:

- your strategy class is executed on historical bars
- brokers manage orders and positions for each symbol
- equity is recorded over time
- a [`BacktestReport`](../../src/quantex/backtester.py:188) is returned at the end

## The backtester class

The central class is [`SimpleBacktester`](../../src/quantex/backtester.py:356).

Constructor signature:

- strategy: a concrete [`Strategy`](../../src/quantex/strategy.py:9)
- `cash`: starting capital
- `commission`: commission rate or cash amount, depending on commission type
- `commission_type`: a [`CommissionType`](../../src/quantex/enums.py:4)
- `lot_size`: used for cash commission calculations
- `margin_call`: threshold used in broker margin logic
- `leverage`: leverage multiplier for amplified position sizing (default: 1.0)

## Smallest working example

```python
from quantex import Strategy, CSVDataSource, SimpleBacktester


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

## What happens inside [`SimpleBacktester.run()`](../../src/quantex/backtester.py:414)

The method does the following in order:

1. splits cash evenly across all brokers in [`SimpleBacktester.run()`](../../src/quantex/backtester.py:443)
2. calls [`Strategy.init()`](../../src/quantex/strategy.py:52)
3. loops over the longest attached data series in [`SimpleBacktester.run()`](../../src/quantex/backtester.py:456)
4. updates each source's current index
5. calls [`Broker._iterate()`](../../src/quantex/broker.py:483) on each broker
6. updates indicator visibility windows
7. calls [`Strategy.next()`](../../src/quantex/strategy.py:71)
8. asks each broker to close open positions at the end via [`Broker.close()`](../../src/quantex/broker.py:307)
9. aggregates broker equity into the final [`BacktestReport.PnlRecord`](../../src/quantex/backtester.py:205)

## Important behavior to understand

### Strategy copy behavior

[`SimpleBacktester.__init__()`](../../src/quantex/backtester.py:380) deep-copies the strategy you pass in.

That means the strategy object inside the backtester is not the same Python object you originally created.

### Cash allocation in multi-symbol tests

If you attach multiple symbols, starting cash is divided equally across the symbol-specific brokers in [`SimpleBacktester.run()`](../../src/quantex/backtester.py:443).

This avoids double-counting cash across symbols, but it also means each symbol starts with only part of the total portfolio cash.

### Orders are processed before [`Strategy.next()`](../../src/quantex/strategy.py:71)

In each loop, brokers process existing orders before the strategy places new ones for that same step. This means a newly placed order is not processed until the following iteration.

### Market orders use the open price

The broker executes market orders using [`DataSource.COpen`](../../src/quantex/datasource.py:145), not the close.

That detail matters when you compare expected trade prices with results.

## Backtest configuration

```python
from quantex import CommissionType

backtester = SimpleBacktester(
    strategy,
    cash=50_000,
    commission=0.002,
    commission_type=CommissionType.PERCENTAGE,
    lot_size=1,
    margin_call=0.5,
    leverage=1.0,
)
```

### Leverage for amplified position sizing

Quantex supports leverage for amplifying position sizes. With leverage:

- A `leverage` of 2.0 means you control 2x the position with the same cash
- The broker calculates margin as `position_value / leverage`
- Commission is still applied on the full position value

Example:

```python
# 2x leverage - control twice the position with same cash
backtester = SimpleBacktester(
    strategy,
    cash=10_000,
    leverage=2.0,  # 2x leverage
)

# Result: A buy order with quantity=0.5 will control 100% of cash worth of shares
# but only use 50% of cash as margin
```

### Commission types

The available commission modes come from [`CommissionType`](../../src/quantex/enums.py:4):

- [`CommissionType.PERCENTAGE`](../../src/quantex/enums.py:17)
- [`CommissionType.CASH`](../../src/quantex/enums.py:18)

Commission is applied through [`Broker._calc_commission()`](../../src/quantex/broker.py:440).

## Understanding the report object

[`SimpleBacktester.run()`](../../src/quantex/backtester.py:414) returns [`BacktestReport`](../../src/quantex/backtester.py:188).

Key fields:

- [`BacktestReport.starting_cash`](../../src/quantex/backtester.py:203)
- [`BacktestReport.final_cash`](../../src/quantex/backtester.py:204)
- [`BacktestReport.PnlRecord`](../../src/quantex/backtester.py:205)
- [`BacktestReport.orders`](../../src/quantex/backtester.py:206)
- [`BacktestReport.tradeRecord`](../../src/quantex/backtester.py:207)

Key derived properties and methods:

- [`BacktestReport.total_return`](../../src/quantex/backtester.py:227)
- [`BacktestReport.periods_per_year`](../../src/quantex/backtester.py:214)
- [`BacktestReport.kelly_criterion`](../../src/quantex/backtester.py:232)
- [`BacktestReport.plot()`](../../src/quantex/backtester.py:252)
- [`BacktestReport.__str__()`](../../src/quantex/backtester.py:299)

Example:

```python
print(report.starting_cash)
print(report.final_cash)
print(report.total_return)
print(report.periods_per_year)
print(report.orders)
report.plot()
```

## About Sharpe ratio and summary output

The summary string from [`BacktestReport.__str__()`](../../src/quantex/backtester.py:299) computes Sharpe ratio directly from the equity returns series.

Important correction: the current [`BacktestReport`](../../src/quantex/backtester.py:188) does **not** expose a `report.sharpe` attribute or a `report.max_drawdown` property. Older docs referenced them as direct attributes, but the current code only computes those values inside string formatting or helper functions.

If you want maximum drawdown directly, use [`max_drawdown()`](../../src/quantex/backtester.py:19) on [`BacktestReport.PnlRecord`](../../src/quantex/backtester.py:205):

```python
from quantex.backtester import max_drawdown

dd = max_drawdown(report.PnlRecord)
print(dd)
```

## Visualization

[`BacktestReport.plot()`](../../src/quantex/backtester.py:252) creates two panels:

- equity curve
- drawdown area chart

```python
report.plot(figsize=(12, 6))
```

## Multi-symbol backtests

```python
from quantex import Strategy, CSVDataSource, SimpleBacktester


class TwoSymbolStrategy(Strategy):
    def init(self):
        self.add_data(CSVDataSource("eurusd.csv"), "EURUSD")
        self.add_data(CSVDataSource("gbpusd.csv"), "GBPUSD")

    def next(self):
        if self.data["EURUSD"].CClose > self.data["GBPUSD"].CClose:
            self.positions["EURUSD"].buy(quantity=0.2)


report = SimpleBacktester(TwoSymbolStrategy(), cash=20_000).run()
```

Remember that each broker receives an equal share of the starting cash in this setup.

## Optimization from the backtester

[`SimpleBacktester`](../../src/quantex/backtester.py:356) also provides parameter search.

### Sequential grid search

```python
params = {
    "fast_period": range(5, 11),
    "slow_period": range(15, 31, 5),
}

best_params, best_report, results_df = backtester.optimize(
    params,
    constraint=lambda p: p["fast_period"] < p["slow_period"],
)
```

### Parallel grid search

```python
best_params, best_report, results_df = backtester.optimize_parallel(
    params,
    workers=4,
    chunksize=1,
)
```

The parallel method is defined in [`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester.py:659).

## Reading optimization results

Both optimization methods return:

1. `best_params`
2. `best_report`
3. `results_df`

`results_df` is a pandas DataFrame assembled from the evaluated parameter combinations.

Important detail from the current implementation:

- [`SimpleBacktester.optimize()`](../../src/quantex/backtester.py:485) stores `report.orders` in the `trades` column, not a trade count
- [`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester.py:659) stores a numeric trade count in the `trades` field returned by workers

So treat the `trades` column carefully depending on which optimizer you used.

## Practical debugging tips

### Print broker state inside the strategy

```python
def next(self):
    broker = self.positions["TEST"]
    print(self.data["TEST"].Index[self.data["TEST"].current_index])
    print(self.data["TEST"].COpen, self.data["TEST"].CClose)
    print(broker.position, broker.cash)
```

### Run without a progress bar when scripting

[`SimpleBacktester.run()`](../../src/quantex/backtester.py:414) defaults to `progress_bar=False`, which is usually the cleanest option for logs and tests.

### Use the tests as reference behavior

See [`tests/test_backtester.py`](../../tests/test_backtester.py) for the current expected behavior of initialization, reporting, optimization validation, and period inference.

## Limitations and caveats

The current implementation does **not** include:

- slippage modeling
- partial fills
- public order cancellation
- built-in walk-forward analysis helpers
- built-in out-of-sample evaluation workflow

Older documentation described some of those topics conceptually, but they are not first-class features in the current code.

## Summary

Use [`SimpleBacktester`](../../src/quantex/backtester.py:356) when you want to:

- execute a strategy on historical data
- get an equity curve and order log
- compare parameter combinations with grid search

For more detail on order behavior, see [Execution guide](./execution.md). For strategy design, see [Strategy guide](./strategy.md).

