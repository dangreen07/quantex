# Optimization Guide

This guide explains the optimization features that Quantex currently provides.

The current implementation supports grid search over parameter combinations through:

- [`SimpleBacktester.optimize()`](../../src/quantex/backtester.py:485)
- [`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester.py:659)

This guide focuses on what those methods actually do in the current codebase.

## What optimization means in Quantex

Optimization in Quantex means:

1. choose one or more strategy attributes to vary
2. provide candidate values for each attribute
3. run a separate backtest for each valid combination
4. compare the resulting metrics

The optimizer does not do Bayesian optimization, evolutionary search, or walk-forward validation automatically. It performs exhaustive grid search.

## Basic example

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
backtester = SimpleBacktester(strategy, cash=10_000)

params = {
    "fast_period": range(5, 11),
    "slow_period": range(15, 31, 5),
}

best_params, best_report, results_df = backtester.optimize(
    params,
    constraint=lambda p: p["fast_period"] < p["slow_period"],
)

print(best_params)
print(best_report)
print(results_df.head())
```

## Parameter format

Both optimization methods expect a dictionary whose keys are strategy attribute names and whose values are iterables.

Example:

```python
params = {
    "fast_period": [5, 10, 15],
    "slow_period": range(20, 41, 10),
    "position_size": [0.1, 0.2, 0.3],
}
```

The optimizer applies those values using `setattr` in [`SimpleBacktester.optimize()`](../../src/quantex/backtester.py:574) and [`_worker_eval()`](../../src/quantex/backtester.py:138).

That means the named attributes must already make sense for your strategy design.

## Constraints

Both optimizers accept an optional `constraint` callable.

Example:

```python
def valid_combo(params):
    return params["fast_period"] < params["slow_period"]


best_params, best_report, results_df = backtester.optimize(
    params,
    constraint=valid_combo,
)
```

The constraint is checked before each combination is evaluated in [`SimpleBacktester.optimize()`](../../src/quantex/backtester.py:563) and [`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester.py:744).

## Sequential optimization

[`SimpleBacktester.optimize()`](../../src/quantex/backtester.py:485) evaluates combinations one after another in the current process.

Use it when:

- the parameter grid is modest in size
- debugging is more important than raw speed
- multiprocessing overhead would be unnecessary

### Validation behavior

The method raises:

- `ValueError` for an empty parameter dictionary
- `ValueError` when any parameter has no candidate values
- `TypeError` when a parameter value is not iterable

These behaviors are tested in [`tests/test_backtester.py`](../../tests/test_backtester.py).

## Parallel optimization

[`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester.py:659) uses multiple worker processes through `ProcessPoolExecutor`.

Example:

```python
best_params, best_report, results_df = backtester.optimize_parallel(
    params,
    workers=4,
    chunksize=1,
)
```

### How it works

The implementation:

1. materializes parameter ranges
2. pickles the base strategy in [`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester.py:760)
3. initializes worker state through [`_worker_init()`](../../src/quantex/backtester.py:77)
4. evaluates combinations in [`_worker_eval()`](../../src/quantex/backtester.py:106)
5. rebuilds a results DataFrame in the main process
6. reruns the best parameter set locally to obtain a full [`BacktestReport`](../../src/quantex/backtester.py:188)

### Choosing worker counts

If you pass `workers=None`, the method chooses a conservative value in [`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester.py:755).

## What metrics are compared

The optimizer calculates or stores:

- `final_cash`
- `total_return`
- `sharpe`
- `max_drawdown`
- `trades`

The sequential optimizer computes these inside [`SimpleBacktester.optimize()`](../../src/quantex/backtester.py:587).

The parallel optimizer computes them in [`_worker_eval()`](../../src/quantex/backtester.py:151).

## Custom optimization objectives

Both optimizers accept an `objective` argument. By default it is `"sharpe"`, but you can target any metric exposed by [`BacktestReport`](../../src/quantex/backtester.py:188) or any computed optimizer metric.

Supported built-in optimization metrics are:

- `final_cash`
- `total_return`
- `sharpe`
- `max_drawdown`
- `trades`

Example:

```python
best_params, best_report, results_df = backtester.optimize(
    params,
    objective="total_return",
)
```

If you choose a `BacktestReport` property directly, the optimizer will call it or read it just like the report does.

## Risk tolerance filtering

Both optimizers also accept an optional `risk_tolerance` dictionary. It is off by default.

Each entry is treated as a maximum allowed value. Any candidate exceeding one of the thresholds is discarded before it can be selected.

Example:

```python
best_params, best_report, results_df = backtester.optimize(
    params,
    objective="total_return",
    risk_tolerance={"max_drawdown": 0.05},
)
```

This example rejects any strategy whose maximum drawdown exceeds 5%.

## How the best result is chosen

### Sequential optimizer

[`SimpleBacktester.optimize()`](../../src/quantex/backtester.py:624) primarily scores combinations by Sharpe ratio. If Sharpe is not finite, it heavily penalizes the result.

### Parallel optimizer

[`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester.py:786) builds a score from Sharpe ratio, but only when total return is positive.

That means the two methods are similar but not perfectly identical in how they rank edge cases.

## Reading `results_df`

The returned `results_df` is a pandas DataFrame containing one row per evaluated combination.

Example:

```python
print(results_df.columns)
print(results_df.head())
```

Useful patterns:

```python
top_10 = results_df.head(10)
positive_returns = results_df[results_df["total_return"] > 0]
```

### Important caveat about the `trades` column

The `trades` field is inconsistent across implementations:

- [`SimpleBacktester.optimize()`](../../src/quantex/backtester.py:619) stores `report.orders`
- [`_worker_eval()`](../../src/quantex/backtester.py:179) stores `len(report.orders)`

So if you compare sequential and parallel optimization outputs, do not assume the `trades` column has the same type.

## Practical optimization workflow

### Step 1: start with a small grid

```python
params = {
    "fast_period": [5, 10],
    "slow_period": [20, 30],
}
```

### Step 2: add a validity constraint

```python
constraint=lambda p: p["fast_period"] < p["slow_period"]
```

### Step 3: inspect the top results

```python
print(results_df.head())
```

### Step 4: rerun and inspect the best report

```python
print(best_params)
print(best_report)
best_report.plot()
```

## Overfitting warning

Optimization can easily produce attractive in-sample results that do not generalize.

Quantex currently gives you the grid search machinery, but it does not automatically perform:

- train/test splits for optimization runs
- walk-forward analysis
- cross-validation
- stability analysis

If you need those workflows, build them around [`DataSource`](../../src/quantex/datasource.py:6), [`SimpleBacktester`](../../src/quantex/backtester.py:356), and your own experiment code.

## Performance considerations

### Use sequential search when

- your grid is small
- you want simpler debugging
- you are iterating on strategy design

### Use parallel search when

- the grid is large enough to justify multiprocessing
- the strategy and data are picklable
- your machine has spare CPU and memory capacity

The parallel implementation keeps a copy of the strategy in each worker, so memory usage scales with worker count.

## What the current optimizer does not include

The current codebase does **not** include:

- Bayesian optimization
- evolutionary algorithms
- native multi-objective optimization
- built-in walk-forward optimization
- automatic out-of-sample validation reports

Some older documentation discussed those ideas conceptually, but they are not implemented as first-class library features.

## Summary

Use [`SimpleBacktester.optimize()`](../../src/quantex/backtester.py:485) or [`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester.py:659) when you want exhaustive grid search over strategy attributes.

Keep these points in mind:

1. parameters are applied as plain strategy attributes
2. constraints are optional but often necessary
3. optimization results are only as meaningful as your validation process

For the underlying simulation model, see [Backtesting guide](./backtesting.md).

