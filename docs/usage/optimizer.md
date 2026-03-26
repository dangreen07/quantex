# Optimization Guide

This guide explains the optimization features that Quantex provides.

The implementation supports:

- **Grid search** over parameter combinations via:
  - [`SimpleBacktester.optimize()`](../../src/quantex/backtester/backtester.py:485) - Sequential grid search
  - [`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester/backtester.py:581) - Parallel grid search with adaptive chunksize
- **Bayesian optimization** via:
  - [`SimpleBacktester.optimize_optuna()`](../../src/quantex/backtester/backtester.py:709) - Smart search using Optuna (recommended for large search spaces)
- **Train/Validate/Test split optimization** via:
  - [`SimpleBacktester.optimize_with_split()`](../../src/quantex/backtester/backtester.py:1059)
- **Gradient descent optimization** via:
  - [`SimpleBacktester.optimize_gradient_descent()`](../../src/quantex/backtester/backtester.py:1336)
- **Walk-forward optimization** via:
  - [`WalkForwardAnalyzer`](../../src/quantex/backtester/walk_forward.py) - Rolling window optimization

This guide focuses on what those methods actually do in the current codebase.

## What optimization means in Quantex

Optimization in Quantex means:

1. Choose one or more strategy attributes to vary
2. Provide candidate values (grid search) or continuous bounds (gradient descent)
3. Run a separate backtest for each valid combination
4. Compare the resulting metrics

The optimizer performs exhaustive grid search or gradient-based optimization with built-in train/validate/test splits to prevent overfitting.

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


backtester = SimpleBacktester(MovingAverageCross(), cash=10_000)

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

[`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester/backtester.py:581) uses multiple worker processes through `ProcessPoolExecutor`.

Example:

```python
best_params, best_report, results_df = backtester.optimize_parallel(
    params,
    workers=4,
    chunksize="auto",  # Adaptive chunksize (recommended)
)
```

### How it works

The implementation:

1. Computes total combinations using `math.prod` (memory-efficient, no materialization)
2. pickles the base strategy in [`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester/backtester.py:582)
3. initializes worker state through [`_worker_init()`](../../src/quantex/backtester/parallel.py:18)
4. evaluates combinations in [`_worker_eval()`](../../src/quantex/backtester/parallel.py:73)
5. rebuilds a results DataFrame in the main process
6. reruns the best parameter set locally to obtain a full [`BacktestReport`](../../src/quantex/backtester/reports.py:1)

### Adaptive Chunksize

The default `chunksize="auto"` calculates optimal chunk size based on total combinations and worker count:

```python
chunksize = max(16, total_combos // (workers * 4))
```

This significantly reduces inter-process communication overhead compared to the previous default of `chunksize=1`.

### Choosing worker counts

If you pass `workers=None`, the method chooses a conservative value in [`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester/backtester.py:582).

## Bayesian Optimization with Optuna

For large parameter spaces, [`SimpleBacktester.optimize_optuna()`](../../src/quantex/backtester/backtester.py:709) provides intelligent search using Optuna's TPE (Tree-structured Parzen Estimator) sampler.

### When to use Optuna

| Scenario | Recommended Method |
|----------|-------------------|
| < 100 combos | `optimize()` or `optimize_parallel()` |
| 100-10,000 combos | `optimize_parallel()` |
| > 10,000 combos | `optimize_optuna()` |
| Continuous parameters | `optimize_optuna()` or `optimize_gradient_descent()` |

### Parameter Space Format

Optuna supports both continuous ranges and discrete lists:

```python
param_space = {
    'fast_period': (5, 50),        # Continuous: uniform sampling 5-50
    'slow_period': [20, 30, 50],   # Discrete: pick from list
    'threshold': (0.01, 0.1),       # Continuous: 1%-10%
}
```

### Basic example

```python
# Install optuna first: pip install optuna

result = backtester.optimize_optuna(
    param_space={
        'fast_period': (5, 50),
        'slow_period': (20, 100),
        'threshold': (0.01, 0.1),
    },
    n_trials=100,        # Number of optimization trials
    workers=4,           # Parallel execution
    objective="sharpe",   # Optimize for Sharpe ratio
)

print(f"Best parameters: {result.best_params}")
print(f"Best Sharpe: {result.train_metrics['sharpe']:.2f}")
```

### Features

- **Early pruning**: Unpromising trials are terminated early
- **Parallel execution**: Use `workers > 1` for faster optimization
- **Continuous & discrete**: Supports both parameter types
- **Reproducibility**: Set `random_seed` for consistent results
- **Timeout**: Set `timeout` to limit optimization time

### Performance comparison

For a grid with 10,000 combinations:

| Method | Evaluations | Time |
|--------|-------------|------|
| Grid search | 10,000 | ~1 hour |
| Optuna (100 trials) | 100 | ~10 minutes |
| Optuna (50 trials) | 50 | ~5 minutes |

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

## Train/Validate/Test Split Optimization

Quantex now supports ML-style optimization with train/validate/test splits via [`SimpleBacktester.optimize_with_split()`](../../src/quantex/backtester.py:1045).

This approach divides your historical data into three sets:
- **Training set** (default 60%): Used to fit strategy parameters
- **Validation set** (default 20%): Used to select the best parameters
- **Test set** (default 20%): Used for final out-of-sample evaluation

### Why use train/validate/test splits?

Traditional grid search optimizes on the entire dataset, leading to overfitting. With train/validate/test splits:

1. Parameters are optimized on training data
2. Best parameters are selected based on validation performance
3. Final evaluation is performed on held-out test data

### Basic example

```python
from quantex import SimpleBacktester, CSVDataSource, Strategy

class MovingAverageCross(Strategy):
    def __init__(self, fast_period=10, slow_period=30):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def init(self):
        self.add_data(CSVDataSource("data.csv"), "TEST")
        # ... indicator setup ...

    def next(self):
        # ... trading logic ...


backtester = SimpleBacktester(MovingAverageCross(), cash=10_000)

# Grid search with train/validate/test splits
result = backtester.optimize_with_split(
    {
        "fast_period": [5, 10, 15, 20],
        "slow_period": [20, 30, 40, 50],
    },
    constraint=lambda p: p["fast_period"] < p["slow_period"],
    selection_criterion="validate",  # Select best on validation performance
)

print(f"Best parameters: {result.best_params}")
print(f"Train Sharpe: {result.train_metrics['sharpe']:.2f}")
print(f"Validate Sharpe: {result.validate_metrics['sharpe']:.2f}")
print(f"Test Sharpe: {result.test_metrics['sharpe']:.2f}")
```

### Custom split ratios

You can customize the train/validate/test split ratios:

```python
result = backtester.optimize_with_split(
    params,
    train_ratio=0.7,      # 70% training
    validate_ratio=0.15,  # 15% validation
    test_ratio=0.15,       # 15% test
    selection_criterion="validate",
)
```

### Understanding the results

The [`OptimizationResult`](../../src/quantex/backtester.py:124) object contains:

- `best_params`: Dictionary of best parameter values
- `train_report`: Full BacktestReport for training data
- `validate_report`: Full BacktestReport for validation data
- `test_report`: Full BacktestReport for test data
- `train_metrics`: Dict of computed metrics for training
- `validate_metrics`: Dict of computed metrics for validation
- `test_metrics`: Dict of computed metrics for testing
- `all_results`: DataFrame with all parameter combinations

## Gradient Descent Optimization

For continuous parameters, Quantex provides gradient descent optimization via [`SimpleBacktester.optimize_gradient_descent()`](../../src/quantex/backtester.py:1328).

This method uses numerical gradients to iteratively optimize strategy parameters, similar to machine learning workflows.

### Features

- **Numerical gradient computation**: Uses central differences to estimate gradients
- **Momentum-based updates**: Accelerates convergence with momentum
- **Parameter bounds**: Constrains parameters within specified ranges
- **Train/validate/test splits**: Prevents overfitting with built-in cross-validation
- **Convergence detection**: Stops when gradient magnitude falls below tolerance

### Basic example

```python
result = backtester.optimize_gradient_descent(
    param_init={
        'fast_period': 10.0,
        'slow_period': 30.0,
    },
    param_bounds={
        'fast_period': (2.0, 50.0),
        'slow_period': (10.0, 100.0),
    },
    learning_rate=0.05,      # Step size
    momentum=0.9,            # Momentum factor
    max_iterations=50,       # Maximum iterations
    tolerance=1e-6,          # Convergence threshold
)

print(f"Optimized parameters: {result.best_params}")
print(f"Final validation Sharpe: {result.validate_metrics['sharpe']:.2f}")

# View optimization history
print(result.all_results.tail())
```

### Hyperparameters

- `learning_rate`: Step size for gradient updates (default: 0.01)
- `momentum`: Momentum factor for accelerated descent (default: 0.9)
- `max_iterations`: Maximum number of optimization iterations (default: 100)
- `tolerance`: Convergence threshold for gradient magnitude (default: 1e-6)
- `train_ratio`: Fraction of data for training (default: 0.7)
- `validate_ratio`: Fraction of data for validation (default: 0.15)
- `test_ratio`: Fraction of data for testing (default: 0.15)

### Interpreting gradient descent results

The returned `all_results` DataFrame contains optimization history:

```python
# View convergence
history = result.all_results
print(history[['iteration', 'validate_score', 'gradient_magnitude']])
```

## Walk-Forward Optimization

Walk-forward optimization is a rigorous method for evaluating trading strategies that simulates real-world deployment conditions. Unlike train/validate/test splits which use fixed boundaries, walk-forward analysis uses rolling windows to test parameter stability over time.

### How walk-forward works

1. **Training window**: Optimize parameters on a rolling historical window
2. **Testing window**: Evaluate the best parameters on the subsequent out-of-sample period
3. **Slide forward**: Move the window forward and repeat
4. **Aggregate**: Compute statistics across all windows

This approach answers the question: "If I had used this optimization method in the past, how would it have performed on unseen future data?"

### Basic example

```python
from quantex.backtester.walk_forward import WalkForwardAnalyzer, walk_forward_analyze
from quantex import SimpleBacktester, CSVDataSource, Strategy

class MovingAverageCross(Strategy):
    def __init__(self, fast_period=10, slow_period=30):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def init(self):
        self.add_data(CSVDataSource("data.csv"), "TEST")
        # ... indicator setup ...

    def next(self):
        # ... trading logic ...


backtester = SimpleBacktester(MovingAverageCross(), cash=10_000)

# Method 1: Using the convenience function
result = walk_forward_analyze(
    backtester=backtester,
    optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
    params={
        "fast_period": [5, 10, 15],
        "slow_period": [20, 30, 50],
    },
    train_periods=252,   # 1 year training (daily data)
    test_periods=63,     # 3 months testing
    step_periods=63,     # Move forward 3 months each window
    constraint=lambda p: p["fast_period"] < p["slow_period"],
    objective="sharpe",
)

# Method 2: Using the analyzer class directly
analyzer = WalkForwardAnalyzer(
    backtester=backtester,
    train_periods=252,
    test_periods=63,
    step_periods=63,
)

result = analyzer.analyze(
    optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
    params={"fast_period": [5, 10, 15], "slow_period": [20, 30, 50]},
    constraint=lambda p: p["fast_period"] < p["slow_period"],
)
```

### Understanding the results

The [`WalkForwardResult`](../../src/quantex/backtester/walk_forward.py:40) object contains:

- `n_windows`: Number of walk-forward windows
- `window_results`: List of [`WalkForwardWindow`](../../src/quantex/backtester/walk_forward.py:20) objects
- `aggregated_metrics`: Aggregated statistics across all windows
- `all_windows_results_df`: DataFrame with all results

```python
# View aggregated results
print(result)

# Average out-of-sample Sharpe ratio
print(f"Average OOS Sharpe: {result.aggregated_metrics['out_of_sample_sharpe_mean']:.2f}")

# Win rate (% of windows with positive OOS return)
print(f"Win rate: {result.aggregated_metrics['oos_win_rate']:.1%}")

# Stability ratio (higher = more stable parameters)
print(f"Stability ratio: {result.aggregated_metrics['oos_to_is_ratio_mean']:.2f}")
```

### Analyzing parameter stability

Walk-forward analysis reveals how stable your optimized parameters are over time:

```python
# Analyze stability of a specific parameter
stability = result.get_param_stability("fast_period")
print(f"Fast period: mean={stability['mean']:.1f}, std={stability['std']:.1f}")
print(f"Range: [{stability['min']:.1f}, {stability['max']:.1f}]")
print(f"Coefficient of variation: {stability['cv']:.2f}")
```

### Using with different optimizers

Walk-forward analysis works with any optimizer function:

```python
# Grid search
result = walk_forward_analyze(
    backtester=bt,
    optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
    params={"fast": [5, 10, 15], "slow": [20, 30, 50]},
    train_periods=252,
    test_periods=63,
)

# Parallel optimization
result = walk_forward_analyze(
    backtester=bt,
    optimizer=lambda bt, params, **kwargs: bt.optimize_parallel(params, workers=2, **kwargs),
    params={"fast": [5, 10, 15], "slow": [20, 30, 50]},
    train_periods=252,
    test_periods=63,
)

# Optuna Bayesian optimization
result = walk_forward_analyze(
    backtester=bt,
    optimizer=lambda bt, params, **kwargs: bt.optimize_optuna(
        param_space={k: (min(v), max(v)) if len(v) > 2 else v 
                    for k, v in params.items()},
        n_trials=50,
        **kwargs,
    ),
    params={"fast": [5, 10, 15], "slow": [20, 30, 50]},
    train_periods=252,
    test_periods=63,
)
```

### Window configuration tips

- **Train periods**: Should be long enough to capture market cycles (e.g., 1 year for daily data)
- **Test periods**: Should represent realistic holding/deployment periods (e.g., 1-3 months)
- **Step periods**: Controls overlap between windows
  - Equal to test_periods = non-overlapping windows
  - Smaller than test_periods = overlapping windows (more windows, more data)

### Visualizing results

```python
# Plot walk-forward results
result.plot()
```

The plot shows:
1. In-sample vs out-of-sample Sharpe ratios per window
2. In-sample vs out-of-sample returns per window
3. Aggregated Sharpe statistics
4. Summary statistics table

## Helper Functions

Quantex provides utility functions for data splitting:

- [`create_train_validate_test_split()`](../../src/quantex/backtester.py:67): Create train/validate/test indices
- [`DataSplitMode`](../../src/quantex/backtester.py:21): Enum for split modes
- [`TrainValidateTestSplit`](../../src/quantex/backtester.py:28): Dataclass for split configuration

```python
from quantex import create_train_validate_test_split

split = create_train_validate_test_split(
    data_length=1000,
    train_ratio=0.6,
    validate_ratio=0.2,
    test_ratio=0.2,
)

print(f"Train: {split.train_start}-{split.train_end}")
print(f"Validate: {split.validate_start}-{split.validate_end}")
print(f"Test: {split.test_start}-{split.test_end}")
```

## Summary

Quantex provides six optimization approaches:

1. **[`SimpleBacktester.optimize()`](../../src/quantex/backtester/backtester.py:485)**: Sequential grid search
2. **[`SimpleBacktester.optimize_parallel()`](../../src/quantex/backtester/backtester.py:581)**: Parallel grid search with adaptive chunksize
3. **[`SimpleBacktester.optimize_optuna()`](../../src/quantex/backtester/backtester.py:709)**: Bayesian optimization (recommended for large search spaces)
4. **[`SimpleBacktester.optimize_with_split()`](../../src/quantex/backtester/backtester.py:1059)**: Grid search with train/validate/test splits
5. **[`SimpleBacktester.optimize_gradient_descent()`](../../src/quantex/backtester/backtester.py:1336)**: Gradient descent with train/validate/test splits
6. **[`WalkForwardAnalyzer`](../../src/quantex/backtester/walk_forward.py)**: Rolling window optimization with any optimizer

Keep these points in mind:

1. Parameters are applied as plain strategy attributes
2. Constraints are optional but often necessary
3. Train/validate/test splits help prevent overfitting
4. Gradient descent is suitable for continuous parameters
5. **Walk-forward analysis is the most rigorous method** - it simulates real-world deployment
6. **For large search spaces (>10,000 combos), use `optimize_optuna()`** - it's 50-100x faster
7. **For parallel grid search, use `chunksize="auto"`** - reduces IPC overhead

### Performance Tips

| Optimization Method | Best Use Case | Typical Speedup vs Grid |
|--------------------|---------------|------------------------|
| `optimize()` | Small grids, debugging | Baseline |
| `optimize_parallel()` | Medium grids, multi-core | 2-4x |
| `optimize_optuna()` | Large grids, continuous params | 50-100x |
| `optimize_with_split()` | Preventing overfitting | 1x |
| `optimize_gradient_descent()` | Continuous parameters | Varies |
| `WalkForwardAnalyzer` | Real-world simulation | Varies |

For the underlying simulation model, see [Backtesting guide](./backtesting.md).

