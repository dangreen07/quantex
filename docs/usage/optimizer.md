# Parameter Optimizer

The `quantex.optimizer.Optimizer` class provides a **gradient-descent** helper that systematically tunes hyper-parameters of a `Strategy` by running many backtests under the hood.

## Quick start
```python
from quantex.optimizer import Optimizer
from quantex.sources import ParquetDataSource
from my_strategy import SMACross

# 1. Prepare data sources (any BacktestingDataSource works)
source = ParquetDataSource("btc.parquet", symbol="BTC-USD")

# 2. Instantiate the optimiser – *no simulator required*
#    The Optimizer now defaults to `NextBarSimulator` so you do **not**
#    need to provide an execution layer for simple use-cases.
optimizer = Optimizer(
    strategy_class=SMACross,
    data_sources={"BTC-USD": source},
    strategy_symbols=["BTC-USD"],
)

# 3. Kick off the search
best_params, best_sharpe = optimizer.optimize_gradient_descent(
    initial_params={"window_fast": 20, "window_slow": 50},
    param_bounds={"window_fast": (5, 50), "window_slow": (20, 200)},
    optimization_metric="sharpe_ratio",
)
print(best_params, best_sharpe)

### Speeding up with multithreading
Set `n_jobs` to the number of CPU cores (or ``-1`` for *all* cores) when calling
`optimize_gradient_descent` to evaluate the numerical gradient **concurrently**:

```python
best_params, best_sharpe = optimizer.optimize_gradient_descent(
    initial_params={"window_fast": 20, "window_slow": 50},
    param_bounds={"window_fast": (5, 50), "window_slow": (20, 200)},
    n_jobs=8,  # uses 8 worker threads
)

# Or simply
best_params, best_sharpe = optimizer.optimize_gradient_descent(..., n_jobs=-1)  # all cores
```

Internally a `ThreadPoolExecutor` runs the objective-function calls in parallel.  
Because it stays within one Python process there’s **no pickling overhead**, so it
works out-of-the-box on Windows/macOS/Linux.  If the backtest is CPU-bound you may
still be limited by the GIL, but heavy NumPy/Pandas code often releases the GIL
so you should see meaningful speed-ups.

## Default simulator
If you omit the `simulator_class` argument the optimiser will now fall back to `quantex.execution.NextBarSimulator`.  This eliminates the previous runtime error:
```
Any cannot be instantiated
```

If you need alternative execution logic simply pass a simulator class:
```python
from quantex.execution import ImmediateFillSimulator
optimizer = Optimizer(
    SMACross,
    data_sources,
    strategy_symbols=["BTC-USD"],
    simulator_class=ImmediateFillSimulator,
)
```

## API reference
| Method | Description |
|--------|-------------|
| `optimize_gradient_descent` | Numerical gradient descent tuner (see docstring) | 