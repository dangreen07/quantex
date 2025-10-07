# Backtester API Reference

This page provides comprehensive API documentation for QuantEx's backtesting engine.

## SimpleBacktester Class

The `SimpleBacktester` class is the main component for running backtests in QuantEx.

```python
class SimpleBacktester():
    def __init__(self,
                 strategy: Strategy,
                 cash: float = 10_000,
                 commission: float = 0.002,
                 commission_type: CommissionType = CommissionType.PERCENTAGE,
                 lot_size: int = 1):
        pass
```

### Constructor Parameters

#### `strategy: Strategy`
The strategy instance to backtest.

#### `cash: float = 10_000`
Initial capital for the backtest.

#### `commission: float = 0.002`
Commission rate (0.2% default).

#### `commission_type: CommissionType = CommissionType.PERCENTAGE`
Commission calculation method:
- `CommissionType.PERCENTAGE`: Commission as percentage of trade value
- `CommissionType.CASH`: Fixed cash commission per trade

#### `lot_size: int = 1`
Lot size multiplier for position sizing.

### Instance Methods

#### `run(self, progress_bar: bool = True) -> BacktestReport`

Run the backtest simulation.

```python
def run(self, progress_bar: bool = True) -> BacktestReport:
    """Run backtest simulation"""
    pass
```

**Parameters:**
- `progress_bar` (bool): Show progress bar during backtest

**Returns:**
- `BacktestReport`: Comprehensive backtest results

**Usage:**
```python
# Run with progress bar (default)
report = backtester.run()

# Run without progress bar for faster execution
report = backtester.run(progress_bar=False)
```

#### `optimize(self, params: dict[str, range], constraint: Callable[[dict[str, Any]], bool] | None = None)`

Perform grid search optimization over parameter ranges.

```python
def optimize(self, params: dict[str, range], constraint: Callable[[dict[str, Any]], bool] | None = None):
    """Grid search optimization"""
    pass
```

**Parameters:**
- `params` (dict): Parameter ranges to test
- `constraint` (callable, optional): Function to filter parameter combinations

**Returns:**
- `best_params` (dict): Best parameter combination found
- `best_report` (BacktestReport): Best backtest results
- `results_df` (pd.DataFrame): All optimization results

**Usage:**
```python
# Define parameter ranges
param_ranges = {
    'fast_period': range(5, 21, 2),
    'slow_period': range(20, 51, 5),
    'position_size': [0.1, 0.2, 0.3, 0.4, 0.5]
}

# Run optimization
best_params, best_report, results_df = backtester.optimize(param_ranges)

# With constraints
def constraint_func(params):
    return params['fast_period'] < params['slow_period']

best_params, best_report, results_df = backtester.optimize(
    param_ranges,
    constraint=constraint_func
)
```

#### `optimize_parallel(self, params: dict[str, range], constraint: Callable[[dict[str, Any]], bool] | None = None, workers: int | None = None, chunksize: int = 1)`

Perform parallel grid search optimization.

```python
def optimize_parallel(self,
                     params: dict[str, range],
                     constraint: Callable[[dict[str, Any]], bool] | None = None,
                     workers: int | None = None,
                     chunksize: int = 1):
    """Parallel grid search optimization"""
    pass
```

**Parameters:**
- `params` (dict): Parameter ranges to test
- `constraint` (callable, optional): Function to filter parameter combinations
- `workers` (int, optional): Number of worker processes (auto-detected if None)
- `chunksize` (int): Number of combinations per worker chunk

**Returns:**
- `best_params` (dict): Best parameter combination found
- `best_report` (BacktestReport): Best backtest results
- `results_df` (pd.DataFrame): All optimization results

**Usage:**
```python
# Parallel optimization with custom worker count
best_params, best_report, results_df = backtester.optimize_parallel(
    param_ranges,
    workers=4,
    chunksize=10
)

# Auto-detect optimal worker count
best_params, best_report, results_df = backtester.optimize_parallel(
    param_ranges,
    workers=None  # Let backtester choose
)
```

## BacktestReport Class

Contains comprehensive backtest results and performance metrics.

```python
@dataclass
class BacktestReport:
    starting_cash: np.float64
    final_cash: np.float64
    PnlRecord: pd.Series
    orders: list[Order]
```

### Properties

#### `periods_per_year`

Calculate the number of periods per year based on data frequency.

```python
@property
def periods_per_year(self):
    """Infer periods per year from data frequency"""
    pass
```

**Usage:**
```python
print(f"Periods per year: {report.periods_per_year}")
```

### Instance Methods

#### `__str__(self) -> str`

Return formatted string representation of backtest results.

```python
def __str__(self) -> str:
    """Formatted backtest summary"""
    pass
```

**Usage:**
```python
print(report)  # Comprehensive performance summary
```

## Utility Functions

### `max_drawdown(equity: pd.Series) -> float`

Calculate maximum drawdown from equity curve.

```python
def max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown"""
    pass
```

**Parameters:**
- `equity` (pd.Series): Equity curve time series

**Returns:**
- `float`: Maximum drawdown as positive percentage

**Usage:**
```python
equity_curve = report.PnlRecord
max_dd = max_drawdown(equity_curve)
print(f"Maximum Drawdown: {max_dd:.2%}")
```

### `_infer_periods_per_year(index: pd.Index, default: int = 252 * 24 * 60) -> int`

Infer the number of periods per year from data frequency.

```python
def _infer_periods_per_year(index: pd.Index, default: int = 252 * 24 * 60) -> int:
    """Infer periods per year from datetime index"""
    pass
```

**Parameters:**
- `index` (pd.Index): Datetime index
- `default` (int): Default value for uncertain frequencies

**Returns:**
- `int`: Inferred periods per year

## Complete Backtesting Example

```python
from quantex import Strategy, SimpleBacktester, CSVDataSource, CommissionType
import pandas as pd
import numpy as np

class OptimizedStrategy(Strategy):
    """Strategy with multiple optimizable parameters"""

    def __init__(self, fast_period=10, slow_period=30, position_size=0.1,
                 stop_loss=None, enable_trailing_stop=False):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.enable_trailing_stop = enable_trailing_stop

    def init(self):
        # Load data
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Create indicators
        close_prices = self.data['EURUSD'].Close
        self.fast_ma = self.Indicator(self.sma(close_prices, self.fast_period))
        self.slow_ma = self.Indicator(self.sma(close_prices, self.slow_period))

    def next(self):
        if len(self.fast_ma) < 2 or len(self.slow_ma) < 2:
            return

        # Generate signals
        if self.generate_buy_signal():
            self.positions['EURUSD'].buy(
                self.position_size,
                stop_loss=self.stop_loss
            )
        elif self.generate_sell_signal():
            self.positions['EURUSD'].sell(self.position_size)

    def generate_buy_signal(self):
        """Generate buy signal based on indicators"""
        return (self.fast_ma[-1] > self.slow_ma[-1] and
                self.fast_ma[-2] <= self.slow_ma[-2])

    def generate_sell_signal(self):
        """Generate sell signal based on indicators"""
        return (self.fast_ma[-1] < self.slow_ma[-1] and
                self.fast_ma[-2] >= self.slow_ma[-2])

    def sma(self, prices, period):
        return pd.Series(prices).rolling(window=period).mean().values

# Set up backtester
strategy = OptimizedStrategy()
backtester = SimpleBacktester(
    strategy,
    cash=100000,                    # $100k starting capital
    commission=0.002,              # 0.2% commission
    commission_type=CommissionType.PERCENTAGE,
    lot_size=1
)

# Define comprehensive parameter ranges
param_ranges = {
    'fast_period': range(5, 21, 2),      # 5, 7, 9, ..., 19
    'slow_period': range(20, 51, 5),     # 20, 25, 30, ..., 50
    'position_size': np.linspace(0.05, 0.5, 10),  # 0.05 to 0.5 in 0.05 steps
    'stop_loss': [None, -0.01, -0.02, -0.03, -0.05]  # No stop or 1-5% stops
}

# Run parallel optimization
print("Running parallel optimization...")
best_params, best_report, results_df = backtester.optimize_parallel(
    params=param_ranges,
    workers=4,
    chunksize=5
)

# Display results
print(f"\nBest Parameters: {best_params}")
print(f"Best Sharpe Ratio: {best_report.sharpe:.2f}")
print(f"Best Total Return: {best_report.total_return:.2%}")
print(f"Maximum Drawdown: {best_report.max_drawdown:.2%}")

# Analyze optimization results
print(f"\nOptimization Summary:")
print(f"Total combinations tested: {len(results_df)}")
print(f"Best 5 Sharpe ratios:")
print(results_df.nlargest(5, 'sharpe')[['fast_period', 'slow_period',
                                       'position_size', 'sharpe', 'total_return']])

# Run final backtest with best parameters
final_strategy = OptimizedStrategy(**best_params)
final_backtester = SimpleBacktester(final_strategy, cash=100000)
final_report = final_backtester.run()

print(f"\nFinal Backtest Results:")
print(final_report)
```

## Advanced Optimization Techniques

### Custom Optimization Metrics

```python
def custom_optimization(backtester, param_ranges, custom_metric_func):
    """Run optimization with custom performance metric"""

    def evaluate_combination(params):
        """Evaluate single parameter combination"""
        strategy = OptimizedStrategy(**params)
        test_backtester = SimpleBacktester(strategy, cash=10000)

        try:
            report = test_backtester.run(progress_bar=False)
            return custom_metric_func(report)
        except Exception as e:
            return float('-inf')  # Invalid combination

    # Test all combinations
    results = []

    for fast in param_ranges['fast_period']:
        for slow in param_ranges['slow_period']:
            for pos_size in param_ranges['position_size']:
                params = {
                    'fast_period': fast,
                    'slow_period': slow,
                    'position_size': pos_size
                }

                score = evaluate_combination(params)
                results.append({**params, 'score': score})

    # Find best parameters
    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df['score'].idxmax()]

    return best_row.drop('score').to_dict(), results_df

# Custom metric: Sharpe ratio with drawdown penalty
def custom_metric(report):
    """Custom performance metric"""
    if report.sharpe == float('nan') or report.max_drawdown == 0:
        return float('-inf')

    # Penalize high drawdowns
    drawdown_penalty = report.max_drawdown * 2
    adjusted_sharpe = report.sharpe - drawdown_penalty

    return adjusted_sharpe

# Use custom optimization
best_params, results_df = custom_optimization(backtester, param_ranges, custom_metric)
```

### Multi-Stage Optimization

```python
def multi_stage_optimization(backtester, param_ranges):
    """Multi-stage optimization process"""

    # Stage 1: Coarse optimization
    print("Stage 1: Coarse optimization...")
    coarse_ranges = {
        'fast_period': range(5, 21, 5),    # Wider steps
        'slow_period': range(20, 51, 10),  # Wider steps
        'position_size': [0.1, 0.3, 0.5]   # Fewer options
    }

    best_coarse, _, coarse_results = backtester.optimize(coarse_ranges)

    # Stage 2: Fine optimization around best coarse parameters
    print("Stage 2: Fine optimization...")
    fine_ranges = {
        'fast_period': range(max(5, best_coarse['fast_period']-4),
                           best_coarse['fast_period']+5, 1),
        'slow_period': range(max(20, best_coarse['slow_period']-9),
                           best_coarse['slow_period']+10, 2),
        'position_size': np.linspace(max(0.05, best_coarse['position_size']-0.1),
                                   best_coarse['position_size']+0.1, 5)
    }

    best_fine, best_report, fine_results = backtester.optimize(fine_ranges)

    return best_fine, best_report, [coarse_results, fine_results]
```

## Performance Analysis

### Detailed Performance Metrics

```python
def analyze_backtest_performance(report):
    """Comprehensive performance analysis"""

    # Basic metrics
    total_return = report.total_return
    sharpe_ratio = report.sharpe
    max_drawdown = report.max_drawdown

    # Additional calculations
    equity_curve = report.PnlRecord
    daily_returns = equity_curve.pct_change().dropna()

    # Volatility
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized

    # Win rate and profit factor
    winning_trades = [order for order in report.orders if self.calculate_trade_pnl(order) > 0]
    losing_trades = [order for order in report.orders if self.calculate_trade_pnl(order) < 0]

    win_rate = len(winning_trades) / len(report.orders) if report.orders else 0

    total_wins = sum(self.calculate_trade_pnl(order) for order in winning_trades)
    total_losses = abs(sum(self.calculate_trade_pnl(order) for order in losing_trades))
    profit_factor = total_wins / total_losses if total_losses != 0 else float('inf')

    # Calmar ratio
    calmar_ratio = total_return / max_drawdown if max_drawdown != 0 else float('inf')

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'calmar_ratio': calmar_ratio,
        'total_trades': len(report.orders)
    }

# Usage
performance = analyze_backtest_performance(best_report)
for metric, value in performance.items():
    print(f"{metric}: {value:.4f}")
```

### Risk Analysis

```python
def analyze_risk_metrics(equity_curve, risk_free_rate=0.04):
    """Comprehensive risk analysis"""

    daily_returns = equity_curve.pct_change().dropna()

    # Downside deviation (for Sortino ratio)
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)

    # Value at Risk (95%)
    var_95 = daily_returns.quantile(0.05)

    # Expected Shortfall (Conditional VaR)
    cvar_95 = daily_returns[daily_returns <= var_95].mean()

    # Maximum drawdown duration
    peak = equity_curve.expanding().max()
    drawdown = equity_curve / peak - 1

    # Drawdown duration (time in drawdown)
    in_drawdown = drawdown < 0
    drawdown_duration = self.calculate_drawdown_duration(in_drawdown)

    return {
        'volatility': daily_returns.std() * np.sqrt(252),
        'downside_deviation': downside_deviation,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'max_drawdown_duration': drawdown_duration,
        'worst_day': daily_returns.min(),
        'best_day': daily_returns.max()
    }
```

## Backtesting Workflow

### Complete Workflow Example

```python
class BacktestingWorkflow:
    """Complete backtesting and optimization workflow"""

    def __init__(self, strategy_class, data_path):
        self.strategy_class = strategy_class
        self.data_path = data_path

    def run_complete_workflow(self):
        """Run complete backtesting workflow"""

        # 1. Initial parameter screening
        print("Phase 1: Initial screening...")
        initial_results = self.initial_screening()

        # 2. Coarse optimization
        print("Phase 2: Coarse optimization...")
        coarse_results = self.coarse_optimization()

        # 3. Fine optimization
        print("Phase 3: Fine optimization...")
        fine_results = self.fine_optimization()

        # 4. Out-of-sample validation
        print("Phase 4: Out-of-sample validation...")
        oos_results = self.out_of_sample_validation()

        # 5. Generate comprehensive report
        print("Phase 5: Generating report...")
        report = self.generate_workflow_report([
            initial_results, coarse_results, fine_results, oos_results
        ])

        return report

    def initial_screening(self):
        """Initial parameter screening"""
        # Implementation
        return {}

    def coarse_optimization(self):
        """Coarse parameter optimization"""
        # Implementation
        return {}

    def fine_optimization(self):
        """Fine parameter optimization"""
        # Implementation
        return {}

    def out_of_sample_validation(self):
        """Out-of-sample validation"""
        # Implementation
        return {}

    def generate_workflow_report(self, results):
        """Generate comprehensive workflow report"""
        # Implementation
        return {}
```

## Error Handling and Debugging

### Common Backtesting Errors

```python
class BacktestingError(Exception):
    """Base exception for backtesting errors"""
    pass

class OptimizationError(BacktestingError):
    """Raised when optimization fails"""
    pass

class InsufficientDataError(BacktestingError):
    """Raised when insufficient data for backtest"""
    pass

# Error handling in backtester
try:
    report = backtester.run()
except InsufficientDataError as e:
    print(f"Insufficient data: {e}")
    # Handle insufficient data case
except OptimizationError as e:
    print(f"Optimization error: {e}")
    # Handle optimization failure
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected errors
```

### Debugging Support

```python
def debug_backtest(backtester, debug_points=None):
    """Debug backtest execution"""

    original_run = backtester.run

    def debug_run(progress_bar=True):
        # Monkey patch strategy for debugging
        original_next = backtester.strategy.next

        def debug_next():
            current_idx = backtester.strategy.data['SYMBOL'].current_index

            if debug_points and current_idx in debug_points:
                print(f"\n=== Debug Point {current_idx} ===")
                print(f"Price: {backtester.strategy.data['SYMBOL'].CClose}")
                print(f"Position: {backtester.strategy.positions['SYMBOL'].position}")
                input("Press Enter to continue...")

            return original_next()

        backtester.strategy.next = debug_next
        return original_run(progress_bar=False)

    backtester.run = debug_run
    return backtester
```

## Best Practices

### 1. Parameter Range Selection

```python
def select_optimal_ranges(data_source, strategy_class):
    """Select optimal parameter ranges based on data characteristics"""

    # Analyze data characteristics
    close_prices = data_source.Close
    data_length = len(close_prices)
    price_series = pd.Series(close_prices)

    # Adaptive range selection
    max_period = min(data_length // 20, 200)  # Max 5% of data length

    ranges = {
        'fast_period': range(5, min(max_period // 4, 50), 2),
        'slow_period': range(max(20, max_period // 10), min(max_period, 200), 5),
        'position_size': np.linspace(0.05, 0.5, 10)
    }

    return ranges
```

### 2. Computational Efficiency

```python
def optimize_efficiently(backtester, param_ranges, max_combinations=5000):
    """Optimize with computational constraints"""

    # Calculate total combinations
    total_combos = np.prod([len(list(values)) for values in param_ranges.values()])

    if total_combos > max_combinations:
        print(f"Too many combinations ({total_combos}). Reducing ranges...")

        # Reduce ranges proportionally
        reduction_factor = max_combinations / total_combos

        optimized_ranges = {}
        for param, values in param_ranges.items():
            values_list = list(values)
            step = max(1, int(1 / reduction_factor))
            optimized_ranges[param] = values_list[::step]

        return optimized_ranges

    return param_ranges
```

## Related Documentation

- **[Usage Guide](../../../usage/backtesting.md)**: Learn how to use the backtester
- **[Strategy API](../../quantex.strategy.md)**: Understand strategy development
- **[Optimization Guide](../../../usage/optimizer.md)**: Master parameter optimization
- **[Execution API](../../quantex.execution.md)**: Handle order execution