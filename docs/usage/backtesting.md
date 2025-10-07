# Backtesting Engine Guide

This guide covers how to use QuantEx's backtesting engine to test your trading strategies against historical data.

## Overview

The `SimpleBacktester` class is the core component for running backtests in QuantEx. It provides:

- **Realistic execution simulation**: Market orders, limit orders, stop losses, take profits
- **Commission modeling**: Percentage-based or fixed cash commissions
- **Performance metrics**: Sharpe ratio, maximum drawdown, total return, and more
- **Progress tracking**: Visual progress bars for long-running backtests
- **Memory efficiency**: Optimized for large datasets

## Basic Backtesting

### Simple Backtest

```python
from quantex import Strategy, SimpleBacktester, CSVDataSource, CommissionType

# Create your strategy
class MyStrategy(Strategy):
    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

    def next(self):
        # Your trading logic here
        pass

# Set up and run backtest
strategy = MyStrategy()
backtester = SimpleBacktester(strategy, cash=10000)
report = backtester.run()

# Display results
print(report)
```

### Backtester Configuration

```python
# Configure backtester with custom settings
backtester = SimpleBacktester(
    strategy,
    cash=50000,                    # Starting capital
    commission=0.002,             # 0.2% commission per trade
    commission_type=CommissionType.PERCENTAGE,  # or CASH
    lot_size=1                    # Lot size multiplier
)

# Run with progress bar disabled for faster execution
report = backtester.run(progress_bar=False)
```

## Understanding Backtest Results

### BacktestReport Structure

The `run()` method returns a `BacktestReport` object with the following attributes:

```python
report = backtester.run()

# Basic information
print(f"Starting Cash: ${report.starting_cash:,.2f}")
print(f"Final Cash: ${report.final_cash:,.2f}")
print(f"Total Return: {report.total_return:.2%}")

# PnL time series (pandas Series)
pnl_series = report.PnlRecord

# All executed orders
orders = report.orders
print(f"Total Trades: {len(orders)}")

# Performance metrics (via __str__ method)
print(report)  # Comprehensive performance summary
```

### Performance Metrics

The backtester automatically calculates several key performance metrics:

```python
# Access detailed metrics
print(f"Total Return: {report.total_return:.2%}")
print(f"Annualized Sharpe Ratio: {report.sharpe:.2f}")
print(f"Maximum Drawdown: {report.max_drawdown:.2%}")
print(f"Number of Trades: {len(report.orders)}")

# Risk-free rate assumption (4% annual)
# Periods per year (inferred from data frequency)
print(f"Periods per Year: {report.periods_per_year}")
```

## Advanced Backtesting Features

### Custom Commission Models

```python
# Percentage-based commission (default)
backtester = SimpleBacktester(
    strategy,
    commission=0.002,  # 0.2%
    commission_type=CommissionType.PERCENTAGE
)

# Fixed cash commission
backtester = SimpleBacktester(
    strategy,
    commission=1.00,  # $1 per trade
    commission_type=CommissionType.CASH
)
```

### Multi-Symbol Backtesting

```python
class MultiSymbolStrategy(Strategy):
    def init(self):
        # Load multiple symbols
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

        for symbol in symbols:
            data = CSVDataSource(f'data/{symbol}.csv')
            self.add_data(data, symbol)

    def next(self):
        # Trade multiple symbols
        for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
            if self.should_trade(symbol):
                self.positions[symbol].buy(0.3)  # 30% allocation each

# Run multi-symbol backtest
strategy = MultiSymbolStrategy()
backtester = SimpleBacktester(strategy, cash=100000)
report = backtester.run()
```

## Parameter Optimization

### Grid Search Optimization

```python
# Define parameter ranges to test
param_ranges = {
    'fast_period': range(5, 21, 2),    # 5, 7, 9, ..., 19
    'slow_period': range(20, 51, 5),   # 20, 25, 30, ..., 50
    'position_size': [0.1, 0.2, 0.3, 0.4, 0.5]
}

# Run optimization
best_params, best_report, results_df = backtester.optimize(
    params=param_ranges
)

print(f"Best Parameters: {best_params}")
print(f"Best Sharpe Ratio: {best_report.sharpe:.2f}")
print(f"Total Combinations Tested: {len(results_df)}")
```

### Optimization with Constraints

```python
# Define parameter constraints
def constraint_func(params):
    """Only test valid parameter combinations"""
    return params['fast_period'] < params['slow_period']

# Run constrained optimization
best_params, best_report, results_df = backtester.optimize(
    params=param_ranges,
    constraint=constraint_func
)
```

### Parallel Optimization

```python
# Use multiple CPU cores for faster optimization
best_params, best_report, results_df = backtester.optimize_parallel(
    params=param_ranges,
    workers=4,           # Use 4 worker processes
    chunksize=10         # Process 10 combinations per chunk
)

# Let backtester choose optimal worker count
best_params, best_report, results_df = backtester.optimize_parallel(
    params=param_ranges,
    workers=None  # Auto-detect optimal worker count
)
```

## Analyzing Backtest Results

### Performance Visualization

```python
import matplotlib.pyplot as plt

# Plot equity curve
plt.figure(figsize=(12, 6))
plt.plot(report.PnlRecord.index, report.PnlRecord.values)
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.show()

# Plot drawdown
def calculate_drawdown(equity_curve):
    """Calculate drawdown series"""
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown

drawdown = calculate_drawdown(report.PnlRecord)
plt.figure(figsize=(12, 6))
plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
plt.plot(drawdown.index, drawdown.values, color='red')
plt.title('Drawdown')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.grid(True)
plt.show()
```

### Trade Analysis

```python
# Analyze individual trades
trades = []

for order in report.orders:
    if order.status == OrderStatus.COMPLETE:
        trades.append({
            'timestamp': order.timestamp,
            'side': order.side,
            'quantity': order.quantity,
            'price': order.price,
            'symbol': 'EURUSD'  # You'd need to track this
        })

trades_df = pd.DataFrame(trades)

# Calculate trade statistics
winning_trades = trades_df[trades_df['pnl'] > 0]
losing_trades = trades_df[trades_df['pnl'] <= 0]

print(f"Win Rate: {len(winning_trades) / len(trades_df):.2%}")
print(f"Average Win: ${winning_trades['pnl'].mean():.2f}")
print(f"Average Loss: ${losing_trades['pnl'].mean():.2f}")
print(f"Profit Factor: {abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()):.2f}")
```

### Risk Metrics

```python
def calculate_risk_metrics(equity_curve, risk_free_rate=0.04):
    """Calculate comprehensive risk metrics"""

    # Daily returns
    daily_returns = equity_curve.pct_change().dropna()

    # Sharpe ratio (already calculated in report)
    sharpe = report.sharpe

    # Maximum drawdown (already calculated in report)
    max_dd = report.max_drawdown

    # Calmar ratio
    calmar = report.total_return / max_dd if max_dd != 0 else np.inf

    # Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    sortino = (daily_returns.mean() - risk_free_rate/252) / downside_std * np.sqrt(252)

    # Value at Risk (95%)
    var_95 = daily_returns.quantile(0.05)

    return {
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'sortino_ratio': sortino,
        'var_95': var_95,
        'volatility': daily_returns.std() * np.sqrt(252)
    }

risk_metrics = calculate_risk_metrics(report.PnlRecord)
for metric, value in risk_metrics.items():
    print(f"{metric}: {value:.4f}")
```

## Backtesting Best Practices

### 1. Walk-Forward Analysis

```python
def walk_forward_analysis(strategy_class, data, train_years=2, test_years=1):
    """Perform walk-forward optimization"""

    results = []
    start_year = data.Index[0].year
    end_year = data.Index[-1].year

    for train_end in range(start_year + train_years, end_year - test_years + 1):
        # Split data
        train_data = data[data.Index.year <= train_end]
        test_data = data[(data.Index.year > train_end) &
                        (data.Index.year <= train_end + test_years)]

        if len(test_data) == 0:
            continue

        # Optimize on training data
        temp_strategy = strategy_class()
        temp_backtester = SimpleBacktester(temp_strategy, cash=10000)

        # You'd need to implement a way to use subset of data
        # This is a simplified example

        results.append({
            'train_end': train_end,
            'test_return': test_return,
            'test_sharpe': test_sharpe
        })

    return pd.DataFrame(results)
```

### 2. Out-of-Sample Testing

```python
# Split your data into in-sample and out-of-sample periods
def split_data_for_oos(data_source, split_ratio=0.7):
    """Split data for out-of-sample testing"""

    total_len = len(data_source)
    split_point = int(total_len * split_ratio)

    # In-sample data (for optimization)
    is_data = type(data_source)(data_source.data.iloc[:split_point])

    # Out-of-sample data (for validation)
    oos_data = type(data_source)(data_source.data.iloc[split_point:])

    return is_data, oos_data

# Use in optimization workflow
is_data, oos_data = split_data_for_oos(data_source)

# Optimize on in-sample data
strategy = MyStrategy()
backtester = SimpleBacktester(strategy, cash=10000)

# Test optimized strategy on out-of-sample data
oos_strategy = MyStrategy(**best_params)
oos_backtester = SimpleBacktester(oos_strategy, cash=10000)
# You'd need to modify backtester to use oos_data
```

### 3. Multiple Time Period Testing

```python
def test_across_periods(strategy_class, data, periods):
    """Test strategy across different market periods"""

    results = []

    for period_name, (start_date, end_date) in periods.items():
        # Filter data for period
        period_data = data[(data.Index >= start_date) & (data.Index <= end_date)]

        if len(period_data) == 0:
            continue

        # Run backtest for this period
        strategy = strategy_class()
        backtester = SimpleBacktester(strategy, cash=10000)

        # Again, simplified - you'd need to modify backtester for data subset

        results.append({
            'period': period_name,
            'return': period_return,
            'sharpe': period_sharpe,
            'max_dd': period_max_dd
        })

    return pd.DataFrame(results)
```

## Handling Backtesting Biases

### Look-Ahead Bias Prevention

```python
class BiasFreeStrategy(Strategy):
    def init(self):
        # Load data
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Initialize indicators with proper lookback
        self.lookback = 50
        self.prices = self.data['EURUSD'].Close

    def next(self):
        # Ensure we have enough data to avoid look-ahead bias
        if len(self.data['EURUSD'].Close) < self.lookback:
            return

        # Use only historical data available at this point
        current_idx = self.data['EURUSD'].current_index

        # Calculate indicator using only data up to current index
        historical_prices = self.prices[:current_idx + 1]
        current_signal = self.calculate_signal(historical_prices)

        # Execute trades based on historical signal
        if current_signal > 0:
            self.positions['EURUSD'].buy(0.5)
```

### Survivorship Bias

```python
# When testing multiple symbols, ensure you include delisted symbols
def load_all_symbols(include_delisted=True):
    """Load all symbols including those that no longer exist"""

    all_symbols = []

    if include_delisted:
        # Load from historical symbol list
        historical_symbols = load_historical_symbol_list()
        for symbol in historical_symbols:
            try:
                data = CSVDataSource(f'data/archive/{symbol}.csv')
                all_symbols.append(data)
            except FileNotFoundError:
                # Symbol was delisted, but we still want to account for it
                print(f"Delisted symbol: {symbol}")
                continue
    else:
        # Only currently active symbols
        active_symbols = load_active_symbol_list()
        for symbol in active_symbols:
            data = CSVDataSource(f'data/{symbol}.csv')
            all_symbols.append(data)

    return all_symbols
```

## Performance Optimization

### Memory Management

```python
# For large datasets, use progress callbacks and memory management
class MemoryEfficientBacktester(SimpleBacktester):
    def run(self, progress_bar=True, chunk_size=10000):
        """Memory-efficient backtest execution"""

        # Process data in chunks
        total_steps = max(len(data) for data in self.strategy.data.values())

        for chunk_start in range(0, total_steps, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_steps)

            # Process this chunk
            for i in range(chunk_start, chunk_end):
                # ... existing backtest logic ...

                # Periodic garbage collection
                if i % 1000 == 0:
                    gc.collect()

        return self._generate_report()
```

### Fast Execution Mode

```python
# Disable progress bar for faster execution
report = backtester.run(progress_bar=False)

# Use parallel optimization for parameter sweeps
best_params, best_report, results_df = backtester.optimize_parallel(
    params=param_ranges,
    workers=os.cpu_count() - 1  # Use all available cores
)
```

## Debugging Backtests

### Logging and Debugging

```python
import logging

class DebuggableStrategy(Strategy):
    def __init__(self):
        super().__init__()
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def next(self):
        # Log key information
        current_price = self.data['EURUSD'].CClose
        position = self.positions['EURUSD'].position

        self.logger.debug(f"Step {self.data['EURUSD'].current_index}: "
                         f"Price={current_price:.5f}, Position={position}")

        # Your trading logic with additional logging
        if self.buy_signal():
            self.logger.info(f"BUY signal at {current_price}")
            self.positions['EURUSD'].buy(0.5)

        elif self.sell_signal():
            self.logger.info(f"SELL signal at {current_price}")
            self.positions['EURUSD'].sell(0.5)
```

### Step-by-Step Debugging

```python
def debug_backtest(strategy, data_source, debug_points=None):
    """Debug backtest at specific points"""

    backtester = SimpleBacktester(strategy, cash=10000)

    # Monkey patch to add debugging
    original_next = strategy.next

    def debug_next():
        current_idx = strategy.data['SYMBOL'].current_index

        if debug_points and current_idx in debug_points:
            print(f"\n=== Debug Point {current_idx} ===")
            print(f"Current Price: {strategy.data['SYMBOL'].CClose}")
            print(f"Position: {strategy.positions['SYMBOL'].position}")
            print(f"Cash: {strategy.positions['SYMBOL'].cash}")
            # Add more debug info as needed

            input("Press Enter to continue...")

        return original_next()

    strategy.next = debug_next
    return backtester.run()
```

## Common Backtesting Pitfalls

### 1. Overfitting

```python
# Avoid overfitting by using out-of-sample testing
def avoid_overfitting(strategy_class, data):
    """Demonstrate overfitting prevention"""

    # Split data
    train_data, test_data = split_data_for_oos(data, split_ratio=0.7)

    # Optimize on training data
    param_ranges = {'period': range(5, 50)}
    backtester = SimpleBacktester(strategy_class(), cash=10000)

    # Test on both training and test data
    # Good performance on both indicates robust strategy
    # Good performance on training but poor on test indicates overfitting
```

### 2. Transaction Costs

```python
# Always include realistic transaction costs
realistic_backtester = SimpleBacktester(
    strategy,
    cash=10000,
    commission=0.002,  # Realistic commission
    commission_type=CommissionType.PERCENTAGE
)

# Compare with zero-cost backtest
zero_cost_backtester = SimpleBacktester(
    strategy,
    cash=10000,
    commission=0.0  # Unrealistic
)

realistic_report = realistic_backtester.run()
zero_cost_report = zero_cost_backtester.run()

print(f"Realistic Sharpe: {realistic_report.sharpe:.2f}")
print(f"Zero-cost Sharpe: {zero_cost_report.sharpe:.2f}")
```

### 3. Data Snooping

```python
# Avoid data snooping by using clear methodology
def proper_testing_methodology():
    """Demonstrate proper testing methodology"""

    # 1. Develop strategy on subset of data/markets
    # 2. Test on different time periods
    # 3. Test on different symbols
    # 4. Use walk-forward analysis
    # 5. Validate with out-of-sample data

    return True
```

## Next Steps

Now that you understand backtesting in QuantEx, explore these related topics:

- **[Strategy Guide](./strategy.md)**: Learn how to create effective trading strategies
- **[Execution Guide](./execution.md)**: Understand order execution and broker simulation
- **[Optimization Guide](./optimizer.md)**: Master parameter optimization techniques

For complete API reference, see the [Backtesting API documentation](../../reference/quantex.backtest.md).