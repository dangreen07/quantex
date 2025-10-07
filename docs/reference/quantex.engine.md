# Engine API Reference

This page provides comprehensive API documentation for QuantEx's backtesting engine components.

## SimpleBacktester Class (Continued)

Additional methods and internal functionality of the SimpleBacktester class.

### Internal Methods

#### `_generate_report(self) -> BacktestReport`

Generate final backtest report from simulation results.

```python
def _generate_report(self) -> BacktestReport:
    """Generate comprehensive backtest report"""
    pass
```

**Returns:**
- `BacktestReport`: Complete backtest results

### Worker Functions (Parallel Optimization)

#### `_worker_init(pickled_strategy: bytes, cash: float, commission: float, commission_type, lot_size: int)`

Initialize worker processes for parallel optimization.

```python
def _worker_init(pickled_strategy: bytes, cash: float, commision: float,
                 commision_type, lot_size: int):
    """Initialize worker processes with strategy and configuration"""
    pass
```

**Parameters:**
- `pickled_strategy` (bytes): Serialized strategy for worker
- `cash` (float): Starting capital
- `commission` (float): Commission rate
- `commission_type`: Commission calculation method
- `lot_size` (int): Lot size multiplier

#### `_worker_eval(param_items)`

Evaluate parameter combination in worker process.

```python
def _worker_eval(param_items):
    """Evaluate parameter combination in worker process"""
    pass
```

**Parameters:**
- `param_items`: Parameter dictionary items

**Returns:**
- `dict`: Performance metrics for parameter combination

## Backtesting Engine Workflow

### Execution Flow

```python
# 1. Strategy initialization
def run_backtest(strategy, data, config):
    """Complete backtest execution flow"""

    # Initialize backtester
    backtester = SimpleBacktester(
        strategy=strategy,
        cash=config['cash'],
        commission=config['commission'],
        commission_type=config['commission_type']
    )

    # Run simulation
    report = backtester.run(progress_bar=config['show_progress'])

    return report

# 2. Parameter optimization flow
def run_optimization(strategy_class, param_ranges, config):
    """Complete optimization workflow"""

    # Create base strategy
    base_strategy = strategy_class()

    # Initialize backtester
    backtester = SimpleBacktester(
        strategy=base_strategy,
        cash=config['cash'],
        commission=config['commission']
    )

    # Run optimization
    if config['parallel']:
        best_params, best_report, results_df = backtester.optimize_parallel(
            params=param_ranges,
            workers=config['workers'],
            chunksize=config['chunksize']
        )
    else:
        best_params, best_report, results_df = backtester.optimize(
            params=param_ranges,
            constraint=config.get('constraint')
        )

    return best_params, best_report, results_df
```

## Engine Configuration

### Backtester Configuration Options

```python
class BacktesterConfig:
    """Configuration options for backtester"""

    def __init__(self,
                 cash: float = 10000,
                 commission: float = 0.002,
                 commission_type: CommissionType = CommissionType.PERCENTAGE,
                 lot_size: int = 1,
                 parallel: bool = True,
                 workers: int = None,
                 chunksize: int = 1,
                 progress_bar: bool = True,
                 memory_limit: int = 1024):  # MB

        self.cash = cash
        self.commission = commission
        self.commission_type = commission_type
        self.lot_size = lot_size
        self.parallel = parallel
        self.workers = workers or (os.cpu_count() - 1)
        self.chunksize = chunksize
        self.progress_bar = progress_bar
        self.memory_limit = memory_limit

# Usage
config = BacktesterConfig(
    cash=50000,
    commission=0.001,
    parallel=True,
    workers=4,
    memory_limit=2048
)

backtester = SimpleBacktester(
    strategy=strategy,
    **config.__dict__  # Unpack configuration
)
```

## Memory Management

### Large Dataset Handling

```python
class LargeDatasetBacktester(SimpleBacktester):
    """Backtester optimized for large datasets"""

    def __init__(self, *args, chunk_size=50000, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

    def run(self, progress_bar=True):
        """Run backtest with memory management"""

        # Process data in chunks
        total_steps = max(len(data) for data in self.strategy.data.values())

        for chunk_start in range(0, total_steps, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_steps)

            # Process chunk
            for i in range(chunk_start, chunk_end):
                # Standard backtest logic
                for data_source in self.strategy.data.values():
                    data_source.current_index = i

                for position in self.strategy.positions.values():
                    position._iterate(i)

                for indicator in self.strategy.indicators:
                    indicator._i = i

                self.strategy.next()

            # Memory cleanup
            if chunk_start % (self.chunk_size * 10) == 0:
                import gc
                gc.collect()

        return self._generate_report()
```

### Memory-Efficient Optimization

```python
def memory_efficient_optimization(backtester, param_ranges, max_memory_mb=1024):
    """Run optimization with memory constraints"""

    # Estimate memory usage
    estimated_memory = estimate_optimization_memory(param_ranges)

    if estimated_memory > max_memory_mb:
        print(f"Optimization requires {estimated_memory}MB, limit is {max_memory_mb}MB")
        print("Using batch optimization...")

        return run_batch_optimization(backtester, param_ranges, max_memory_mb)
    else:
        return backtester.optimize_parallel(param_ranges)

def estimate_optimization_memory(param_ranges):
    """Estimate memory requirements for optimization"""

    # Calculate total combinations
    total_combos = 1
    for values in param_ranges.values():
        total_combos *= len(list(values))

    # Estimate memory per combination (rough)
    memory_per_combo = 1  # MB

    return total_combos * memory_per_combo

def run_batch_optimization(backtester, param_ranges, max_memory_mb):
    """Run optimization in batches"""

    # Split parameter ranges into batches
    batches = split_parameter_ranges(param_ranges, max_memory_mb)

    all_results = []

    for batch in batches:
        print(f"Processing batch with {len(batch)} combinations...")

        # Run optimization on batch
        _, _, batch_results = backtester.optimize(batch)

        all_results.append(batch_results)

        # Memory cleanup
        import gc
        gc.collect()

    # Combine results
    return pd.concat(all_results, ignore_index=True)
```

## Performance Monitoring

### Backtest Performance Metrics

```python
def monitor_backtest_performance(backtester, report):
    """Monitor and analyze backtest performance"""

    # Execution time analysis
    start_time = time.time()
    report = backtester.run()
    end_time = time.time()

    execution_time = end_time - start_time

    # Memory usage
    import psutil
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB

    # Performance metrics
    data_points = sum(len(data) for data in backtester.strategy.data.values())
    total_steps = max(len(data) for data in backtester.strategy.data.values())

    print("Backtest Performance:")
    print(f"  Execution Time: {execution_time:.2f} seconds")
    print(f"  Memory Usage: {memory_usage:.1f} MB")
    print(f"  Data Points: {data_points:,}")
    print(f"  Time Steps: {total_steps:,}")
    print(f"  Points/Second: {data_points/execution_time:.0f}")
    print(f"  Final Return: {report.total_return:.2%}")

    return {
        'execution_time': execution_time,
        'memory_usage': memory_usage,
        'data_points': data_points,
        'total_return': report.total_return
    }
```

### Optimization Performance

```python
def monitor_optimization_performance(backtester, param_ranges, results_df):
    """Monitor optimization performance"""

    # Calculate optimization metrics
    total_combinations = len(results_df)
    execution_time = getattr(backtester, '_optimization_time', 0)

    if execution_time == 0:
        return {}

    # Performance per combination
    time_per_combination = execution_time / total_combinations

    # Best result timing
    best_result = results_df.loc[results_df['sharpe'].idxmax()]
    best_time = getattr(best_result, 'evaluation_time', time_per_combination)

    print("Optimization Performance:")
    print(f"  Total Combinations: {total_combinations:,}")
    print(f"  Total Time: {execution_time:.2f} seconds")
    print(f"  Time per Combination: {time_per_combination:.4f} seconds")
    print(f"  Combinations/Second: {total_combinations/execution_time:.1f}")
    print(f"  Best Sharpe: {best_result['sharpe']:.2f}")
    print(f"  Efficiency: {best_result['sharpe']/time_per_combination:.4f} Sharpe/second")

    return {
        'total_combinations': total_combinations,
        'execution_time': execution_time,
        'time_per_combination': time_per_combination,
        'best_sharpe': best_result['sharpe']
    }
```

## Error Handling and Recovery

### Backtest Error Recovery

```python
class RobustBacktester(SimpleBacktester):
    """Backtester with error handling and recovery"""

    def __init__(self, *args, max_retries=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.error_log = []

    def run(self, progress_bar=True):
        """Run backtest with error recovery"""

        try:
            return super().run(progress_bar)
        except Exception as e:
            self.error_log.append(f"Backtest error: {e}")

            # Attempt recovery
            if self.max_retries > 0:
                print(f"Attempting recovery (retries left: {self.max_retries})")
                self.max_retries -= 1

                # Reset strategy state
                self.reset_strategy_state()

                # Retry
                return self.run(progress_bar)
            else:
                raise BacktestingError(f"Backtest failed after retries: {e}")

    def reset_strategy_state(self):
        """Reset strategy to clean state"""
        # Reset positions
        for position in self.strategy.positions.values():
            position.position = 0
            position.cash = self.cash
            position.orders = []
            position.complete_orders = []

        # Reset indicators
        for indicator in self.strategy.indicators:
            indicator._i = 0

        # Reset PnL record
        for position in self.strategy.positions.values():
            position.PnLRecord = pd.Series([position.cash] * len(position.source.data['Close']),
                                         index=position.source.data['Close'].index,
                                         dtype=np.float64)
```

### Strategy Error Handling

```python
def safe_strategy_execution(strategy, data_sources):
    """Execute strategy with comprehensive error handling"""

    try:
        # Validate strategy
        if not hasattr(strategy, 'init') or not hasattr(strategy, 'next'):
            raise StrategyError("Strategy missing required methods")

        # Validate data sources
        for symbol, data_source in data_sources.items():
            if not hasattr(data_source, 'Close') or not hasattr(data_source, 'CClose'):
                raise DataError(f"Invalid data source for {symbol}")

        # Execute strategy
        strategy.init()

        # Run through data
        max_steps = max(len(ds) for ds in data_sources.values())

        for i in range(max_steps):
            try:
                # Update data sources
                for data_source in data_sources.values():
                    data_source.current_index = i

                # Execute strategy step
                strategy.next()

            except Exception as e:
                print(f"Error in step {i}: {e}")
                # Decide whether to continue or stop
                if is_critical_error(e):
                    raise

    except StrategyError as e:
        print(f"Strategy configuration error: {e}")
        raise

    except DataError as e:
        print(f"Data error: {e}")
        raise

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
```

## Advanced Engine Features

### Custom Backtest Metrics

```python
class CustomMetricsBacktester(SimpleBacktester):
    """Backtester with custom performance metrics"""

    def __init__(self, *args, custom_metrics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_metrics = custom_metrics or []

    def _generate_report(self):
        """Generate report with custom metrics"""

        # Get standard report
        report = super()._generate_report()

        # Add custom metrics
        for metric_func in self.custom_metrics:
            metric_name = metric_func.__name__
            metric_value = metric_func(report)

            # Add to report (you'd need to modify BacktestReport for this)
            setattr(report, metric_name, metric_value)

        return report

# Usage
def profit_factor_metric(report):
    """Calculate profit factor"""
    winning_trades = [o for o in report.orders if calculate_trade_pnl(o) > 0]
    losing_trades = [o for o in report.orders if calculate_trade_pnl(o) < 0]

    total_wins = sum(calculate_trade_pnl(o) for o in winning_trades)
    total_losses = abs(sum(calculate_trade_pnl(o) for o in losing_trades))

    return total_wins / total_losses if total_losses > 0 else float('inf')

def max_consecutive_losses_metric(report):
    """Calculate maximum consecutive losses"""
    # Implementation here
    return 0

# Use custom metrics
custom_backtester = CustomMetricsBacktester(
    strategy,
    custom_metrics=[profit_factor_metric, max_consecutive_losses_metric]
)

report = custom_backtester.run()
```

### Multi-Strategy Engine

```python
class MultiStrategyEngine:
    """Engine for running multiple strategies"""

    def __init__(self, strategies, common_data):
        self.strategies = strategies
        self.common_data = common_data
        self.results = {}

    def run_all_strategies(self, config):
        """Run all strategies with same configuration"""

        for name, strategy_class in self.strategies.items():
            print(f"Running strategy: {name}")

            try:
                # Create strategy instance
                strategy = strategy_class()

                # Add common data
                for symbol, data in self.common_data.items():
                    strategy.add_data(data, symbol)

                # Create backtester
                backtester = SimpleBacktester(strategy, **config)

                # Run backtest
                report = backtester.run(progress_bar=False)

                self.results[name] = report

                print(f"  Completed: {report.total_return:.2%} return")

            except Exception as e:
                print(f"  Error: {e}")
                self.results[name] = None

        return self.results

    def compare_strategies(self):
        """Compare performance across strategies"""

        comparison = []

        for name, report in self.results.items():
            if report is not None:
                comparison.append({
                    'strategy': name,
                    'total_return': report.total_return,
                    'sharpe_ratio': report.sharpe,
                    'max_drawdown': report.max_drawdown,
                    'total_trades': len(report.orders)
                })

        return pd.DataFrame(comparison)
```

## Integration with External Systems

### Database Integration

```python
class DatabaseBacktester(SimpleBacktester):
    """Backtester with database integration"""

    def __init__(self, *args, db_connection=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_connection = db_connection

    def _generate_report(self):
        """Generate report and save to database"""

        report = super()._generate_report()

        if self.db_connection:
            self.save_report_to_db(report)

        return report

    def save_report_to_db(self, report):
        """Save backtest results to database"""

        # Implementation would save to database
        # This is a conceptual example

        cursor = self.db_connection.cursor()

        # Save basic metrics
        cursor.execute("""
            INSERT INTO backtest_results
            (timestamp, strategy, total_return, sharpe_ratio, max_drawdown)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            self.strategy.__class__.__name__,
            report.total_return,
            report.sharpe,
            report.max_drawdown
        ))

        # Save detailed results
        for order in report.orders:
            cursor.execute("""
                INSERT INTO backtest_orders
                (backtest_id, side, quantity, price, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                cursor.lastrowid,
                order.side.value,
                order.quantity,
                order.price,
                order.timestamp
            ))

        self.db_connection.commit()
```

### Real-time Data Integration

```python
class RealTimeBacktester(SimpleBacktester):
    """Backtester with real-time data capabilities"""

    def __init__(self, *args, data_stream=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_stream = data_stream

    def run_with_streaming_data(self):
        """Run backtest with streaming data updates"""

        # Initialize with historical data
        report = self.run()

        if self.data_stream:
            # Continue with real-time data
            self.run_real_time_extension()

        return report

    def run_real_time_extension(self):
        """Extend backtest with real-time data"""

        for real_time_bar in self.data_stream:
            # Update data sources with new bar
            for symbol, data_source in self.strategy.data.items():
                # Add new bar to data source
                self.add_real_time_bar(data_source, real_time_bar)

            # Execute strategy for new bar
            self.strategy.next()

            # Update PnL record
            self.update_pnl_record()
```

## Best Practices

### 1. Engine Configuration

```python
def create_optimized_backtester(strategy, data_characteristics):
    """Create optimally configured backtester"""

    # Analyze data characteristics
    data_length = max(len(ds) for ds in strategy.data.values())
    data_frequency = infer_data_frequency(strategy.data)

    # Configure based on data size
    if data_length > 100000:  # Large dataset
        config = {
            'progress_bar': False,  # Faster execution
            'chunk_size': 50000,    # Memory management
            'parallel': True,       # Use multiple cores
            'workers': min(4, os.cpu_count() - 1)
        }
    else:  # Smaller dataset
        config = {
            'progress_bar': True,
            'chunk_size': None,
            'parallel': False,
            'workers': 1
        }

    return SimpleBacktester(strategy, **config)
```

### 2. Performance Optimization

```python
def optimize_engine_performance(backtester):
    """Optimize backtester performance"""

    # Profile execution
    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()

    report = backtester.run()

    profiler.disable()
    profiler.print_stats(sort='cumulative')

    # Based on profiling results, optimize bottlenecks
    # - Use faster data structures
    # - Optimize indicator calculations
    # - Reduce memory allocations
    # - Use parallel processing where appropriate

    return report
```

### 3. Resource Management

```python
class ResourceManagedBacktester(SimpleBacktester):
    """Backtester with resource management"""

    def __init__(self, *args, memory_limit_mb=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_limit_mb = memory_limit_mb

    def run(self, progress_bar=True):
        """Run with resource monitoring"""

        import psutil
        process = psutil.Process()

        # Monitor memory usage
        def check_memory_usage():
            memory_usage = process.memory_info().rss / 1024 / 1024

            if memory_usage > self.memory_limit_mb:
                print(f"Memory usage ({memory_usage:.1f}MB) exceeds limit ({self.memory_limit_mb}MB)")
                print("Forcing garbage collection...")

                import gc
                gc.collect()

                # If still high, reduce chunk size or switch to serial processing
                if process.memory_info().rss / 1024 / 1024 > self.memory_limit_mb:
                    print("Still high memory usage. Switching to memory-efficient mode.")

        # Run with memory monitoring
        original_run = super().run

        def monitored_run():
            # Monkey patch for monitoring
            original_iterate = None

            for position in self.strategy.positions.values():
                original_iterate = position._iterate

                def monitored_iterate(current_index):
                    check_memory_usage()
                    return original_iterate(current_index)

                position._iterate = monitored_iterate

            return original_run(progress_bar)

        return monitored_run()
```

## Related Documentation

- **[Backtester Usage Guide](../../../usage/backtesting.md)**: Learn how to use the backtesting engine
- **[Strategy API](../../quantex.strategy.md)**: Understand strategy development
- **[Optimization Guide](../../../usage/optimizer.md)**: Master parameter optimization
- **[Execution API](../../quantex.execution.md)**: Handle order execution