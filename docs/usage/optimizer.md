# Parameter Optimization Guide

This guide covers how to use QuantEx's parameter optimization features to find the best strategy parameters.

## Overview

QuantEx provides powerful optimization capabilities:

- **Grid Search**: Test all combinations of parameter values
- **Parallel Processing**: Use multiple CPU cores for faster optimization
- **Constraint Support**: Apply business logic constraints to parameter combinations
- **Memory Efficient**: Optimized for large parameter spaces
- **Comprehensive Results**: Detailed analysis of optimization results

## Basic Parameter Optimization

### Simple Grid Search

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

### Understanding Optimization Results

```python
# Analyze optimization results
print("Top 5 parameter combinations:")
print(results_df.head())

# Summary statistics
print(f"\nBest Sharpe: {results_df['sharpe'].max():.2f}")
print(f"Worst Sharpe: {results_df['sharpe'].min():.2f}")
print(f"Mean Sharpe: {results_df['sharpe'].mean():.2f}")
print(f"Std Sharpe: {results_df['sharpe'].std():.2f}")

# Distribution of results
print(f"\nPositive Sharpe combinations: {len(results_df[results_df['sharpe'] > 0])}")
print(f"Negative Sharpe combinations: {len(results_df[results_df['sharpe'] < 0])}")
```

## Advanced Optimization Features

### Parameter Constraints

```python
# Define logical constraints between parameters
def optimization_constraints(params):
    """Apply business logic constraints"""

    # Fast period should be less than slow period
    if params['fast_period'] >= params['slow_period']:
        return False

    # Position size should be reasonable
    if not (0.01 <= params['position_size'] <= 1.0):
        return False

    # Custom logic constraints
    if params['fast_period'] * 2 > params['slow_period']:
        return False

    return True

# Run constrained optimization
best_params, best_report, results_df = backtester.optimize(
    params=param_ranges,
    constraint=optimization_constraints
)
```

### Custom Parameter Types

```python
# Use different parameter types
advanced_param_ranges = {
    'fast_period': range(5, 21, 2),                    # Integer range
    'slow_period': [20, 25, 30, 35, 40, 45, 50],      # Specific values
    'position_size': np.linspace(0.1, 0.5, 9),        # Float range
    'stop_loss': np.logspace(-0.02, -0.001, 10),       # Log-spaced values
    'rsi_period': [7, 10, 14, 21, 28],                # Fibonacci-like
    'bb_std': [1.5, 1.8, 2.0, 2.2, 2.5]              # Standard deviations
}

# Generate parameter combinations manually for complex cases
import itertools

def generate_custom_combinations():
    """Generate custom parameter combinations"""

    combinations = []

    for fast in range(5, 21, 2):
        for slow in [20, 25, 30, 35, 40, 45, 50]:
            if fast < slow:  # Apply constraint
                for pos_size in np.linspace(0.1, 0.5, 5):
                    for stop_loss in [-0.01, -0.02, -0.03]:
                        combinations.append({
                            'fast_period': fast,
                            'slow_period': slow,
                            'position_size': pos_size,
                            'stop_loss': stop_loss
                        })

    return combinations
```

## Parallel Optimization

### Multi-Core Optimization

```python
# Use all available CPU cores
import os

best_params, best_report, results_df = backtester.optimize_parallel(
    params=param_ranges,
    workers=os.cpu_count(),  # Use all cores
    chunksize=5              # Process 5 combinations per chunk
)

# Conservative worker allocation
cpu_count = os.cpu_count() or 1
safe_workers = max(1, cpu_count - 1)  # Leave one core free

best_params, best_report, results_df = backtester.optimize_parallel(
    params=param_ranges,
    workers=safe_workers,
    chunksize=10
)
```

### Memory Management for Large Optimizations

```python
# For very large parameter spaces, manage memory
class MemoryEfficientOptimizer:
    def __init__(self, backtester, max_memory_gb=4):
        self.backtester = backtester
        self.max_memory_gb = max_memory_gb

    def optimize_in_batches(self, param_ranges, batch_size=100):
        """Optimize in smaller batches to manage memory"""

        # Generate all parameter combinations
        all_params = list(self.generate_combinations(param_ranges))

        # Process in batches
        all_results = []

        for i in range(0, len(all_params), batch_size):
            batch_params = all_params[i:i+batch_size]

            # Run optimization on this batch
            _, _, batch_results = self.backtester.optimize(
                params={k: [p[k] for p in batch_params] for k in param_ranges.keys()},
                constraint=lambda p: p in batch_params
            )

            all_results.append(batch_results)

            # Memory management
            if i % (batch_size * 5) == 0:
                import gc
                gc.collect()

        # Combine all results
        return pd.concat(all_results, ignore_index=True)

    def generate_combinations(self, param_ranges):
        """Generate all parameter combinations"""
        import itertools

        keys = list(param_ranges.keys())
        values = [list(param_ranges[key]) for key in keys]

        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
```

## Optimization Analysis

### Results Visualization

```python
def plot_optimization_results(self, results_df):
    """Visualize optimization results"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Sharpe ratio distribution
    axes[0, 0].hist(results_df['sharpe'].dropna(), bins=50, alpha=0.7)
    axes[0, 0].axvline(results_df['sharpe'].max(), color='red', linestyle='--',
                       label=f'Best: {results_df["sharpe"].max():.2f}')
    axes[0, 0].set_xlabel('Sharpe Ratio')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Sharpe Ratio Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Parameter relationships (example with 2 parameters)
    if 'fast_period' in results_df.columns and 'slow_period' in results_df.columns:
        scatter = axes[0, 1].scatter(results_df['fast_period'],
                                   results_df['slow_period'],
                                   c=results_df['sharpe'],
                                   cmap='viridis', alpha=0.6)
        axes[0, 1].set_xlabel('Fast Period')
        axes[0, 1].set_ylabel('Slow Period')
        axes[0, 1].set_title('Sharpe Ratio by Parameters')
        plt.colorbar(scatter, ax=axes[0, 1], label='Sharpe Ratio')

    # Top results table
    top_results = results_df.nlargest(10, 'sharpe')
    axes[1, 0].axis('tight')
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Top 10 Results')

    table_data = []
    for _, row in top_results.iterrows():
        table_data.append([f"{row.get('fast_period', 'N/A')}",
                          f"{row.get('slow_period', 'N/A')}",
                          f"{row.get('sharpe', 0):.2f}",
                          f"{row.get('total_return', 0):.2%}"])

    table = axes[1, 0].table(cellText=table_data,
                           colLabels=['Fast Period', 'Slow Period', 'Sharpe', 'Return'],
                           cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)

    # Cumulative returns distribution
    axes[1, 1].hist(results_df['total_return'].dropna(), bins=50, alpha=0.7)
    axes[1, 1].axvline(results_df['total_return'].max(), color='red', linestyle='--',
                       label=f'Best: {results_df["total_return"].max():.2%}')
    axes[1, 1].set_xlabel('Total Return')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Total Return Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()
```

### Parameter Sensitivity Analysis

```python
def analyze_parameter_sensitivity(self, results_df, param_columns):
    """Analyze how sensitive results are to each parameter"""

    sensitivity_results = {}

    for param in param_columns:
        if param in results_df.columns:
            # Group by parameter value and calculate statistics
            param_stats = results_df.groupby(param)['sharpe'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(4)

            sensitivity_results[param] = param_stats

            print(f"\n{param} Sensitivity:")
            print(param_stats)

    return sensitivity_results
```

### Optimization Stability

```python
def test_optimization_stability(self, strategy_class, data, num_runs=5):
    """Test if optimization results are stable across multiple runs"""

    optimization_results = []

    for run in range(num_runs):
        # Create fresh strategy and backtester
        strategy = strategy_class()
        backtester = SimpleBacktester(strategy, cash=10000)

        # Run optimization
        best_params, best_report, results_df = backtester.optimize(
            params=param_ranges
        )

        optimization_results.append({
            'run': run,
            'best_params': best_params,
            'best_sharpe': best_report.sharpe,
            'best_return': best_report.total_return
        })

    # Analyze stability
    results_df = pd.DataFrame(optimization_results)

    print("Optimization Stability Analysis:")
    print(f"Best Sharpe std: {results_df['best_sharpe'].std():.4f}")
    print(f"Best Return std: {results_df['best_return'].std():.4f}")

    # Check parameter consistency
    param_consistency = {}
    for param in param_ranges.keys():
        param_values = [result['best_params'][param] for result in optimization_results]
        param_consistency[param] = {
            'mean': np.mean(param_values),
            'std': np.std(param_values),
            'unique_values': len(set(param_values))
        }

    print("\nParameter Consistency:")
    for param, stats in param_consistency.items():
        print(f"{param}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
              f"unique_values={stats['unique_values']}")

    return results_df
```

## Walk-Forward Optimization

### Rolling Window Optimization

```python
def walk_forward_optimization(self, strategy_class, data, window_years=2, step_months=3):
    """Perform walk-forward optimization"""

    results = []
    start_time = data.Index[0]
    end_time = data.Index[-1]

    current_start = start_time

    while True:
        # Define optimization window
        window_end = current_start + pd.DateOffset(years=window_years)

        # Define test window
        test_start = window_end
        test_end = test_start + pd.DateOffset(months=step_months)

        if test_end > end_time:
            break

        # Filter data for optimization window
        opt_data = data[(data.Index >= current_start) & (data.Index < window_end)]

        if len(opt_data) < 100:  # Need minimum data
            current_start = current_start + pd.DateOffset(months=1)
            continue

        # Run optimization on window
        strategy = strategy_class()
        backtester = SimpleBacktester(strategy, cash=10000)

        # You'd need to modify backtester to use data subset
        # This is a conceptual example

        # Test optimized parameters on test window
        test_strategy = strategy_class(**best_params)
        test_backtester = SimpleBacktester(test_strategy, cash=10000)

        results.append({
            'opt_start': current_start,
            'opt_end': window_end,
            'test_start': test_start,
            'test_end': test_end,
            'best_params': best_params,
            'oos_sharpe': oos_sharpe,
            'oos_return': oos_return
        })

        # Move window
        current_start = current_start + pd.DateOffset(months=step_months)

    return pd.DataFrame(results)
```

### Walk-Forward Analysis

```python
def analyze_walk_forward_results(self, wf_results):
    """Analyze walk-forward optimization results"""

    print("Walk-Forward Analysis Results:")
    print(f"Total windows: {len(wf_results)}")
    print(f"Average OOS Sharpe: {wf_results['oos_sharpe'].mean():.2f}")
    print(f"OOS Sharpe std: {wf_results['oos_sharpe'].std():.2f}")
    print(f"Best OOS Sharpe: {wf_results['oos_sharpe'].max():.2f}")
    print(f"Worst OOS Sharpe: {wf_results['oos_sharpe'].min():.2f}")

    # Check for consistency
    profitable_windows = len(wf_results[wf_results['oos_sharpe'] > 0])
    print(f"Profitable windows: {profitable_windows}/{len(wf_results)} "
          f"({profitable_windows/len(wf_results):.1%})")

    # Parameter stability over time
    print("\nParameter Evolution:")
    for _, row in wf_results.iterrows():
        print(f"{row['test_start'].strftime('%Y-%m')}: {row['best_params']}")

    return {
        'total_windows': len(wf_results),
        'avg_oos_sharpe': wf_results['oos_sharpe'].mean(),
        'sharpe_std': wf_results['oos_sharpe'].std(),
        'profitability_rate': profitable_windows / len(wf_results)
    }
```

## Optimization Best Practices

### 1. Parameter Range Selection

```python
def select_reasonable_ranges(self, strategy_class, initial_data):
    """Select reasonable parameter ranges based on data characteristics"""

    # Analyze data to inform parameter selection
    close_prices = initial_data.Close
    price_series = pd.Series(close_prices)

    # Calculate data statistics
    avg_price = price_series.mean()
    price_std = price_series.std()
    data_length = len(price_series)

    # Suggest parameter ranges based on data
    suggested_ranges = {}

    # Period ranges based on data length
    max_period = min(data_length // 10, 100)  # Max 10% of data or 100
    suggested_ranges['fast_period'] = range(5, min(max_period, 30), 2)
    suggested_ranges['slow_period'] = range(20, min(max_period * 2, 200), 5)

    # Position size based on risk tolerance
    suggested_ranges['position_size'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]

    # Stop loss based on volatility
    volatility = price_series.pct_change().std()
    suggested_ranges['stop_loss'] = [-volatility * 2, -volatility * 3, -volatility * 4]

    print("Suggested parameter ranges:")
    for param, values in suggested_ranges.items():
        print(f"{param}: {list(values)[:5]}{'...' if len(values) > 5 else ''}")

    return suggested_ranges
```

### 2. Overfitting Prevention

```python
def prevent_overfitting(self, strategy_class, data):
    """Implement overfitting prevention measures"""

    # 1. Use out-of-sample testing
    train_data, test_data = self.split_data_for_oos(data, split_ratio=0.7)

    # 2. Optimize on training data
    train_strategy = strategy_class()
    train_backtester = SimpleBacktester(train_strategy, cash=10000)

    # 3. Test on both training and test data
    # 4. Compare performance - if training >> test, likely overfitting

    # 5. Use walk-forward analysis for realistic expectations

    # 6. Consider transaction costs in optimization

    # 7. Use multiple performance metrics, not just one

    return True
```

### 3. Computational Efficiency

```python
def optimize_efficiently(self, param_ranges, max_combinations=10000):
    """Optimize computational efficiency"""

    # Calculate total combinations
    total_combos = 1
    for values in param_ranges.values():
        total_combos *= len(list(values))

    print(f"Total combinations: {total_combos}")

    if total_combos > max_combinations:
        print(f"Too many combinations ({total_combos} > {max_combinations})")
        print("Consider reducing parameter ranges or using constraints")

        # Suggest reductions
        reduction_suggestions = self.suggest_range_reductions(param_ranges, max_combinations)
        return reduction_suggestions

    return param_ranges

def suggest_range_reductions(self, param_ranges, target_combinations):
    """Suggest how to reduce parameter ranges"""

    suggestions = {}

    for param, values in param_ranges.items():
        current_count = len(list(values))

        # Calculate how many we need to remove
        total_current = 1
        for v in param_ranges.values():
            total_current *= len(list(v))

        # Proportionally reduce this parameter
        target_per_param = int(target_combinations ** (1/len(param_ranges)))

        if current_count > target_per_param:
            values_list = list(values)
            step = max(1, current_count // target_per_param)
            suggestions[param] = values_list[::step]
        else:
            suggestions[param] = values

    return suggestions
```

## Custom Optimization Metrics

### Multi-Objective Optimization

```python
def multi_objective_optimization(self, param_ranges, weights={'sharpe': 0.5, 'return': 0.3, 'drawdown': 0.2}):
    """Optimize multiple objectives simultaneously"""

    def calculate_composite_score(row):
        """Calculate composite score from multiple metrics"""

        score = 0
        for metric, weight in weights.items():
            if metric == 'sharpe':
                value = row.get('sharpe', 0)
            elif metric == 'return':
                value = row.get('total_return', 0)
            elif metric == 'drawdown':
                value = -row.get('max_drawdown', 0)  # Negative because lower is better
            else:
                value = row.get(metric, 0)

            score += value * weight

        return score

    # Run standard optimization
    best_params, best_report, results_df = self.backtester.optimize(
        params=param_ranges
    )

    # Calculate composite scores
    results_df['composite_score'] = results_df.apply(calculate_composite_score, axis=1)

    # Find best by composite score
    best_composite_idx = results_df['composite_score'].idxmax()
    best_composite_params = results_df.loc[best_composite_idx]

    return best_composite_params, results_df
```

### Risk-Adjusted Optimization

```python
def risk_adjusted_optimization(self, param_ranges, risk_free_rate=0.04):
    """Optimize risk-adjusted returns"""

    def calculate_risk_adjusted_score(row):
        """Calculate risk-adjusted performance score"""

        sharpe = row.get('sharpe', 0)
        max_dd = row.get('max_drawdown', 0)
        total_return = row.get('total_return', 0)

        # Penalize high drawdowns
        if max_dd > 0.1:  # If drawdown > 10%
            penalty = max_dd * 2  # Penalty factor
            adjusted_sharpe = sharpe - penalty
        else:
            adjusted_sharpe = sharpe

        # Boost for high returns with reasonable risk
        if total_return > 0.5 and max_dd < 0.15:  # 50% return, 15% max DD
            bonus = 0.5
            adjusted_sharpe += bonus

        return adjusted_sharpe

    # Run optimization with custom scoring
    best_params, best_report, results_df = self.backtester.optimize(
        params=param_ranges
    )

    # Apply risk-adjusted scoring
    results_df['risk_adjusted_score'] = results_df.apply(calculate_risk_adjusted_score, axis=1)

    # Re-rank results
    results_df = results_df.sort_values('risk_adjusted_score', ascending=False)

    return results_df.iloc[0], results_df
```

## Optimization Workflow

### Complete Optimization Pipeline

```python
class OptimizationPipeline:
    def __init__(self, strategy_class, data_source):
        self.strategy_class = strategy_class
        self.data_source = data_source

    def run_complete_optimization(self, param_ranges, enable_parallel=True):
        """Run complete optimization pipeline"""

        print("=== Optimization Pipeline ===")

        # 1. Data validation
        print("1. Validating data...")
        if not self.validate_data():
            return None

        # 2. Parameter range optimization
        print("2. Optimizing parameter ranges...")
        optimized_ranges = self.optimize_parameter_ranges(param_ranges)

        # 3. Run optimization
        print("3. Running optimization...")
        strategy = self.strategy_class()
        backtester = SimpleBacktester(strategy, cash=10000)

        if enable_parallel:
            best_params, best_report, results_df = backtester.optimize_parallel(
                params=optimized_ranges,
                workers=os.cpu_count() - 1
            )
        else:
            best_params, best_report, results_df = backtester.optimize(
                params=optimized_ranges
            )

        # 4. Out-of-sample validation
        print("4. Running out-of-sample validation...")
        oos_performance = self.validate_out_of_sample(best_params)

        # 5. Walk-forward analysis
        print("5. Running walk-forward analysis...")
        wf_results = self.walk_forward_analysis(best_params)

        # 6. Generate report
        print("6. Generating optimization report...")
        report = self.generate_optimization_report(
            best_params, best_report, results_df, oos_performance, wf_results
        )

        return report

    def validate_data(self):
        """Validate input data quality"""
        # Implementation here
        return True

    def optimize_parameter_ranges(self, param_ranges):
        """Optimize parameter ranges for efficiency"""
        # Implementation here
        return param_ranges

    def validate_out_of_sample(self, best_params):
        """Validate best parameters on out-of-sample data"""
        # Implementation here
        return {}

    def walk_forward_analysis(self, best_params):
        """Perform walk-forward analysis"""
        # Implementation here
        return pd.DataFrame()

    def generate_optimization_report(self, *args):
        """Generate comprehensive optimization report"""
        # Implementation here
        return {}
```

## Next Steps

Now that you understand parameter optimization in QuantEx, explore these related topics:

- **[Strategy Guide](./strategy.md)**: Learn how to design optimizable strategies
- **[Backtesting Guide](./backtesting.md)**: Understand how optimization integrates with backtesting
- **[Technical Indicators Guide](./indicators.md)**: Optimize indicator parameters

For complete API reference, see the [Optimization API documentation](../../reference/quantex.optimizer.md).