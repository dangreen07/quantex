# Optimizer API Reference

This page provides comprehensive API documentation for QuantEx's parameter optimization system.

## Optimization Methods

### Grid Search Optimization

#### `optimize(params: dict[str, range], constraint: Callable[[dict[str, Any]], bool] | None = None)`

Perform exhaustive grid search over parameter combinations.

```python
def optimize(self, params: dict[str, range],
             constraint: Callable[[dict[str, Any]], bool] | None = None):
    """Exhaustive grid search optimization"""
    pass
```

**Parameters:**
- `params` (dict): Dictionary mapping parameter names to iterables of values
- `constraint` (callable, optional): Function to filter parameter combinations

**Returns:**
- `best_params` (dict): Best parameter combination found
- `best_report` (BacktestReport): Best backtest results
- `results_df` (pd.DataFrame): All optimization results

**Usage:**
```python
# Define parameter ranges
param_ranges = {
    'fast_period': range(5, 21, 2),    # 5, 7, 9, ..., 19
    'slow_period': range(20, 51, 5),   # 20, 25, 30, ..., 50
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

#### `optimize_parallel(params: dict[str, range], constraint: Callable[[dict[str, Any]], bool] | None = None, workers: int | None = None, chunksize: int = 1)`

Perform parallel grid search optimization using multiple processes.

```python
def optimize_parallel(self, params: dict[str, range],
                     constraint: Callable[[dict[str, Any]], bool] | None = None,
                     workers: int | None = None, chunksize: int = 1):
    """Parallel grid search optimization"""
    pass
```

**Parameters:**
- `params` (dict): Dictionary mapping parameter names to iterables of values
- `constraint` (callable, optional): Function to filter parameter combinations
- `workers` (int, optional): Number of worker processes (auto-detected if None)
- `chunksize` (int): Number of combinations to process per worker chunk

**Returns:**
- `best_params` (dict): Best parameter combination found
- `best_report` (BacktestReport): Best backtest results
- `results_df` (pd.DataFrame): All optimization results

**Usage:**
```python
# Parallel optimization with custom configuration
best_params, best_report, results_df = backtester.optimize_parallel(
    params=param_ranges,
    workers=4,           # Use 4 worker processes
    chunksize=10         # Process 10 combinations per chunk
)

# Auto-detect optimal worker count
best_params, best_report, results_df = backtester.optimize_parallel(
    params=param_ranges,
    workers=None  # Let backtester choose optimal number
)
```

## Parameter Range Types

### Supported Parameter Formats

```python
# Integer ranges
int_range = range(5, 21, 2)  # 5, 7, 9, ..., 19

# List of specific values
specific_values = [10, 15, 20, 25, 30]

# Float ranges using numpy
import numpy as np
float_range = np.linspace(0.1, 0.5, 9)  # 0.1, 0.15, 0.2, ..., 0.5

# Log-spaced values
log_values = np.logspace(-2, -1, 5)  # 0.01, 0.025, 0.04, 0.063, 0.1

# Mixed parameter types
mixed_params = {
    'fast_period': range(5, 21, 2),                    # Integer range
    'slow_period': [20, 25, 30, 35, 40, 45, 50],      # Specific values
    'position_size': np.linspace(0.1, 0.5, 9),        # Float range
    'stop_loss': np.logspace(-0.02, -0.001, 10),       # Log-spaced
    'rsi_period': [7, 10, 14, 21, 28],                # Fibonacci-like
    'bb_std': [1.5, 1.8, 2.0, 2.2, 2.5]              # Standard deviations
}
```

## Constraint Functions

### Basic Constraints

```python
def basic_constraints(params):
    """Basic parameter constraints"""

    # Fast period should be less than slow period
    if params['fast_period'] >= params['slow_period']:
        return False

    # Position size should be reasonable
    if not (0.01 <= params['position_size'] <= 1.0):
        return False

    # Stop loss should be negative (for sell orders)
    if 'stop_loss' in params and params['stop_loss'] >= 0:
        return False

    return True
```

### Advanced Constraints

```python
def advanced_constraints(params):
    """Advanced parameter constraints"""

    # Custom logic constraints
    if params['fast_period'] * 2 > params['slow_period']:
        return False

    # Volatility-based constraints
    if 'volatility_period' in params:
        if params['volatility_period'] < params['fast_period']:
            return False

    # Risk management constraints
    if 'position_size' in params and 'stop_loss' in params:
        # Position size should be inversely related to stop loss magnitude
        stop_distance = abs(params['stop_loss'])
        if params['position_size'] > 1.0 / (1 + stop_distance * 100):
            return False

    # Market condition constraints
    if 'regime' in params:
        # Different parameters for different market regimes
        if params['regime'] == 'trending':
            if params['slow_period'] < 30:
                return False
        elif params['regime'] == 'ranging':
            if params['slow_period'] > 40:
                return False

    return True
```

## Optimization Results Analysis

### Results DataFrame Structure

```python
# Analyze optimization results
def analyze_optimization_results(results_df):
    """Comprehensive analysis of optimization results"""

    print("Optimization Results Summary:")
    print(f"Total combinations tested: {len(results_df)}")

    # Basic statistics
    print(f"Best Sharpe: {results_df['sharpe'].max():.2f}")
    print(f"Worst Sharpe: {results_df['sharpe'].min():.2f}")
    print(f"Mean Sharpe: {results_df['sharpe'].mean():.2f}")
    print(f"Sharpe std: {results_df['sharpe'].std():.2f}")

    # Distribution analysis
    positive_sharpe = len(results_df[results_df['sharpe'] > 0])
    negative_sharpe = len(results_df[results_df['sharpe'] < 0])

    print(f"Positive Sharpe combinations: {positive_sharpe}")
    print(f"Negative Sharpe combinations: {negative_sharpe}")
    print(f"Success rate: {positive_sharpe/len(results_df):.1%}")

    # Parameter sensitivity
    for param in ['fast_period', 'slow_period', 'position_size']:
        if param in results_df.columns:
            param_stats = results_df.groupby(param)['sharpe'].agg(['mean', 'std', 'count'])
            print(f"\n{param} sensitivity:")
            print(param_stats)

    return {
        'total_combinations': len(results_df),
        'best_sharpe': results_df['sharpe'].max(),
        'success_rate': positive_sharpe / len(results_df),
        'parameter_sensitivity': param_stats
    }
```

### Results Visualization

```python
def visualize_optimization_results(results_df):
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

    # Parameter relationships (2D scatter if possible)
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

    # Return distribution
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

## Advanced Optimization Techniques

### Multi-Objective Optimization

```python
def multi_objective_optimization(backtester, param_ranges, objectives):
    """Optimize multiple objectives simultaneously"""

    def calculate_composite_score(row):
        """Calculate composite score from multiple objectives"""

        score = 0
        for obj in objectives:
            metric_name = obj['name']
            weight = obj['weight']

            if metric_name == 'sharpe':
                value = row.get('sharpe', 0)
            elif metric_name == 'return':
                value = row.get('total_return', 0)
            elif metric_name == 'drawdown':
                value = -row.get('max_drawdown', 0)  # Negative because lower is better
            else:
                value = row.get(metric_name, 0)

            score += value * weight

        return score

    # Run standard optimization
    best_params, best_report, results_df = backtester.optimize(param_ranges)

    # Calculate composite scores
    results_df['composite_score'] = results_df.apply(calculate_composite_score, axis=1)

    # Find best by composite score
    best_composite_idx = results_df['composite_score'].idxmax()
    best_composite_params = results_df.loc[best_composite_idx]

    return best_composite_params, results_df

# Usage
objectives = [
    {'name': 'sharpe', 'weight': 0.5},
    {'name': 'return', 'weight': 0.3},
    {'name': 'drawdown', 'weight': 0.2}
]

best_params, results_df = multi_objective_optimization(backtester, param_ranges, objectives)
```

### Walk-Forward Optimization

```python
def walk_forward_optimization(strategy_class, data, param_ranges, window_config):
    """Perform walk-forward optimization"""

    results = []
    start_time = data.Index[0]
    end_time = data.Index[-1]

    current_start = start_time

    while True:
        # Define optimization window
        opt_window_end = current_start + pd.DateOffset(years=window_config['opt_years'])

        # Define test window
        test_window_start = opt_window_end
        test_window_end = test_window_start + pd.DateOffset(months=window_config['test_months'])

        if test_window_end > end_time:
            break

        # Filter data for optimization window
        opt_data = data[(data.Index >= current_start) & (data.Index < opt_window_end)]

        if len(opt_data) < window_config['min_data_points']:
            current_start = current_start + pd.DateOffset(months=1)
            continue

        # Run optimization on window
        strategy = strategy_class()
        backtester = SimpleBacktester(strategy, cash=window_config['cash'])

        # You'd need to modify backtester to use data subset
        # This is a conceptual example

        # Test optimized parameters on test window
        test_strategy = strategy_class(**best_params)
        test_backtester = SimpleBacktester(test_strategy, cash=window_config['cash'])

        results.append({
            'opt_start': current_start,
            'opt_end': opt_window_end,
            'test_start': test_window_start,
            'test_end': test_window_end,
            'best_params': best_params,
            'oos_sharpe': oos_sharpe,
            'oos_return': oos_return
        })

        # Move window
        current_start = current_start + pd.DateOffset(months=window_config['step_months'])

    return pd.DataFrame(results)
```

## Optimization Best Practices

### Parameter Range Selection

```python
def select_optimal_parameter_ranges(data_source, strategy_class, config):
    """Select optimal parameter ranges based on data characteristics"""

    # Analyze data characteristics
    close_prices = data_source.Close
    data_length = len(close_prices)
    price_series = pd.Series(close_prices)

    # Calculate data statistics
    avg_price = price_series.mean()
    price_std = price_series.std()
    data_range = data_length

    # Suggest parameter ranges based on data
    suggested_ranges = {}

    # Period ranges based on data length and frequency
    max_period = min(data_range // 10, 200)  # Max 10% of data or 200
    suggested_ranges['fast_period'] = range(5, min(max_period // 4, 50), 2)
    suggested_ranges['slow_period'] = range(max(20, max_period // 10), min(max_period, 200), 5)

    # Position size based on risk tolerance
    suggested_ranges['position_size'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]

    # Stop loss based on volatility
    volatility = price_series.pct_change().std()
    suggested_ranges['stop_loss'] = [-volatility * 2, -volatility * 3, -volatility * 4]

    # Risk-free rate adjustment
    if 'risk_free_rate' in config:
        # Adjust ranges based on risk-free rate
        pass

    print("Suggested parameter ranges:")
    for param, values in suggested_ranges.items():
        print(f"  {param}: {list(values)[:5]}{'...' if len(values) > 5 else ''}")

    return suggested_ranges
```

### Computational Efficiency

```python
def optimize_efficiently(backtester, param_ranges, constraints):
    """Optimize with computational constraints"""

    # Calculate total combinations
    total_combos = 1
    for values in param_ranges.values():
        total_combos *= len(list(values))

    print(f"Total combinations: {total_combos:,}")

    # Apply constraints to reduce search space
    if constraints:
        estimated_combos = estimate_constrained_combinations(param_ranges, constraints)
        print(f"Estimated combinations after constraints: {estimated_combos:,}")
    else:
        estimated_combos = total_combos

    # Decide optimization strategy
    if estimated_combos > 10000:
        print("Large search space detected. Using parallel optimization.")
        return backtester.optimize_parallel(param_ranges, constraint=constraints)
    else:
        print("Small search space. Using standard optimization.")
        return backtester.optimize(param_ranges, constraint=constraints)

def estimate_constrained_combinations(param_ranges, constraint_func):
    """Estimate combinations after applying constraints"""

    # Sample a subset to estimate constraint acceptance rate
    sample_size = min(1000, np.prod([len(list(v)) for v in param_ranges.values()]))

    accepted = 0
    total_sampled = 0

    # Generate sample combinations
    for params in generate_sample_combinations(param_ranges, sample_size):
        if constraint_func(params):
            accepted += 1
        total_sampled += 1

    acceptance_rate = accepted / total_sampled if total_sampled > 0 else 0

    # Estimate total accepted combinations
    total_combos = np.prod([len(list(v)) for v in param_ranges.values()])
    estimated_accepted = total_combos * acceptance_rate

    return int(estimated_accepted)
```

## Custom Optimization Strategies

### Bayesian Optimization

```python
class BayesianOptimizer:
    """Bayesian optimization for parameter tuning"""

    def __init__(self, backtester, param_ranges, n_iter=50):
        self.backtester = backtester
        self.param_ranges = param_ranges
        self.n_iter = n_iter
        self.observations = []

    def optimize(self):
        """Run Bayesian optimization"""

        # Initialize with random sampling
        initial_points = self.generate_initial_points(5)

        for params in initial_points:
            score = self.evaluate_params(params)
            self.observations.append({**params, 'score': score})

        # Bayesian optimization iterations
        for i in range(self.n_iter):
            # Fit Gaussian process to observations
            # Suggest next point to evaluate
            # Evaluate and add to observations

            next_params = self.suggest_next_params()
            score = self.evaluate_params(next_params)
            self.observations.append({**next_params, 'score': score})

        # Return best parameters
        best_obs = max(self.observations, key=lambda x: x['score'])
        return best_obs, pd.DataFrame(self.observations)

    def evaluate_params(self, params):
        """Evaluate parameter combination"""
        strategy = self.backtester.strategy.__class__(**params)
        test_backtester = SimpleBacktester(strategy, cash=self.backtester.cash)

        try:
            report = test_backtester.run(progress_bar=False)
            return report.sharpe
        except:
            return float('-inf')

    def generate_initial_points(self, n_points):
        """Generate initial random points"""
        # Implementation here
        return []

    def suggest_next_params(self):
        """Suggest next parameters using Bayesian optimization"""
        # Implementation here
        return {}
```

### Genetic Algorithm Optimization

```python
class GeneticOptimizer:
    """Genetic algorithm for parameter optimization"""

    def __init__(self, backtester, param_ranges, population_size=50, generations=20):
        self.backtester = backtester
        self.param_ranges = param_ranges
        self.population_size = population_size
        self.generations = generations

    def optimize(self):
        """Run genetic algorithm optimization"""

        # Initialize population
        population = self.initialize_population()

        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(individual) for individual in population]

            # Select best individuals
            selected = self.select_best(population, fitness_scores)

            # Create next generation
            population = self.create_next_generation(selected)

            # Add some random mutation
            population = self.mutate_population(population)

        # Return best individual
        fitness_scores = [self.evaluate_fitness(individual) for individual in population]
        best_idx = np.argmax(fitness_scores)

        return population[best_idx], pd.DataFrame([
            {**individual, 'fitness': fitness}
            for individual, fitness in zip(population, fitness_scores)
        ])

    def initialize_population(self):
        """Initialize random population"""
        # Implementation here
        return []

    def evaluate_fitness(self, individual):
        """Evaluate individual fitness"""
        # Implementation here
        return 0.0

    def select_best(self, population, fitness_scores):
        """Select best individuals"""
        # Implementation here
        return []

    def create_next_generation(self, selected):
        """Create next generation through crossover"""
        # Implementation here
        return []

    def mutate_population(self, population):
        """Apply mutation to population"""
        # Implementation here
        return population
```

## Optimization Pipeline

### Complete Optimization Workflow

```python
class OptimizationPipeline:
    """Complete optimization pipeline"""

    def __init__(self, strategy_class, data_source):
        self.strategy_class = strategy_class
        self.data_source = data_source

    def run_complete_optimization(self, param_ranges, config):
        """Run complete optimization pipeline"""

        print("=== QuantEx Optimization Pipeline ===")

        # 1. Data validation and preparation
        print("1. Validating data...")
        if not self.validate_data():
            raise OptimizationError("Data validation failed")

        # 2. Parameter range optimization
        print("2. Optimizing parameter ranges...")
        optimized_ranges = self.optimize_parameter_ranges(param_ranges)

        # 3. Initial screening
        print("3. Running initial screening...")
        screening_results = self.run_initial_screening(optimized_ranges)

        # 4. Main optimization
        print("4. Running main optimization...")
        if config['parallel']:
            best_params, best_report, results_df = self.run_parallel_optimization(
                optimized_ranges, config
            )
        else:
            best_params, best_report, results_df = self.run_sequential_optimization(
                optimized_ranges, config
            )

        # 5. Out-of-sample validation
        print("5. Running out-of-sample validation...")
        oos_results = self.validate_out_of_sample(best_params, config)

        # 6. Walk-forward analysis
        print("6. Running walk-forward analysis...")
        wf_results = self.walk_forward_analysis(best_params, config)

        # 7. Generate comprehensive report
        print("7. Generating optimization report...")
        report = self.generate_optimization_report(
            best_params, best_report, results_df, oos_results, wf_results
        )

        return report

    def validate_data(self):
        """Validate input data"""
        # Implementation
        return True

    def optimize_parameter_ranges(self, param_ranges):
        """Optimize parameter ranges for efficiency"""
        # Implementation
        return param_ranges

    def run_initial_screening(self, param_ranges):
        """Run initial parameter screening"""
        # Implementation
        return pd.DataFrame()

    def run_parallel_optimization(self, param_ranges, config):
        """Run parallel optimization"""
        strategy = self.strategy_class()
        backtester = SimpleBacktester(strategy, **config['backtester_config'])

        return backtester.optimize_parallel(
            param_ranges,
            workers=config['workers'],
            chunksize=config['chunksize']
        )

    def run_sequential_optimization(self, param_ranges, config):
        """Run sequential optimization"""
        strategy = self.strategy_class()
        backtester = SimpleBacktester(strategy, **config['backtester_config'])

        return backtester.optimize(param_ranges)

    def validate_out_of_sample(self, best_params, config):
        """Validate best parameters out-of-sample"""
        # Implementation
        return {}

    def walk_forward_analysis(self, best_params, config):
        """Perform walk-forward analysis"""
        # Implementation
        return pd.DataFrame()

    def generate_optimization_report(self, *args):
        """Generate comprehensive optimization report"""
        # Implementation
        return {}
```

## Error Handling

### Optimization Errors

```python
class OptimizationError(Exception):
    """Base exception for optimization errors"""
    pass

class InvalidParameterError(OptimizationError):
    """Raised when parameter ranges are invalid"""
    pass

class ConstraintError(OptimizationError):
    """Raised when constraint function fails"""
    pass

class OptimizationTimeoutError(OptimizationError):
    """Raised when optimization times out"""
    pass

# Safe optimization execution
def safe_optimization(backtester, param_ranges, constraint=None, timeout=3600):
    """Run optimization with error handling and timeout"""

    import signal

    def timeout_handler(signum, frame):
        raise OptimizationTimeoutError("Optimization timed out")

    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        # Run optimization
        return backtester.optimize(param_ranges, constraint=constraint)

    except InvalidParameterError as e:
        print(f"Invalid parameters: {e}")
        raise

    except ConstraintError as e:
        print(f"Constraint error: {e}")
        raise

    except OptimizationTimeoutError as e:
        print(f"Optimization timeout: {e}")
        raise

    except Exception as e:
        print(f"Unexpected optimization error: {e}")
        raise

    finally:
        # Clear timeout
        signal.alarm(0)
```

## Performance Monitoring

### Optimization Performance Metrics

```python
def monitor_optimization_performance(backtester, param_ranges, results_df):
    """Monitor optimization performance"""

    # Calculate performance metrics
    total_combinations = len(results_df)
    execution_time = getattr(backtester, '_optimization_time', 0)

    if execution_time == 0:
        return {}

    # Performance per combination
    time_per_combination = execution_time / total_combinations
    combinations_per_second = total_combinations / execution_time

    # Best result timing
    best_result = results_df.loc[results_df['sharpe'].idxmax()]
    best_time = time_per_combination  # Simplified

    print("Optimization Performance:")
    print(f"  Total Combinations: {total_combinations:,}")
    print(f"  Total Time: {execution_time:.2f} seconds")
    print(f"  Time per Combination: {time_per_combination:.4f} seconds")
    print(f"  Combinations/Second: {combinations_per_second:.1f}")
    print(f"  Best Sharpe: {best_result['sharpe']:.2f}")
    print(f"  Efficiency: {best_result['sharpe']/time_per_combination:.4f} Sharpe/second")

    return {
        'total_combinations': total_combinations,
        'execution_time': execution_time,
        'time_per_combination': time_per_combination,
        'combinations_per_second': combinations_per_second,
        'best_sharpe': best_result['sharpe']
    }
```

## Related Documentation

- **[Optimization Usage Guide](../../../usage/optimizer.md)**: Learn how to use the optimizer
- **[Backtester API](../../quantex.backtest.md)**: Understand backtesting integration
- **[Strategy API](../../quantex.strategy.md)**: Design optimizable strategies
- **[Engine API](../../quantex.engine.md)**: Advanced optimization features