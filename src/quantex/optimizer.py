from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from typing import Any, Mapping, Type
import pandas as pd
from quantex.backtest import BacktestRunner
from quantex.execution import ImmediateFillSimulator, NextBarSimulator
from quantex.sources import BacktestingDataSource
from quantex.strategy import Strategy
from tqdm import tqdm


def _run_single_backtest(
    strategy_class: Type[Strategy],
    data_sources: Mapping[str, BacktestingDataSource],
    strategy_symbols: list[str],
    initial_cash: float,
    risk_free_rate: float,
    min_holding_period: pd.Timedelta | None,
    simulator_class: Type[ImmediateFillSimulator],
    current_params_dict: dict[str, Any],
    optimization_metric: str,
) -> tuple[dict[str, Any], float | None, Exception | None]:
    """
    Executes a single backtest for a given set of parameters.
    Returns the parameters, the metric value, and any exception.
    """
    try:
        # Instantiate a fresh strategy for this run
        strategy_instance = strategy_class(
            symbols=strategy_symbols, initial_cash=initial_cash
        )

        # Set the current parameters on the strategy instance
        for param_name, param_value in current_params_dict.items():
            # Check if the attribute exists before setting to prevent silent errors
            if not hasattr(strategy_instance, param_name):
                raise AttributeError(
                    f"Strategy '{strategy_class.__name__}' "
                    f"does not have a parameter named '{param_name}'."
                )
            setattr(strategy_instance, param_name, param_value)

        # Instantiate simulator (its portfolio needs to be linked to strategy_instance.portfolio)
        # Note: BacktestRunner's __init__ typically handles simulator instantiation/linking
        simulator = simulator_class(
            portfolio=strategy_instance.portfolio,
            commission=0.0,  # Make this configurable if needed
            min_holding_period=min_holding_period,
        )

        runner = BacktestRunner(
            strategy=strategy_instance,
            data_sources=data_sources,
            risk_free_rate=risk_free_rate,
            min_holding_period=min_holding_period,
            simulator=simulator,
        )
        runner.event_bus.show_progress = False
        result = runner.run()
        metric_value = result.metrics.get(optimization_metric)
        return current_params_dict, metric_value, None
    except Exception as e:
        return current_params_dict, None, e


class Optimizer:
    def __init__(
        self,
        strategy_class: Type[Strategy],
        data_sources: Mapping[str, BacktestingDataSource],
        strategy_symbols: list[str],
        initial_cash: float = 100_000,
        risk_free_rate: float = 0.0,
        periods_per_year: int = int(
            252 * 60 * 6.5
        ),  # Default for 1-min bars, 6.5 trading hours/day
        min_holding_period: pd.Timedelta | None = None,
        simulator_class: Type[ImmediateFillSimulator] = NextBarSimulator,
    ):
        self.strategy_class = strategy_class
        self.data_sources = data_sources
        self.strategy_symbols = strategy_symbols
        self.initial_cash = initial_cash
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.min_holding_period = min_holding_period
        self.simulator_class = simulator_class

    def optimize_grid_search(
        self,
        params: dict[str, tuple[float, float, float]],  # {param_name: (min, max, step)}
        optimization_metric: str = "sharpe_ratio",
        minimize_metric: bool = False,
        max_workers: int | None = None,  # Number of parallel threads/processes
    ) -> tuple[dict[str, Any], float]:
        """
        Performs a grid search optimization over strategy parameters using multithreading.

        Args:
            params (dict): A dictionary mapping parameter names (as attributes
                           of the strategy) to tuples of (min_val, max_val, step_size).
                           Example: {"fast_period": (10.0, 30.0, 5.0), "slow_period": (50.0, 100.0, 10.0)}
            optimization_metric (str): The name of the metric in BacktestResult.metrics
                                       to optimize (e.g., "sharpe_ratio", "total_return", "max_drawdown").
            minimize_metric (bool): If True, the goal is to minimize the metric (e.g.,
                                    max_drawdown). If False, maximize (e.g., sharpe_ratio).
            max_workers (int | None): The number of threads to use. Defaults to CPU count.

        Returns:
            tuple: A tuple containing:
                - dict: The best parameter combination found.
                - float: The value of the optimization metric for the best parameters.
        """
        param_names = []
        param_values_lists = []

        for variable, (min_val, max_val, step_size) in params.items():
            param_names.append(variable)
            values_for_param = []
            current_val = min_val
            while current_val <= max_val + 1e-9:
                values_for_param.append(round(current_val, 5))
                current_val += step_size
            if (
                not values_for_param and min_val <= max_val
            ):  # Catch case where step_size is too large for loop
                values_for_param.append(min_val)
            param_values_lists.append(values_for_param)

        all_param_combinations = list(itertools.product(*param_values_lists))
        total_iterations = len(all_param_combinations)

        if total_iterations == 0:
            print("No parameter combinations to test. Returning empty result.")
            return {}, float("nan")

        print(
            f"Preparing to run {total_iterations:,} backtests for optimization "
            f"using {max_workers if max_workers else 'default'} workers."
        )
        print(
            f"Optimizing for metric: '{optimization_metric}' (Goal: {'Minimize' if minimize_metric else 'Maximize'})"
        )

        best_metric_value = float("inf") if minimize_metric else float("-inf")
        best_params_found: dict[str, Any] = {}

        # Using ThreadPoolExecutor as backtesting often involves I/O and C-extensions (NumPy/Pandas)
        # that release the GIL, allowing for concurrency even in Python.
        # If the backtesting logic is pure-Python and CPU-bound, ProcessPoolExecutor would be better.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for param_combination in all_param_combinations:
                current_params_dict = dict(zip(param_names, param_combination))
                future = executor.submit(
                    _run_single_backtest,
                    self.strategy_class,
                    self.data_sources,
                    self.strategy_symbols,
                    self.initial_cash,
                    self.risk_free_rate,
                    self.min_holding_period,
                    self.simulator_class,
                    current_params_dict,
                    optimization_metric,
                )
                futures.append(future)

            # Process results as they complete with a progress bar
            for future in tqdm(
                as_completed(futures),
                total=total_iterations,
                desc="Running Backtests",
            ):
                current_params, metric_value, exception = future.result()

                if exception:
                    # print(f"Error for params {current_params}: {exception}")
                    # In a real scenario, you might want more detailed logging
                    continue

                if metric_value is None:
                    # print(f"Warning: Metric '{optimization_metric}' not found for params {current_params}. Skipping.")
                    continue

                # Update best parameters (thread-safe operations on primitives)
                if minimize_metric:
                    if metric_value < best_metric_value:
                        best_metric_value = metric_value
                        best_params_found = current_params.copy()
                else:
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_params_found = current_params.copy()

        print("\nOptimization Complete.")
        if best_params_found:
            print(f"Best Parameters: {best_params_found}")
            print(f"Best '{optimization_metric}': {best_metric_value:.4f}")
        else:
            print(
                "No valid backtest results found for optimization or all runs resulted in errors."
            )

        return best_params_found, best_metric_value
