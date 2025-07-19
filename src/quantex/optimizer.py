"""
optimizer.py

Provides a grid‐search optimizer for QuantEx strategies, running backtests
in parallel and selecting the best parameter combination.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from typing import Any, Mapping, Type, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import copy

from quantex.backtest import BacktestRunner
from quantex.execution import ImmediateFillSimulator, NextBarSimulator
from quantex.sources import BacktestingDataSource
from quantex.strategy import Strategy


def _run_single_backtest(
    strategy_template: Strategy,
    data_sources: Mapping[str, BacktestingDataSource],
    risk_free_rate: float,
    min_holding_period: Optional[pd.Timedelta],
    simulator_class: Type[ImmediateFillSimulator],
    current_params_dict: dict[str, Any],
    optimization_metric: str,
) -> Tuple[dict[str, Any], Optional[float], Optional[Exception]]:
    """Execute one backtest with a given parameter set.

    Instantiates the strategy, applies parameters, runs the backtest,
    and extracts the requested metric.

    Args:
        strategy_template: *Pre-initialised* strategy instance that will be
            deep-copied for the individual backtest run. This avoids costly
            re-initialisation work (e.g. indicator caches) for every grid
            search combination.
        data_sources: Mapping of source names to data sources.
        risk_free_rate: Risk‐free rate for Sharpe ratio calculation.
        min_holding_period: Minimum holding period for positions.
        simulator_class: Execution simulator class to use.
        current_params_dict: Parameter values for this run.
        optimization_metric: Metric name to extract from results.

    Returns:
        A tuple containing:
        - The parameter dict used.
        - The metric value (or None if missing or error).
        - An Exception instance if an error occurred, else None.
    """
    try:
        # Instantiate and configure strategy by deep-copying the template to
        # ensure *full* independence between parallel workers.
        strategy_instance = copy.deepcopy(strategy_template)
        for name, value in current_params_dict.items():
            if not hasattr(strategy_instance, name):
                raise AttributeError(
                    f"'{strategy_instance.__class__.__name__}' has no parameter '{name}'"
                )
            setattr(strategy_instance, name, value)

        # Instantiate simulator and runner
        simulator = simulator_class(
            portfolio=strategy_instance.portfolio,
            commission=0.0,
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
    """Grid‐search optimizer for QuantEx strategies."""

    def __init__(
        self,
        strategy_class: Type[Strategy],
        data_sources: Mapping[str, BacktestingDataSource],
        strategy_symbols: list[str],
        initial_cash: float = 100_000,
        risk_free_rate: float = 0.0,
        periods_per_year: int = int(252 * 60 * 6.5),
        min_holding_period: Optional[pd.Timedelta] = None,
        simulator_class: Type[ImmediateFillSimulator] = NextBarSimulator,
    ):
        """Initialize the Optimizer.

        Args:
            strategy_class: Strategy subclass to optimize.
            data_sources: Mapping of source names to data sources.
            strategy_symbols: List of symbols the strategy trades.
            initial_cash: Starting cash balance.
            risk_free_rate: Risk‐free rate for metric calculation.
            periods_per_year: Periods per year for annualized metrics.
            min_holding_period: Minimum holding period for positions.
            simulator_class: Execution simulator to use.
        """
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
        params: dict[str, Tuple[float, float, float]],
        optimization_metric: str = "sharpe_ratio",
        minimize_metric: bool = False,
        max_workers: Optional[int] = None,
    ) -> Tuple[dict[str, Any], float]:
        """Perform a multithreaded grid search over strategy parameters.

        Args:
            params: A dict mapping parameter names to (min, max, step).
                E.g., {"fast": (10, 30, 5), "slow": (50, 100, 10)}.
            optimization_metric: Metric name in BacktestResult.metrics to
                optimize (e.g., "sharpe_ratio").
            minimize_metric: If True, seeks to minimize the metric.
            max_workers: Number of parallel threads to use.

        Returns:
            A tuple containing:
            - The best parameter combination found.
            - The best metric value achieved.
        """
        # Build value lists for each parameter
        names: list[str] = []
        grids: list[list[float]] = []
        for var, (mn, mx, step) in params.items():
            names.append(var)
            vals: list[float] = []
            v = mn
            while v <= mx + 1e-9:
                vals.append(round(v, 8))
                v += step
            if not vals and mn <= mx:
                vals.append(mn)
            grids.append(vals)

        combos = list(itertools.product(*grids))
        total = len(combos)
        if total == 0:
            return {}, float("nan")

        print(f"Running {total:,} backtests using {max_workers or 'default'} workers")
        best_val = float("inf") if minimize_metric else float("-inf")
        best_params: dict[str, Any] = {}

        # Reuse a single *template* strategy instance and deepcopy it inside
        # each parallel worker to avoid repeated heavy initialisation.
        strategy_template = self.strategy_class(
            symbols=self.strategy_symbols, initial_cash=self.initial_cash
        )

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = [
                exe.submit(
                    _run_single_backtest,
                    strategy_template,
                    self.data_sources,
                    self.risk_free_rate,
                    self.min_holding_period,
                    self.simulator_class,
                    dict(zip(names, combo)),
                    optimization_metric,
                )
                for combo in combos
            ]

            for future in tqdm(
                as_completed(futures),
                total=total,
                desc="Optimizing",
            ):
                params_dict, value, err = future.result()
                if err or value is None:
                    continue
                if minimize_metric:
                    if value < best_val:
                        best_val = value
                        best_params = params_dict.copy()
                else:
                    if value > best_val:
                        best_val = value
                        best_params = params_dict.copy()

        print("Optimization complete.")
        return best_params, best_val
