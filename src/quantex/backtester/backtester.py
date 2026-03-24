import copy
import itertools
import math
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..broker import Order
from ..strategy import Strategy

from .constants import DataSplitMode
from ..broker.types import CommissionType
from .data_splits import create_train_validate_test_split
from .metrics import (
    _compute_backtest_metrics,
    _risk_tolerance_passes,
)
from .parallel import _worker_eval, _worker_init
from .reports import BacktestReport, OptimizationResult


from .montecarlo import monte_carlo, MonteCarloResult, MonteCarloMode

class SimpleBacktester:
    """
    Simple backtester for executing trading strategies on historical data.
    
    This class provides functionality to run backtests on trading strategies,
    calculate performance metrics, and perform parameter optimization through
    grid search (both sequential and parallel).
    
    The backtester simulates realistic trading conditions including:
    - Order execution with market and limit orders
    - Commission calculations
    - Position management
    - Margin calls
    - P&L tracking
    - Leverage for amplified position sizing
    
    Example:
        >>> from quantex import SimpleBacktester, CSVDataSource  
        >>> # Create strategy and data source  
        >>> source = CSVDataSource("data.csv")  
        >>> # strategy = MyStrategy()  # Your custom strategy  
        >>> bt = SimpleBacktester(strategy, cash=10000)  
        >>> report = bt.run()  
        >>> print(report)  
    """
    def __init__(self, 
                 strategy: Strategy,
                cash: float = 10_000, 
                commission: float = 0.002, 
                commission_type: CommissionType = CommissionType.PERCENTAGE,
                lot_size: int = 1,
                margin_call: float = 0.5,  ## 50% of the cash lost
                leverage: float = 1.0  ## Leverage multiplier (1.0 = no leverage)
                ):
        """
        Initialize the backtester with strategy and configuration parameters.
        
        Args:
            strategy (Strategy): Trading strategy to backtest. Must implement
                the Strategy interface with init() and next() methods.
            cash (float, optional): Initial cash amount. Defaults to 10,000.
            commission (float, optional): Commission rate per trade. Defaults to 0.002 (0.2%).
            commission_type (CommissionType, optional): Type of commission calculation.
                Can be CommissionType.PERCENTAGE or CommissionType.CASH.
                Defaults to CommissionType.PERCENTAGE.
            lot_size (int, optional): Size of trading lots. Defaults to 1.
            margin_call (float, optional): Margin call threshold as fraction of
                cash value. Defaults to 0.5 (50%).
            leverage (float, optional): Leverage multiplier for position sizing.
                Defaults to 1.0 (no leverage). For example:
                - 2.0 = 2x leverage (control 2x the position with same cash)
                - 0.5 = half leverage (control half the position)
                
        Raises:
            ValueError: If strategy is None or commission rate is negative.
        """
        if leverage < 0.1:
            raise ValueError("leverage must be at least 0.1")
        if leverage > 100:
            raise ValueError("leverage cannot exceed 100")
        self.strategy = copy.deepcopy(strategy)
        self.cash = cash
        self.commission = commission
        self.commission_type = commission_type
        self.lot_size = lot_size
        self.margin_call = margin_call
        self.leverage = leverage
        source = self.strategy.positions[list(self.strategy.positions.keys())[0]].source
        self.PnLRecord = np.zeros(len(source.data['Close']), dtype=np.float64)

    def _reset_runtime_state(self) -> None:
        source = self.strategy.positions[list(self.strategy.positions.keys())[0]].source
        self.PnLRecord = np.zeros(len(source.data['Close']), dtype=np.float64)
        self.strategy.indicators = []

        for data_source in self.strategy.data.values():
            data_source.current_index = len(data_source.data)

        for broker in self.strategy.positions.values():
            broker.position = np.float64(0)
            broker.position_avg_price = np.float64(0)
            broker.cash = np.float64(0)
            broker.orders = []
            broker.complete_orders = []
            broker._i = 0
            broker.PnLRecord = np.full(len(broker.source.data['Close']), self.cash, dtype=np.float64)
            broker.cashRecord = []

    def run(self, progress_bar: bool = False) -> BacktestReport:
        """
        Execute the backtest for the configured strategy.
        
        This method runs the complete backtest simulation, iterating through
        all data points in the strategy's data sources, executing strategy logic,
        processing orders, and tracking performance metrics.
        
        Args:
            progress_bar (bool, optional): Whether to show a progress bar during
                backtest execution. Useful for long-running backtests.
                Defaults to False.
                
        Returns:
            BacktestReport: Object containing complete backtest results including:
                - Starting and final cash amounts
                - P&L record over time
                - List of all executed orders
                - Calculated performance metrics
                
        Note:
            This method modifies the internal state of the strategy and
            should not be called multiple times on the same instance
            without resetting.
        """
        self._reset_runtime_state()

        # Distribute the initial portfolio cash evenly across all symbols so that
        # the aggregate starting equity equals `self.cash`, regardless of the
        # number of data sources attached to the strategy. This avoids
        # double-counting cash when multiple symbols are used.
        n_positions = max(len(self.strategy.positions), 1)
        per_position_cash = np.float64(self.cash / n_positions)

        for key in self.strategy.positions.keys():
            broker = self.strategy.positions[key]
            broker.cash = per_position_cash
            broker.lot_size = self.lot_size
            broker.margin_call = self.margin_call
            broker.leverage = self.leverage
            broker.commision = np.float64(self.commission)
            broker.commision_type = self.commission_type

        self.strategy.init()

        for data_source in self.strategy.data.values():
            data_source.current_index = 0

        ## Simple backtesting loop
        for i in tqdm(range(0, max([len(i) for i in self.strategy.data.values()])), disable=(not progress_bar)):
            for val in self.strategy.data.values():
                val.current_index = i
            for val in self.strategy.positions.values():
                val._iterate(i)
            for item in self.strategy.indicators:
                # Make indicators time-aware in the same way as DataSource:
                # at step i, expose data up to and including index i.
                # Clamp to the underlying array length to avoid overflow.
                item._i = min(i + 1, item.shape[0])
            self.strategy.next()
        orders: list[Order] = []
        tradeRecord: list[np.float64] = []
        for val in self.strategy.positions.values():
            val.close()
            self.PnLRecord += val.PnLRecord
            cashRecord = np.array(val.cashRecord)
            trades = np.diff(cashRecord)
            tradeRecord.extend(trades)
            orders.extend(val.complete_orders)

        index = list(self.strategy.positions.values())[0].source.data['Close'].index
        return BacktestReport(
            starting_cash=np.float64(self.cash),
            final_cash=self.PnLRecord[-1],
            PnlRecord=pd.Series(self.PnLRecord, index=index),
            orders=orders,
            tradeRecord=tradeRecord,
            margin_call_events=[
                event
                for broker in self.strategy.positions.values()
                for event in getattr(broker, "margin_call_events", [])
            ] or None)
    
    def optimize(
        self,
        params: dict[str, range],
        constraint: Callable[[dict[str, Any]], bool] | None = None,
        objective: str = "sharpe",
        risk_tolerance: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """
        Perform a grid search over the provided parameter ranges.
        
        This method systematically tests all combinations of parameter values
        to find the optimal configuration for the trading strategy. Each
        parameter combination is backtested individually to evaluate performance.
        
        Args:
            params (dict[str, range]): Dictionary mapping strategy attribute names
                to iterables of candidate values. For example:
                ```python
                {
                    'fast_period': range(5, 21, 5),    # [5, 10, 15, 20]
                    'slow_period': range(20, 51, 10),  # [20, 30, 40, 50]
                    'threshold': np.linspace(0.01, 0.1, 10)
                }
                ```
            constraint (Callable[[dict[str, Any]], bool] | None, optional):
                Optional callable that takes a candidate parameter dict and returns
                True to evaluate the combo or False to skip it. Useful for enforcing
                logical constraints like ensuring fast_period < slow_period.
                Defaults to None (no constraints).
            objective (str, optional): BacktestReport attribute or computed metric to
                optimize. Defaults to "sharpe". Supports any attribute exposed by
                BacktestReport and the computed metrics "final_cash", "total_return",
                "sharpe", "max_drawdown", and "trades".
            risk_tolerance (dict[str, float] | None, optional): Optional maximum
                allowed values for candidate metrics. Any candidate that exceeds a
                threshold is discarded before scoring. For example,
                {"max_drawdown": 0.05} rejects strategies with drawdown above 5%.
                Defaults to None.
                
        Returns:
            OptimizationResult: Object containing:
                - best_params: Best parameter values found
                - train_report: BacktestReport for the best parameters
                - validate_report: None for single-split optimization
                - test_report: None for single-split optimization
                - train_metrics: Metrics computed for best parameters
                - validate_metrics: Empty dict for single-split optimization
                - test_metrics: Empty dict for single-split optimization
                - all_results: DataFrame with all parameter combinations
                   
        Raises:
            ValueError: If params is empty or contains parameters with no values.
            TypeError: If any parameter values are not iterable.
            
        Note:
            The optimization uses the selected objective as the primary selection
            criterion. If the objective is invalid (NaN), the candidate is skipped.
            
        Example:
            >>> bt = SimpleBacktester(strategy)  
            >>> best_params, best_report, results = bt.optimize({  
            ...     'fast_period': [5, 10, 20],  
            ...     'slow_period': [20, 50, 100]  
            ... }, constraint=lambda p: p['fast_period'] < p['slow_period'])  
            >>> print(f"Best parameters: {best_params}")  
            >>> print(f"Best Sharpe ratio: {best_report.periods_per_year}")  
        """
        if not params:
            raise ValueError("params must not be empty")

        keys = list(params.keys())
        value_lists = []
        for k in keys:
            vals = params[k]
            # Ensure iterability and materialize to list for cartesian product
            try:
                candidates = list(vals)
            except TypeError:
                raise TypeError(f"Parameter '{k}' must be iterable")
            if len(candidates) == 0:
                raise ValueError(f"Parameter '{k}' has no candidate values")
            value_lists.append(candidates)

        results_rows = []
        best_report = None
        best_params = None
        best_score = -np.inf

        valid_metrics = {"final_cash", "total_return", "sharpe", "max_drawdown", "trades"}

        # Use math.prod instead of len(list(itertools.product(...))) 
        # to avoid materializing all combinations in memory
        total_combos = math.prod(len(v) for v in value_lists)

        for combo in tqdm(itertools.product(*value_lists), total=(total_combos)):
            # Build parameter dict for this combo
            row_params = {k: v for k, v in zip(keys, combo)}

            # Apply optional constraint; skip combo if it returns False or raises
            if constraint is not None:
                try:
                    if not bool(constraint(row_params)):
                        continue
                except Exception:
                    # If the constraint itself errors, treat as invalid combo
                    continue

            # Fresh strategy copy per combo
            strat_copy = copy.deepcopy(self.strategy)
            for k, v in row_params.items():
                setattr(strat_copy, k, v)

            # Run a fresh backtest instance retaining runtime settings
            bt = SimpleBacktester(
                strat_copy,
                cash=self.cash,
                commission=self.commission,
                commission_type=self.commission_type,
                lot_size=self.lot_size,
                leverage=self.leverage,
            )
            report = bt.run(progress_bar=False)

            metrics = _compute_backtest_metrics(report)

            if not _risk_tolerance_passes(report, risk_tolerance):
                continue

            if objective in valid_metrics:
                score = metrics.get(objective)
            else:
                score = getattr(report, objective, None)
                if callable(score):
                    score = score()

            if score is None:
                raise AttributeError(f"BacktestReport does not expose objective '{objective}'")

            try:
                score = float(score)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                raise TypeError(f"Objective '{objective}' must be numeric")

            if not np.isfinite(score):
                continue

            row = dict(row_params)
            row.update(metrics)
            row["objective_score"] = score
            results_rows.append(row)

            if score > best_score:
                best_score = score
                best_params = {k: v for k, v in zip(keys, combo)}
                best_report = report

        results_df = pd.DataFrame(results_rows)

        # Sort results by composite score (Sharpe desc, then return, then cash)
        if not results_df.empty:
            results_df.sort_values(by=["objective_score"], ascending=False, inplace=True, kind="mergesort")

        # Compute metrics for best report if available
        best_metrics = {}
        if best_report is not None:
            best_metrics = _compute_backtest_metrics(best_report)

        return OptimizationResult(
            best_params=best_params or {},
            train_report=best_report,
            validate_report=None,
            test_report=None,
            train_metrics=best_metrics,
            validate_metrics={},
            test_metrics={},
            all_results=results_df
        )
    
    def optimize_parallel(
             self,
             params: dict[str, range],
             constraint: Callable[[dict[str, Any]], bool] | None = None,
             objective: str = "sharpe",
             risk_tolerance: dict[str, float] | None = None,
             workers: int | None = None,
             chunksize: int | str = "auto") -> OptimizationResult:
        """
        Perform parallel grid search over parameter ranges for optimization.
        
        This method is identical to optimize() but uses multiprocessing to
        distribute parameter combinations across multiple worker processes,
        significantly reducing computation time for large parameter spaces.
        
        Args:
            params (dict[str, range]): Dictionary mapping strategy attribute names
                to iterables of candidate values (same format as optimize()).
             constraint (Callable[[dict[str, Any]], bool] | None, optional):
                Optional callable for parameter constraints (same as optimize()).
                Defaults to None.
             objective (str, optional): BacktestReport attribute or computed metric to
                optimize. Defaults to "sharpe".
             risk_tolerance (dict[str, float] | None, optional): Optional maximum
                allowed metric values for candidate rejection before scoring.
                Defaults to None.
            workers (int | None, optional): Maximum number of worker processes to use.
                If None, defaults to min(os.cpu_count()-1, 4) to avoid overwhelming
                the system. Defaults to None.
            chunksize (int | str, optional): Chunk size for ProcessPoolExecutor.map.
                Can be an integer or "auto" for adaptive sizing based on total
                combinations and worker count. Smaller values provide better load
                balancing for many small tasks. Larger values reduce IPC overhead.
                Defaults to "auto" (previously 1).
                Auto-calculation: max(16, total_combos // (workers * 4))
                
        Returns:
            OptimizationResult: Object containing:
                - best_params: Best parameter values found
                - train_report: BacktestReport for the best parameters
                - validate_report: None for single-split optimization
                - test_report: None for single-split optimization
                - train_metrics: Metrics computed for best parameters
                - validate_metrics: Empty dict for single-split optimization
                - test_metrics: Empty dict for single-split optimization
                - all_results: DataFrame with all parameter combinations
                
        Raises:
            ValueError: If params is empty or contains parameters with no values.
            TypeError: If any parameter values are not iterable.
            
        Note:
            - This method creates separate processes, so the strategy must be
              picklable for multiprocessing to work.
            - The main process re-runs the best configuration to get the full
              BacktestReport (parallel workers only return summary metrics).
            - Uses ProcessPoolExecutor for true parallelism across CPU cores.
            - Memory usage scales with the number of workers as each worker
              maintains a copy of the strategy.
               
        Performance Tips:
            - For parameter spaces with many combinations (>1000), prefer
              optimize_parallel over optimize for better performance.
            - For small parameter spaces, optimize() may be faster due to
              lower multiprocessing overhead.
            - Monitor system memory usage as each worker maintains a full
              copy of the strategy and data.
            - Auto chunksize provides better throughput for large parameter spaces.
               
        Example:
            >>> bt = SimpleBacktester(strategy)  
            >>> # Use 4 workers for parallel optimization  
            >>> best_params, best_report, results = bt.optimize_parallel(  
            ...     {'period1': range(5, 50, 5), 'period2': range(20, 100, 10)},  
            ...     workers=4  
            ... )  
        """
        import concurrent.futures
        import pickle
        import math

        if not params:
            raise ValueError("params must not be empty")

        keys = list(params.keys())
        value_lists = []
        for k in keys:
            vals = params[k]
            try:
                candidates = list(vals)
            except TypeError:
                raise TypeError(f"Parameter '{k}' must be iterable")
            if len(candidates) == 0:
                raise ValueError(f"Parameter '{k}' has no candidate values")
            value_lists.append(candidates)

        # determine total combos without materializing them using math.prod
        # (previously used len(list(itertools.product(...))) which materialized all combos)
        total_combos = math.prod(len(v) for v in value_lists)

        # choose worker count conservatively to avoid RAM hogging
        cpu_count = os.cpu_count() or 1
        if workers is None:
            workers = max(1, min(cpu_count - 1, 4))
        else:
            workers = max(1, int(workers))

        # Adaptive chunksize calculation
        # Previous default was chunksize=1 which causes high IPC overhead
        # New default "auto" uses: max(16, total_combos // (workers * 4))
        if chunksize == "auto":
            chunksize = max(16, total_combos // (workers * 4))
        else:
            chunksize = max(1, int(chunksize))

        # prepare iterable of param dicts as sequences of items (so pickling is slightly cheaper)
        # Also pre-compute constraint results to avoid repeated checks
        def _param_items_iter():
            for combo in itertools.product(*value_lists):
                row_params = {k: v for k, v in zip(keys, combo)}
                if constraint is not None:
                    try:
                        if not bool(constraint(row_params)):
                            continue
                    except Exception:
                        continue
                # yield as tuple of items for stable order and smaller IPC
                yield tuple(row_params.items())

        # pickle the base strategy once and send bytes to worker initializer
        pickled_strategy = pickle.dumps(self.strategy)

        results_rows = []
        # Use ProcessPoolExecutor with worker initializer so each worker holds
        # exactly one copy of the strategy in memory.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(
                pickled_strategy,
                self.cash,
                self.commission,
                self.commission_type,
                self.lot_size,
                self.leverage,
            ),
        ) as exe:
            # map the worker over param item tuples
            # use list() on map to iterate with tqdm and collect results
            it = exe.map(_worker_eval, _param_items_iter(), chunksize=chunksize)
            # iterate with progress display
            for res in tqdm(it, total=total_combos, disable=(total_combos <= 1)):
                results_rows.append(res)

        valid_metrics = {"final_cash", "total_return", "sharpe", "max_drawdown", "trades"}
        filtered_rows = []
        for row in results_rows:
            row_params = row["params"]
            if risk_tolerance is not None:
                if any(float(row.get(metric, np.inf)) > float(limit) for metric, limit in risk_tolerance.items() if limit is not None):
                    continue
            if objective in valid_metrics:
                score = row.get(objective)
            else:
                score = row.get(objective)
            if score is None or not np.isfinite(float(score)):
                continue
            row["objective_score"] = float(score)
            filtered_rows.append(row)

        # Build DataFrame of small metrics returned from workers
        results_df = pd.DataFrame(filtered_rows)
        if not results_df.empty:
            results_df.sort_values(by=["objective_score"], ascending=False, inplace=True, kind="mergesort")

        # Determine best params from results_df if any
        if results_df.empty:
            return OptimizationResult(
                best_params={},
                train_report=None,
                validate_report=None,
                test_report=None,
                train_metrics={},
                validate_metrics={},
                test_metrics={},
                all_results=results_df
            )

        best_row = results_df.iloc[0]
        best_params = best_row["params"]

        # Re-run full backtest locally in main process for best_params to obtain
        # the full BacktestReport (includes PnLRecord and orders)
        strat_copy = copy.deepcopy(self.strategy)
        for k, v in best_params.items():
            setattr(strat_copy, k, v)
        bt = SimpleBacktester(
            strat_copy,
            cash=self.cash,
            commission=self.commission,
            commission_type=self.commission_type,
            lot_size=self.lot_size,
            leverage=self.leverage,
        )
        best_report = bt.run(progress_bar=False)

        # Compute metrics for best report
        best_metrics = _compute_backtest_metrics(best_report)

        return OptimizationResult(
            best_params=best_params,
            train_report=best_report,
            validate_report=None,
            test_report=None,
            train_metrics=best_metrics,
            validate_metrics={},
            test_metrics={},
                all_results=results_df
            )

    def optimize_optuna(
        self,
        param_space: dict[str, tuple[Any, Any] | list[Any]],
        n_trials: int = 100,
        objective: str = "sharpe",
        risk_tolerance: dict[str, float] | None = None,
        constraint: Callable[[dict[str, Any]], bool] | None = None,
        timeout: int | None = None,
        random_seed: int | None = None,
        workers: int | None = None,
        progress_bar: bool = True,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using Optuna (Bayesian optimization).
        
        This method uses Optuna's optimization framework with TPE (Tree-structured
        Parzen Estimator) sampler for intelligent parameter search. It typically
        finds better solutions than grid search with fewer evaluations.
        
        The method supports:
        - Continuous parameter ranges (sampled uniformly)
        - Discrete/categorical parameter lists
        - Early pruning of unpromising trials
        - Parallel execution for faster optimization
        
        Args:
            param_space (dict[str, tuple[Any, Any] | list[Any]]): Parameter search space.
                Can be:
                - Continuous range: (min, max) tuple for uniform sampling
                - Discrete list: [val1, val2, ...] for categorical sampling
                Example: {'period': (5, 50), 'threshold': [0.01, 0.02, 0.05]}
            n_trials (int, optional): Maximum number of optimization trials.
                Defaults to 100.
            objective (str, optional): Metric to optimize. Defaults to "sharpe".
                Supports: "final_cash", "total_return", "sharpe", "max_drawdown", "trades".
            risk_tolerance (dict[str, float] | None, optional): Maximum allowed values
                for risk metrics. Trials exceeding thresholds are pruned. Defaults to None.
            constraint (Callable[[dict[str, Any]], bool] | None, optional): Optional
                callable to enforce parameter constraints. Defaults to None.
            timeout (int | None, optional): Maximum time in seconds for optimization.
                Defaults to None (no limit).
            random_seed (int | None, optional): Random seed for reproducibility.
                Defaults to None.
            workers (int | None, optional): Number of parallel workers for Optuna
                study. Defaults to None (sequential).
            progress_bar (bool, optional): Whether to show progress bar. Defaults to True.
            verbose (bool, optional): Whether to show Optuna trial logs. Defaults to False
                (suppresses verbose output).
            
        Returns:
            OptimizationResult: Object containing:
                - best_params: Best parameter values found
                - train_report: BacktestReport for best parameters (None for Optuna)
                - validate_report: None
                - test_report: None
                - train_metrics: Metrics for best parameters
                - validate_metrics: Empty dict
                - test_metrics: Empty dict
                - all_results: DataFrame with all trial results
                
        Performance Notes:
            - Optuna typically finds good solutions in 50-200 trials
            - For 10,000+ grid combos, Optuna can be 50-100x faster
            - Use workers > 1 for parallel trial evaluation
            - Pruning callbacks significantly speed up optimization
            
        Example:
            >>> # Optimize with continuous and discrete parameters
            >>> result = bt.optimize_optuna({
            ...     'fast_period': (5, 50),      # Continuous: 5-50
            ...     'slow_period': [20, 30, 50], # Discrete: pick one
            ...     'threshold': (0.01, 0.1),    # Continuous: 1%-10%
            ... }, n_trials=100)
            >>> print(f"Best params: {result.best_params}")
            >>> print(f"Best Sharpe: {result.train_metrics['sharpe']}")
            
        Note:
            Requires optuna package: pip install optuna
        """
        try:
            import optuna
            # Control optuna verbosity - logs every trial by default
            optuna.logging.set_verbosity(
                optuna.logging.INFO if verbose else optuna.logging.WARNING
            )
        except ImportError:
            raise ImportError(
                "optuna is required for optimize_optuna. "
                "Install it with: pip install optuna"
            )
        
        # Check for invalid objective
        valid_metrics = {"final_cash", "total_return", "sharpe", "max_drawdown", "trades"}
        if objective not in valid_metrics:
            raise ValueError(
                f"objective must be one of {valid_metrics}, got '{objective}'"
            )
        
        # Convert param_space to Optuna distribution format
        param_names = list(param_space.keys())
        
        def _create_objective(
            strategy_template: Strategy,
            cash: float,
            commission: float,
            commission_type: CommissionType,
            lot_size: int,
            leverage: float,
            objective: str,
            risk_tolerance: dict[str, float] | None,
            constraint: Callable[[dict[str, Any]], bool] | None,
        ):
            """Create objective function for Optuna."""
            
            def objective_fn(trial: optuna.Trial) -> float:
                # Sample parameters based on space definition
                params = {}
                for name, space in param_space.items():
                    if isinstance(space, (list, tuple)) and len(space) == 2:
                        # Check if it's a range (numeric) or discrete list
                        if all(isinstance(v, (int, float)) for v in space):
                            # Numeric range: treat as continuous if range > 10 values
                            try:
                                if len(space) == 2 and all(isinstance(v, (int, float)) for v in space):
                                    # Check if values suggest discrete or continuous
                                    if all(isinstance(v, int) for v in space) and len(space) == 2:
                                        # Check if it's meant to be discrete (like range values)
                                        pass
                            except:
                                pass
                            # Try as discrete list first
                            try:
                                # Assume discrete if second value is list
                                if isinstance(space[1], list):
                                    choice = trial.suggest_categorical(name, space)
                                    params[name] = choice
                                else:
                                    # Continuous range
                                    low, high = sorted(space)
                                    if all(isinstance(v, int) for v in space):
                                        params[name] = trial.suggest_int(name, int(low), int(high))
                                    else:
                                        params[name] = trial.suggest_float(name, float(low), float(high))
                            except:
                                # Treat as continuous
                                low, high = sorted(space)
                                if all(isinstance(v, int) for v in space):
                                    params[name] = trial.suggest_int(name, int(low), int(high))
                                else:
                                    params[name] = trial.suggest_float(name, float(low), float(high))
                        else:
                            # Discrete list
                            params[name] = trial.suggest_categorical(name, space)
                    else:
                        # Direct list of choices
                        params[name] = trial.suggest_categorical(name, list(space))
                
                # Apply constraint if provided
                if constraint is not None:
                    try:
                        if not bool(constraint(params)):
                            raise optuna.TrialPruned("Constraint violated")
                    except optuna.TrialPruned:
                        raise
                    except Exception:
                        raise optuna.TrialPruned("Constraint error")
                
                # Create strategy copy and apply params
                strat_copy = copy.deepcopy(strategy_template)
                for k, v in params.items():
                    setattr(strat_copy, k, v)
                
                # Run backtest
                bt = SimpleBacktester(
                    strat_copy,
                    cash=cash,
                    commission=commission,
                    commission_type=commission_type,
                    lot_size=lot_size,
                    leverage=leverage,
                )
                report = bt.run(progress_bar=False)
                
                # Compute metrics
                metrics = _compute_backtest_metrics(report)
                
                # Apply risk tolerance filter
                if risk_tolerance is not None:
                    if not _risk_tolerance_passes(report, risk_tolerance):
                        raise optuna.TrialPruned("Risk tolerance exceeded")
                
                # Get objective score
                if objective in valid_metrics:
                    score = metrics.get(objective)
                else:
                    score = getattr(report, objective, None)
                    if callable(score):
                        score = score()
                
                if score is None or not np.isfinite(float(score)):  # type: ignore[arg-type]
                    raise optuna.TrialPruned("Invalid objective score")
                
                return float(score)  # type: ignore[arg-type]
            
            return objective_fn
        
        # Create and configure Optuna study
        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )
        
        # Create objective function with closure
        obj_fn = _create_objective(
            strategy_template=self.strategy,
            cash=self.cash,
            commission=self.commission,
            commission_type=self.commission_type,
            lot_size=self.lot_size,
            leverage=self.leverage,
            objective=objective,
            risk_tolerance=risk_tolerance,
            constraint=constraint,
        )
        
        # Run optimization
        show_progress = progress_bar and workers is None  # Only if sequential
        
        if workers is not None and workers > 1:
            # Parallel execution using joblib backend
            study.optimize(
                obj_fn,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=workers,
                show_progress_bar=progress_bar,
            )
        else:
            # Sequential execution
            study.optimize(
                obj_fn,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=show_progress,
            )
        
        # Get best params
        best_params = study.best_params
        
        # Build results DataFrame from completed trials
        results_rows = []
        for trial in study.trials:
            if trial.value is not None and trial.value > -np.inf:
                row = dict(trial.params)
                row["objective_score"] = trial.value
                row["state"] = trial.state.name
                results_rows.append(row)
        
        results_df = pd.DataFrame(results_rows)
        if not results_df.empty:
            results_df.sort_values(by=["objective_score"], ascending=False, inplace=True, kind="mergesort")
        
        # Run full backtest with best params for detailed report
        strat_copy = copy.deepcopy(self.strategy)
        for k, v in best_params.items():
            setattr(strat_copy, k, v)
        
        bt = SimpleBacktester(
            strat_copy,
            cash=self.cash,
            commission=self.commission,
            commission_type=self.commission_type,
            lot_size=self.lot_size,
            leverage=self.leverage,
        )
        best_report = bt.run(progress_bar=False)
        best_metrics = _compute_backtest_metrics(best_report)
        
        return OptimizationResult(
            best_params=best_params,
            train_report=best_report,
            validate_report=None,
            test_report=None,
            train_metrics=best_metrics,
            validate_metrics={},
            test_metrics={},
            all_results=results_df,
        )

    def optimize_with_split(
        self,
        params: dict[str, Any],
        constraint: Callable[[dict[str, Any]], bool] | None = None,
        objective: str = "sharpe",
        risk_tolerance: dict[str, float] | None = None,
        train_ratio: float = 0.6,
        validate_ratio: float = 0.2,
        test_ratio: float = 0.2,
        selection_criterion: str = "validate",
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using train/validate/test splits.
        
        This method implements ML-style optimization with three data splits:
        - Training set: Used to fit strategy parameters
        - Validation set: Used to select the best parameters
        - Test set: Used for final out-of-sample evaluation
        
        This approach helps prevent overfitting by evaluating generalization
        performance on held-out data before final selection.
        
        Args:
            params (dict[str, range]): Dictionary mapping strategy attribute names
                to iterables of candidate values (same format as optimize()).
            constraint (Callable[[dict[str, Any]], bool] | None, optional):
                Optional callable for parameter constraints. Defaults to None.
            objective (str, optional): Metric to optimize. Defaults to "sharpe".
                Supports same metrics as optimize().
            risk_tolerance (dict[str, float] | None, optional): Optional maximum
                allowed metric values. Defaults to None.
            train_ratio (float, optional): Fraction of data for training.
                Defaults to 0.6 (60%).
            validate_ratio (float, optional): Fraction of data for validation.
                Defaults to 0.2 (20%).
            test_ratio (float, optional): Fraction of data for testing.
                Defaults to 0.2 (20%).
            selection_criterion (str, optional): Which split to use for final
                parameter selection. Options: "train", "validate", "test".
                Defaults to "validate".
                
        Returns:
            OptimizationResult: Object containing:
                - best_params: Best parameters found
                - train_report: BacktestReport for training data
                - validate_report: BacktestReport for validation data
                - test_report: BacktestReport for test data
                - train_metrics: Metrics computed on training data
                - validate_metrics: Metrics computed on validation data
                - test_metrics: Metrics computed on test data
                - all_results: DataFrame with all results
                
        Raises:
            ValueError: If split ratios don't sum to 1.0 or selection_criterion
                is invalid.
                
        Example:
            >>> bt = SimpleBacktester(strategy)
            >>> result = bt.optimize_with_split(
            ...     {'fast_period': [5, 10, 15], 'slow_period': [20, 30, 50]},
            ...     selection_criterion='validate'
            ... )
            >>> print(f"Best params: {result.best_params}")
            >>> print(f"Train Sharpe: {result.train_metrics['sharpe']}")
            >>> print(f"Validate Sharpe: {result.validate_metrics['sharpe']}")
            >>> print(f"Test Sharpe: {result.test_metrics['sharpe']}")
        """
        from ..datasource import DataSource

        # Validate selection criterion
        valid_criteria = {"train", "validate", "test"}
        if selection_criterion not in valid_criteria:
            raise ValueError(
                f"selection_criterion must be one of {valid_criteria}, "
                f"got '{selection_criterion}'"
            )
        
        # Get data length from the strategy's data source
        source = self.strategy.positions[list(self.strategy.positions.keys())[0]].source
        data_length = len(source.data)
        
        # Create the split
        split = create_train_validate_test_split(
            data_length,
            train_ratio,
            validate_ratio,
            test_ratio
        )
        
        # Prepare parameter combinations
        if not params:
            raise ValueError("params must not be empty")

        keys = list(params.keys())
        value_lists = []
        for k in keys:
            vals = params[k]
            try:
                candidates = list(vals)
            except TypeError:
                raise TypeError(f"Parameter '{k}' must be iterable")
            if len(candidates) == 0:
                raise ValueError(f"Parameter '{k}' has no candidate values")
            value_lists.append(candidates)

        # Store results for each split
        train_results = []
        validate_results = []
        test_results = []
        
        valid_metrics = {"final_cash", "total_return", "sharpe", "max_drawdown", "trades"}
        
        # Use math.prod instead of len(list(...)) to avoid materializing all combos
        total_combos = math.prod(len(v) for v in value_lists)
        
        # Create a modified strategy that uses data slices
        def create_split_strategy(params_dict: dict, split_mode: DataSplitMode):
            """Create a strategy copy with data sliced to the specified split."""
            strat_copy = copy.deepcopy(self.strategy)
            for k, v in params_dict.items():
                # Convert float to int if the strategy expects integer parameters
                if isinstance(v, float) and v == int(v):
                    v = int(v)
                setattr(strat_copy, k, v)
            
            # Slice each data source to the appropriate split
            for key, broker in strat_copy.positions.items():
                source = broker.source
                if split_mode == DataSplitMode.TRAIN:
                    start, end = split.train_start, split.train_end
                elif split_mode == DataSplitMode.VALIDATE:
                    start, end = split.validate_start, split.validate_end
                else:  # TEST
                    start, end = split.test_start, split.test_end
                
                # Create a new data source with sliced data
                sliced_df = source.data.iloc[start:end].copy()
                new_source = DataSource(sliced_df)
                broker.source = new_source
                # Also update the strategy's data dictionary
                strat_copy.data[key] = new_source
            
            return strat_copy, split_mode

        # Run optimization for each split
        for combo in tqdm(itertools.product(*value_lists), total=total_combos, desc="Optimizing"):
            row_params = {k: v for k, v in zip(keys, combo)}
            
            # Apply constraint
            if constraint is not None:
                try:
                    if not bool(constraint(row_params)):
                        continue
                except Exception:
                    continue
            
            # Evaluate on all three splits
            for mode in [DataSplitMode.TRAIN, DataSplitMode.VALIDATE, DataSplitMode.TEST]:
                strat_copy, _ = create_split_strategy(row_params, mode)
                
                bt = SimpleBacktester(
                    strat_copy,
                    cash=self.cash,
                    commission=self.commission,
                    commission_type=self.commission_type,
                    lot_size=self.lot_size,
                    leverage=self.leverage,
                )
                report = bt.run(progress_bar=False)
                metrics = _compute_backtest_metrics(report)
                
                # Apply risk tolerance filter
                if risk_tolerance is not None:
                    if not _risk_tolerance_passes(report, risk_tolerance):
                        continue
                
                # Compute objective score
                if objective in valid_metrics:
                    score = metrics.get(objective)
                else:
                    score = getattr(report, objective, None)
                    if callable(score):
                        score = score()
                
                if score is None or not np.isfinite(float(score)):  # type: ignore[arg-type]
                    continue
                
                row = dict(row_params)
                row["objective_score"] = float(score)  # type: ignore[arg-type]
                row.update(metrics)
                
                if mode == DataSplitMode.TRAIN:
                    train_results.append(row)
                elif mode == DataSplitMode.VALIDATE:
                    validate_results.append(row)
                else:
                    test_results.append(row)
        
        # Create DataFrames
        train_df = pd.DataFrame(train_results) if train_results else pd.DataFrame()
        validate_df = pd.DataFrame(validate_results) if validate_results else pd.DataFrame()
        test_df = pd.DataFrame(test_results) if test_results else pd.DataFrame()
        
        # Select best parameters based on selection criterion
        if selection_criterion == "validate" and not validate_df.empty:
            validate_df_sorted = validate_df.sort_values(
                by=["objective_score"], ascending=False, kind="mergesort"
            )
            best_idx = validate_df_sorted.index[0]
            best_params = {k: validate_df.loc[best_idx, k] for k in keys}
            best_validate_score = validate_df.loc[best_idx, "objective_score"]
        elif selection_criterion == "train" and not train_df.empty:
            train_df_sorted = train_df.sort_values(
                by=["objective_score"], ascending=False, kind="mergesort"
            )
            best_idx = train_df_sorted.index[0]
            best_params = {k: train_df.loc[best_idx, k] for k in keys}
            best_validate_score = train_df.loc[best_idx, "objective_score"]
        elif selection_criterion == "test" and not test_df.empty:
            test_df_sorted = test_df.sort_values(
                by=["objective_score"], ascending=False, kind="mergesort"
            )
            best_idx = test_df_sorted.index[0]
            best_params = {k: test_df.loc[best_idx, k] for k in keys}
            best_validate_score = test_df.loc[best_idx, "objective_score"]
        else:
            best_params = {}
            best_validate_score = -np.inf
        
            # Get full reports for best parameters
        train_report = None
        validate_report = None
        test_report = None
        train_metrics = {}
        validate_metrics = {}
        test_metrics = {}
        
        if best_params:
            # Run full backtests for best parameters on each split
            for mode, report_attr, metrics_attr in [
                (DataSplitMode.TRAIN, 'train_report', 'train_metrics'),
                (DataSplitMode.VALIDATE, 'validate_report', 'validate_metrics'),
                (DataSplitMode.TEST, 'test_report', 'test_metrics'),
            ]:
                strat_copy, _ = create_split_strategy(best_params, mode)
                bt = SimpleBacktester(
                    strat_copy,
                    cash=self.cash,
                    commission=self.commission,
                    commission_type=self.commission_type,
                    lot_size=self.lot_size,
                    leverage=self.leverage,
                )
                report = bt.run(progress_bar=False)
                metrics = _compute_backtest_metrics(report)
                
                if mode == DataSplitMode.TRAIN:
                    train_report = report
                    train_metrics = metrics
                elif mode == DataSplitMode.VALIDATE:
                    validate_report = report
                    validate_metrics = metrics
                else:
                    test_report = report
                    test_metrics = metrics
        
        # Combine all results
        all_results = pd.DataFrame()
        if not train_df.empty:
            train_df_copy = train_df.copy()
            train_df_copy["split"] = "train"
            all_results = pd.concat([all_results, train_df_copy], ignore_index=True)
        if not validate_df.empty:
            validate_df_copy = validate_df.copy()
            validate_df_copy["split"] = "validate"
            all_results = pd.concat([all_results, validate_df_copy], ignore_index=True)
        if not test_df.empty:
            test_df_copy = test_df.copy()
            test_df_copy["split"] = "test"
            all_results = pd.concat([all_results, test_df_copy], ignore_index=True)
        
        return OptimizationResult(
            best_params=best_params,
            train_report=train_report,
            validate_report=validate_report,
            test_report=test_report,
            train_metrics=train_metrics,
            validate_metrics=validate_metrics,
            test_metrics=test_metrics,
            all_results=all_results
        )

    def optimize_gradient_descent(
        self,
        param_init: dict[str, float],
        param_bounds: dict[str, tuple[float, float]],
        objective: str = "sharpe",
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        momentum: float = 0.9,
        train_ratio: float = 0.7,
        validate_ratio: float = 0.15,
        test_ratio: float = 0.15,
        selection_criterion: str = "validate",
        progress_bar: bool = True,
        integer_params: set[str] | None = None,
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using gradient descent.
        
        This method performs gradient-based optimization on continuous
        strategy parameters, similar to machine learning workflows. It uses
        train/validate/test splits to prevent overfitting and select the best
        model based on validation performance.
        
        The optimization computes numerical gradients by evaluating small
        perturbations around the current parameter values.
        
        Args:
            param_init (dict[str, float]): Initial parameter values.
            param_bounds (dict[str, tuple[float, float]]): Bounds for each
                parameter as (min, max) tuples.
            objective (str, optional): Metric to optimize. Defaults to "sharpe".
                Supports same metrics as optimize().
            learning_rate (float, optional): Step size for gradient descent.
                Defaults to 0.01.
            max_iterations (int, optional): Maximum number of iterations.
                Defaults to 100.
            tolerance (float, optional): Convergence tolerance. Optimization
                stops when gradient magnitude falls below this threshold.
                Defaults to 1e-6.
            momentum (float, optional): Momentum factor for accelerated
                descent. Defaults to 0.9.
            train_ratio (float, optional): Fraction of data for training.
                Defaults to 0.7 (70%).
            validate_ratio (float, optional): Fraction of data for validation.
                Defaults to 0.15 (15%).
            test_ratio (float, optional): Fraction of data for testing.
                Defaults to 0.15 (15%).
            selection_criterion (str, optional): Which split to use for final
                parameter selection. Options: "train", "validate", "test".
                Defaults to "validate".
            progress_bar (bool, optional): Whether to show progress bar.
                Defaults to True.
            integer_params (set[str] | None, optional): Set of parameter names
                that should be treated as integers. These parameters will be
                rounded to the nearest integer after each gradient update.
                Defaults to None (all parameters are continuous).
                
        Returns:
            OptimizationResult: Object containing:
                - best_params: Optimized parameter values
                - train_report: BacktestReport for training data
                - validate_report: BacktestReport for validation data
                - test_report: BacktestReport for test data
                - train_metrics: Metrics computed on training data
                - validate_metrics: Metrics computed on validation data
                - test_metrics: Metrics computed on test data
                - all_results: DataFrame with iteration history
                
        Example:
            >>> # Optimize with integer parameters
            >>> result = bt.optimize_gradient_descent(
            ...     param_init={'fast_period': 10.0, 'slow_period': 30.0},
            ...     param_bounds={
            ...         'fast_period': (2.0, 50.0),
            ...         'slow_period': (10.0, 100.0)
            ...     },
            ...     integer_params={'fast_period', 'slow_period'},
            ...     learning_rate=0.05,
            ...     max_iterations=50
            ... )
            >>> print(f"Optimized params: {result.best_params}")
            >>> print(f"Final validation Sharpe: {result.validate_metrics['sharpe']}")
        """
        from ..datasource import DataSource

        if integer_params is None:
            integer_params = set()
        # Validate selection criterion
        valid_criteria = {"train", "validate", "test"}
        if selection_criterion not in valid_criteria:
            raise ValueError(
                f"selection_criterion must be one of {valid_criteria}, "
                f"got '{selection_criterion}'"
            )
        
        # Validate parameters
        if not param_init:
            raise ValueError("param_init must not be empty")
        if set(param_init.keys()) != set(param_bounds.keys()):
            raise ValueError("param_init and param_bounds must have the same keys")
        
        # Get data length from the strategy's data source
        source = self.strategy.positions[list(self.strategy.positions.keys())[0]].source
        data_length = len(source.data)
        
        # Create the split
        split = create_train_validate_test_split(
            data_length,
            train_ratio,
            validate_ratio,
            test_ratio
        )
        
        valid_metrics = {"final_cash", "total_return", "sharpe", "max_drawdown", "trades"}
        
        # Helper to create sliced strategy
        def create_split_strategy(params_dict: dict, split_mode: DataSplitMode):
            """Create a strategy copy with data sliced to the specified split."""
            strat_copy = copy.deepcopy(self.strategy)
            for k, v in params_dict.items():
                # Convert float to int if the strategy expects integer parameters
                if isinstance(v, float) and v == int(v):
                    v = int(v)
                setattr(strat_copy, k, v)
            
            # Slice each data source to the appropriate split
            for key, broker in strat_copy.positions.items():
                source = broker.source
                if split_mode == DataSplitMode.TRAIN:
                    start, end = split.train_start, split.train_end
                elif split_mode == DataSplitMode.VALIDATE:
                    start, end = split.validate_start, split.validate_end
                else:  # TEST
                    start, end = split.test_start, split.test_end
                
                sliced_df = source.data.iloc[start:end].copy()
                new_source = DataSource(sliced_df)
                broker.source = new_source
                # Also update the strategy's data dictionary
                strat_copy.data[key] = new_source
            
            return strat_copy
        
        # Function to evaluate parameters on a specific split
        def evaluate_params(params_dict: dict, split_mode: DataSplitMode) -> float:
            """Evaluate objective function on specified split."""
            strat_copy = create_split_strategy(params_dict, split_mode)
            bt = SimpleBacktester(
                strat_copy,
                cash=self.cash,
                commission=self.commission,
                commission_type=self.commission_type,
                lot_size=self.lot_size,
                leverage=self.leverage,
            )
            report = bt.run(progress_bar=False)
            metrics = _compute_backtest_metrics(report)
            
            if objective in valid_metrics:
                score = metrics.get(objective)
            else:
                score = getattr(report, objective, None)
                if callable(score):
                    score = score()
            
            if score is None or not np.isfinite(float(score)):  # type: ignore[arg-type]
                return -np.inf
            
            return float(score)  # type: ignore[arg-type]
        
        # Compute numerical gradient
        def compute_gradient(params: dict, eps: float = 1e-5) -> dict:
            """Compute numerical gradient using central differences."""
            grad = {}
            for key in params:
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus[key] = params[key] + eps
                params_minus[key] = params[key] - eps
                
                # Use validation set for gradient computation
                f_plus = evaluate_params(params_plus, DataSplitMode.VALIDATE)
                f_minus = evaluate_params(params_minus, DataSplitMode.VALIDATE)
                
                grad[key] = (f_plus - f_minus) / (2 * eps)
            
            return grad
        
        # Gradient descent optimization
        current_params = param_init.copy()
        velocities = {k: 0.0 for k in current_params}
        
        iteration_history = []
        best_params = current_params.copy()
        best_score = -np.inf
        
        param_names = list(current_params.keys())
        
        iterator = range(max_iterations)
        if progress_bar:
            iterator = tqdm(iterator, desc="Gradient Descent")
        
        for iteration in iterator:
            # Compute gradient
            gradient = compute_gradient(current_params)
            
            # Check for convergence (gradient magnitude)
            grad_magnitude = np.sqrt(sum(g**2 for g in gradient.values()))
            if grad_magnitude < tolerance:
                if progress_bar:
                    print(f"\nConverged at iteration {iteration}")
                break
            
            # Update velocities with momentum
            for key in param_names:
                velocities[key] = momentum * velocities[key] - learning_rate * gradient[key]
            
            # Update parameters
            for key in param_names:
                current_params[key] += velocities[key]
                
                # Apply bounds
                min_val, max_val = param_bounds[key]
                current_params[key] = np.clip(current_params[key], min_val, max_val)
                
                # Round integer parameters to nearest integer
                if key in integer_params:
                    current_params[key] = round(current_params[key])
            
            # Evaluate on all splits
            train_score = evaluate_params(current_params, DataSplitMode.TRAIN)
            validate_score = evaluate_params(current_params, DataSplitMode.VALIDATE)
            test_score = evaluate_params(current_params, DataSplitMode.TEST)
            
            # Track best parameters based on selection criterion
            if selection_criterion == "validate" and validate_score > best_score:
                best_score = validate_score
                best_params = current_params.copy()
            elif selection_criterion == "train" and train_score > best_score:
                best_score = train_score
                best_params = current_params.copy()
            elif selection_criterion == "test" and test_score > best_score:
                best_score = test_score
                best_params = current_params.copy()
            
            # Record iteration history
            row = current_params.copy()
            row["iteration"] = iteration
            row["train_score"] = train_score
            row["validate_score"] = validate_score
            row["test_score"] = test_score
            row["gradient_magnitude"] = grad_magnitude
            iteration_history.append(row)
        
        # Create history DataFrame
        history_df = pd.DataFrame(iteration_history)
        
        # Get final reports for best parameters
        train_report = None
        validate_report = None
        test_report = None
        train_metrics = {}
        validate_metrics = {}
        test_metrics = {}
        
        for mode, report_attr, metrics_attr in [
            (DataSplitMode.TRAIN, 'train_report', 'train_metrics'),
            (DataSplitMode.VALIDATE, 'validate_report', 'validate_metrics'),
            (DataSplitMode.TEST, 'test_report', 'test_metrics'),
        ]:
            strat_copy = create_split_strategy(best_params, mode)
            bt = SimpleBacktester(
                strat_copy,
                cash=self.cash,
                commission=self.commission,
                commission_type=self.commission_type,
                lot_size=self.lot_size,
                leverage=self.leverage,
            )
            report = bt.run(progress_bar=False)
            metrics = _compute_backtest_metrics(report)
            
            if mode == DataSplitMode.TRAIN:
                train_report = report
                train_metrics = metrics
            elif mode == DataSplitMode.VALIDATE:
                validate_report = report
                validate_metrics = metrics
            else:
                test_report = report
                test_metrics = metrics
        
        return OptimizationResult(
            best_params=best_params,
            train_report=train_report,
            validate_report=validate_report,
            test_report=test_report,
            train_metrics=train_metrics,
            validate_metrics=validate_metrics,
            test_metrics=test_metrics,
            all_results=history_df
        )

    def monte_carlo(
        self,
        simulations: int = 100,
        mode: MonteCarloMode | str = MonteCarloMode.BOTH,
        seed: int | None = None,
        progress_bar: bool = False,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on the strategy.
        
        This method runs multiple simulations to test strategy robustness using
        either trade order randomization, price path resampling, or both.
        
        Args:
            simulations (int, optional): Number of simulations to run. Defaults to 100.
            mode (MonteCarloMode | str, optional): Simulation mode. Options:
                - "trade_order": Randomize trade execution order
                - "price_path": Resample price returns to create synthetic paths
                - "both": Run both analyses and combine results
                Defaults to "both".
            seed (int | None, optional): Random seed for reproducibility.
                Defaults to None.
            progress_bar (bool, optional): Whether to show progress bar during simulation.
                Defaults to False.
        
        Returns:
            MonteCarloResult: Object containing:
                - equity_curves: List of equity curves from each simulation
                - summary_stats: Mean, std, min, max of final returns
                - percentile_results: 5th, 25th, 50th, 75th, 95th percentiles
                - plot(): Visualization method for the "spaghetti plot"
        
        Example:
            >>> from quantex import SimpleBacktester, CSVDataSource
            >>> source = CSVDataSource("data.csv")
            >>> # Create and configure strategy
            >>> bt = SimpleBacktester(strategy, cash=10000)
            >>> result = bt.monte_carlo(simulations=500, mode="both")
            >>> print(result)  # Print summary statistics
            >>> result.plot()  # Show spaghetti plot
        
        Note:
            - Trade order randomization shuffles when trades execute while keeping the same trades
            - Price path resampling creates synthetic market scenarios from historical returns
            - Both methods help identify if strategy performance is robust or dependent on specific conditions
        """
        # Import here to avoid circular imports
        from .montecarlo import monte_carlo as _monte_carlo
        
        return _monte_carlo(
            self,
            simulations=simulations,
            mode=mode,
            seed=seed,
            progress_bar=progress_bar,
        )
