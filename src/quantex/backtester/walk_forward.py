"""
Walk-Forward Optimization for Trading Strategies.

This module provides walk-forward analysis capabilities for evaluating
trading strategies in a more realistic out-of-sample manner. Walk-forward
optimization uses rolling windows to test parameter stability over time.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class WalkForwardWindow:
    """
    Container for a single walk-forward window.
    
    Attributes:
        window_index (int): Index of this window (0-based).
        train_start (int): Starting index for training data.
        train_end (int): Ending index for training data.
        test_start (int): Starting index for test data.
        test_end (int): Ending index for test data.
        train_periods (int): Number of periods in training.
        test_periods (int): Number of periods in testing.
        best_params (dict): Best parameters found during training.
        train_metrics (dict): Metrics computed on training data.
        test_metrics (dict): Metrics computed on test (out-of-sample) data.
        train_report: BacktestReport for training period.
        test_report: BacktestReport for test period.
    """
    window_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_periods: int
    test_periods: int
    best_params: dict
    train_metrics: dict
    test_metrics: dict
    train_report: Any = None
    test_report: Any = None


@dataclass
class WalkForwardResult:
    """
    Container for walk-forward optimization results.
    
    This class holds the complete results of a walk-forward optimization run,
    including per-window results and aggregated statistics.
    
    Attributes:
        n_windows (int): Total number of walk-forward windows.
        train_periods (int): Number of periods in each training window.
        test_periods (int): Number of periods in each test window.
        window_results (list[WalkForwardWindow]): Results for each window.
        aggregated_metrics (dict): Aggregated statistics across all windows.
        all_windows_results_df (pd.DataFrame): DataFrame with results from all windows.
    """
    n_windows: int
    train_periods: int
    test_periods: int
    window_results: list[WalkForwardWindow] = field(default_factory=list)
    aggregated_metrics: dict = field(default_factory=dict)
    all_windows_results_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def in_sample_sharpe(self) -> list[float]:
        """Sharpe ratios from training (in-sample) for each window."""
        return [w.train_metrics.get("sharpe", np.nan) for w in self.window_results]

    @property
    def out_of_sample_sharpe(self) -> list[float]:
        """Sharpe ratios from testing (out-of-sample) for each window."""
        return [w.test_metrics.get("sharpe", np.nan) for w in self.window_results]

    @property
    def in_sample_returns(self) -> list[float]:
        """Total returns from training (in-sample) for each window."""
        return [w.train_metrics.get("total_return", np.nan) for w in self.window_results]

    @property
    def out_of_sample_returns(self) -> list[float]:
        """Total returns from testing (out-of-sample) for each window."""
        return [w.test_metrics.get("total_return", np.nan) for w in self.window_results]

    def get_param_stability(self, param_name: str) -> dict:
        """
        Analyze the stability of a parameter across windows.
        
        Args:
            param_name (str): Name of the parameter to analyze.
            
        Returns:
            dict: Dictionary with 'mean', 'std', 'min', 'max', 'cv' (coefficient of variation)
                  for the parameter across windows.
        """
        values = [w.best_params.get(param_name, np.nan) for w in self.window_results]
        values = [v for v in values if v is not None and not np.isnan(v)]
        
        if not values:
            return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "cv": np.nan}
        
        mean_val = np.mean(values)
        return {
            "mean": mean_val,
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "cv": np.std(values) / np.abs(mean_val) if not np.isclose(mean_val, 0) else np.nan,
        }

    def plot(self, figsize: tuple = (14, 8)) -> None:
        """
        Plot walk-forward analysis results.
        
        Creates a multi-panel figure showing:
        1. In-sample vs out-of-sample Sharpe ratios
        2. In-sample vs out-of-sample returns
        3. Parameter stability over time
        
        Args:
            figsize (tuple, optional): Figure size as (width, height) in inches.
                Defaults to (14, 8).
        """
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        n_windows = len(self.window_results)
        if n_windows == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        windows = list(range(n_windows))
        
        # Plot 1: Sharpe Ratios
        ax1 = axes[0, 0]
        in_sharpe = self.in_sample_sharpe
        out_sharpe = self.out_of_sample_sharpe
        ax1.bar([w - 0.2 for w in windows], in_sharpe, 0.4, label='In-Sample', alpha=0.7)
        ax1.bar([w + 0.2 for w in windows], out_sharpe, 0.4, label='Out-of-Sample', alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('In-Sample vs Out-of-Sample Sharpe Ratio')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Returns
        ax2 = axes[0, 1]
        in_returns = [r * 100 for r in self.in_sample_returns]
        out_returns = [r * 100 for r in self.out_of_sample_returns]
        ax2.bar([w - 0.2 for w in windows], in_returns, 0.4, label='In-Sample', alpha=0.7)
        ax2.bar([w + 0.2 for w in windows], out_returns, 0.4, label='Out-of-Sample', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Window')
        ax2.set_ylabel('Total Return (%)')
        ax2.set_title('In-Sample vs Out-of-Sample Returns')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Aggregation metrics
        ax3 = axes[1, 0]
        metrics = ['In-Sample\nSharpe', 'Out-of-Sample\nSharpe']
        is_sharpe_mean = np.nanmean(in_sharpe)
        os_sharpe_mean = np.nanmean(out_sharpe)
        is_sharpe_std = np.nanstd(in_sharpe)
        os_sharpe_std = np.nanstd(out_sharpe)
        
        x_pos = [0, 1]
        ax3.bar(x_pos, [is_sharpe_mean, os_sharpe_mean], 
                yerr=[is_sharpe_std, os_sharpe_std], 
                capsize=5, alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(metrics)
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title('Aggregated Sharpe Ratio (Mean ± Std)')
        ax3.grid(alpha=0.3)
        
        # Plot 4: Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = "Walk-Forward Summary\n"
        summary_text += "=" * 40 + "\n\n"
        summary_text += f"Total Windows: {self.n_windows}\n"
        summary_text += f"Train Periods: {self.train_periods}\n"
        summary_text += f"Test Periods: {self.test_periods}\n\n"
        
        agg = self.aggregated_metrics
        if 'in_sample_sharpe_mean' in agg:
            summary_text += f"In-Sample Sharpe:\n"
            summary_text += f"  Mean: {agg['in_sample_sharpe_mean']:.3f}\n"
            summary_text += f"  Std:  {agg['in_sample_sharpe_std']:.3f}\n\n"
        if 'out_of_sample_sharpe_mean' in agg:
            summary_text += f"Out-of-Sample Sharpe:\n"
            summary_text += f"  Mean: {agg['out_of_sample_sharpe_mean']:.3f}\n"
            summary_text += f"  Std:  {agg['out_of_sample_sharpe_std']:.3f}\n\n"
        if 'oos_to_is_ratio_mean' in agg:
            summary_text += f"Stability Ratio (OOS/IS):\n"
            summary_text += f"  Mean: {agg['oos_to_is_ratio_mean']:.3f}\n"
            summary_text += f"  Std:  {agg['oos_to_is_ratio_std']:.3f}\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        """Generate a formatted string summary of walk-forward results."""
        agg = self.aggregated_metrics
        lines = [
            "Walk-Forward Optimization Results",
            "=" * 40,
            f"Total Windows: {self.n_windows}",
            f"Train Periods per Window: {self.train_periods}",
            f"Test Periods per Window: {self.test_periods}",
            "",
        ]
        
        if 'in_sample_sharpe_mean' in agg:
            lines.extend([
                "In-Sample Performance (Training):",
                f"  Sharpe - Mean: {agg['in_sample_sharpe_mean']:.3f}, "
                f"Std: {agg['in_sample_sharpe_std']:.3f}",
                f"  Return - Mean: {agg['in_sample_return_mean']:.2%}, "
                f"Std: {agg['in_sample_return_std']:.2%}",
                "",
            ])
        
        if 'out_of_sample_sharpe_mean' in agg:
            lines.extend([
                "Out-of-Sample Performance (Testing):",
                f"  Sharpe - Mean: {agg['out_of_sample_sharpe_mean']:.3f}, "
                f"Std: {agg['out_of_sample_sharpe_std']:.3f}",
                f"  Return - Mean: {agg['out_of_sample_return_mean']:.2%}, "
                f"Std: {agg['out_of_sample_return_std']:.2%}",
                "",
            ])
        
        if 'oos_to_is_ratio_mean' in agg:
            lines.append(f"Stability Ratio (OOS/IS Sharpe): {agg['oos_to_is_ratio_mean']:.3f} "
                        f"± {agg['oos_to_is_ratio_std']:.3f}")
        
        return "\n".join(lines)


class OptimizerProtocol(Protocol):
    """
    Protocol defining the interface for optimizer functions.
    
    An optimizer should accept a SimpleBacktester instance and parameter
    configuration, then return an OptimizationResult.
    
    Note:
        The parameter names are flexible - optimizers can use any names for
        the backtester and params arguments (e.g., 'bt', 'backtester', 'bt_inst').
    """
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class WalkForwardAnalyzer:
    """
    Analyzer for walk-forward optimization.
    
    This class performs walk-forward analysis on a trading strategy,
    evaluating parameter optimization over rolling windows. It can work
    with any optimizer function that follows the OptimizerProtocol.
    
    Walk-forward analysis is a rigorous method for evaluating trading
    strategies that simulates real-world deployment conditions. Each
    window represents a complete cycle of:
    1. Training: Optimize parameters on historical data
    2. Testing: Evaluate best parameters on unseen future data
    
    Attributes:
        backtester (SimpleBacktester): The backtester instance to use.
        train_periods (int): Number of periods for each training window.
        test_periods (int): Number of periods for each test window.
        step_periods (int): Number of periods to step forward between windows.
            If None, uses test_periods (non-overlapping windows).
        min_train_periods (int): Minimum required training periods.
        min_test_periods (int): Minimum required test periods.
        selection_criterion (str): Metric to use for selecting best parameters.
    
    Example:
        >>> from quantex.backtester.walk_forward import WalkForwardAnalyzer
        >>> from quantex import SimpleBacktester
        >>> 
        >>> # Create analyzer with grid search optimizer
        >>> analyzer = WalkForwardAnalyzer(
        ...     backtester=bt,
        ...     train_periods=252,  # 1 year training
        ...     test_periods=63,   # 3 months testing
        ...     step_periods=63    # Move forward 3 months each window
        ... )
        >>> 
        >>> # Run with grid search
        >>> result = analyzer.analyze(
        ...     optimizer=lambda bt, params: bt.optimize(params),
        ...     params={'fast': [5, 10, 20], 'slow': [20, 50, 100]}
        ... )
        >>> 
        >>> # View aggregated results
        >>> print(result)
        >>> print(f"Average out-of-sample Sharpe: {result.aggregated_metrics['out_of_sample_sharpe_mean']:.2f}")
    """
    backtester: Any
    train_periods: int
    test_periods: int
    step_periods: int = 0  # Will be set to test_periods in __post_init__ if 0
    min_train_periods: int = 30
    min_test_periods: int = 10
    selection_criterion: str = "sharpe"

    def __post_init__(self):
        """Validate and set default values."""
        # Handle default step_periods
        if self.step_periods <= 0:
            self.step_periods = self.test_periods
        
        if self.train_periods < self.min_train_periods:
            raise ValueError(
                f"train_periods must be at least {self.min_train_periods}, "
                f"got {self.train_periods}"
            )
        
        if self.test_periods < self.min_test_periods:
            raise ValueError(
                f"test_periods must be at least {self.min_test_periods}, "
                f"got {self.test_periods}"
            )
        
        if self.step_periods <= 0:
            raise ValueError("step_periods must be positive")
        
        # Get data length from the strategy's data source
        positions = self.backtester.strategy.positions
        if not positions:
            raise ValueError(
                "Strategy must have at least one data source registered. "
                "Use strategy.add_data(datasource, name) before walk-forward analysis."
            )
        source = positions[next(iter(positions))].source
        self.data_length = len(source.data)

    def _create_window_splits(self) -> list[tuple[int, int, int, int]]:
        """
        Create the train/test splits for all walk-forward windows.
        
        Returns:
            List of tuples: (train_start, train_end, test_start, test_end)
        """
        splits = []
        train_start = 0
        step = self.step_periods  # Already validated to be non-None in __post_init__
        
        while True:
            train_end = train_start + self.train_periods
            test_start = train_end
            test_end = test_start + self.test_periods
            
            # Check if we have enough data for test period
            if test_end > self.data_length:
                break
            
            splits.append((train_start, train_end, test_start, test_end))
            
            # Move forward
            train_start += step
        
        return splits

    def _slice_strategy_for_window(
        self,
        params: dict,
        train_start: int,
        train_end: int,
    ):
        """
        Create a strategy copy with data sliced for training.
        
        Args:
            params (dict): Parameter values to apply.
            train_start (int): Start index for training data.
            train_end (int): End index for training data.
            
        Returns:
            Tuple of (sliced_strategy, data_source_for_metrics)
        """
        import copy
        from ..datasource import DataSource
        
        strat_copy = copy.deepcopy(self.backtester.strategy)
        
        # Apply parameters
        for k, v in params.items():
            if isinstance(v, float) and v == int(v):
                v = int(v)
            setattr(strat_copy, k, v)
        
        # Slice each data source
        for key, broker in strat_copy.positions.items():
            source = broker.source
            sliced_df = source.data.iloc[train_start:train_end].copy()
            new_source = DataSource(sliced_df)
            broker.source = new_source
            strat_copy.data[key] = new_source
        
        return strat_copy

    def _slice_strategy_for_test(
        self,
        params: dict,
        test_start: int,
        test_end: int,
    ):
        """
        Create a strategy copy with data sliced for testing.
        
        Args:
            params (dict): Parameter values to apply.
            test_start (int): Start index for test data.
            test_end (int): End index for test data.
            
        Returns:
            Tuple of (sliced_strategy, data_source_for_metrics)
        """
        import copy
        from ..datasource import DataSource
        
        strat_copy = copy.deepcopy(self.backtester.strategy)
        
        # Apply parameters (same as training)
        for k, v in params.items():
            if isinstance(v, float) and v == int(v):
                v = int(v)
            setattr(strat_copy, k, v)
        
        # Slice each data source
        for key, broker in strat_copy.positions.items():
            source = broker.source
            sliced_df = source.data.iloc[test_start:test_end].copy()
            new_source = DataSource(sliced_df)
            broker.source = new_source
            strat_copy.data[key] = new_source
        
        return strat_copy

    def analyze(
        self,
        optimizer: OptimizerProtocol,
        params: dict[str, Any],
        constraint: Callable[[dict], bool] | None = None,
        objective: str = "sharpe",
        risk_tolerance: dict[str, float] | None = None,
        progress_bar: bool = True,
        **optimizer_kwargs: Any,
    ) -> WalkForwardResult:
        """
        Perform walk-forward analysis using the specified optimizer.
        
        This method runs walk-forward optimization over multiple rolling windows.
        For each window:
        1. Train: Optimize parameters using the specified optimizer on training data
        2. Test: Evaluate the best parameters on out-of-sample test data
        
        Args:
            optimizer (OptimizerProtocol): Optimizer function to use. Should accept
                a SimpleBacktester and params dict, and return an OptimizationResult.
                Can be any of: bt.optimize, bt.optimize_parallel, bt.optimize_optuna,
                or a custom optimizer function.
            params (dict[str, Any]): Parameter space for optimization.
            constraint (Callable[[dict], bool] | None, optional): Optional constraint
                function for parameter validation. Defaults to None.
            objective (str, optional): Metric to optimize. Defaults to "sharpe".
            risk_tolerance (dict[str, float] | None, optional): Risk tolerance
                constraints. Defaults to None.
            progress_bar (bool, optional): Whether to show progress bar.
                Defaults to True.
            **optimizer_kwargs: Additional keyword arguments passed to optimizer.
        
        Returns:
            WalkForwardResult: Object containing:
                - window_results: List of WalkForwardWindow objects for each window
                - aggregated_metrics: Aggregated statistics across all windows
                - all_windows_results_df: DataFrame with all results
        
        Example:
            Using grid search:
            >>> result = analyzer.analyze(
            ...     optimizer=lambda bt, params: bt.optimize(params),
            ...     params={'fast': [5, 10, 20], 'slow': [20, 50, 100]},
            ...     objective='sharpe'
            ... )
            
            Using Optuna:
            >>> result = analyzer.analyze(
            ...     optimizer=lambda bt, params: bt.optimize_optuna(
            ...         {k: v if isinstance(v[0], (int, float)) else v 
            ...          for k, v in params.items()}
            ...     ),
            ...     params={'fast': (5, 50), 'slow': (20, 100)},
            ...     n_trials=50
            ... )
        """
        from .backtester import SimpleBacktester
        from .metrics import _compute_backtest_metrics
        
        # Get window splits
        splits = self._create_window_splits()
        n_windows = len(splits)
        
        if n_windows == 0:
            raise ValueError(
                f"Data length ({self.data_length}) is too short for the configured "
                f"train_periods ({self.train_periods}) and test_periods ({self.test_periods}). "
                f"Need at least {self.train_periods + self.test_periods} periods."
            )
        
        window_results: list[WalkForwardWindow] = []
        all_rows = []
        
        iterator = tqdm(splits, desc="Walk-Forward", disable=not progress_bar)
        
        for window_idx, (train_start, train_end, test_start, test_end) in enumerate(iterator):
            # Create training backtester with sliced data
            train_strat = self._slice_strategy_for_window(
                {}, train_start, train_end
            )
            train_bt = SimpleBacktester(
                train_strat,
                cash=self.backtester.cash,
                commission=self.backtester.commission,
                commission_type=self.backtester.commission_type,
                lot_size=self.backtester.lot_size,
                leverage=self.backtester.leverage,
            )
            
            # Run optimization on training data
            # Create a modified params dict that maintains parameter types
            opt_params = {}
            for k, v in params.items():
                # Convert to list if it's a range
                if isinstance(v, range):
                    opt_params[k] = list(v)
                else:
                    opt_params[k] = v
            
            # Run optimizer
            train_result = optimizer(
                train_bt,
                opt_params,
                constraint=constraint,
                objective=objective,
                risk_tolerance=risk_tolerance,
                **optimizer_kwargs,
            )
            
            # Get best parameters
            best_params = train_result.best_params if train_result.best_params else {}
            
            # Compute training metrics
            if train_result.train_report is not None:
                train_report = train_result.train_report
                train_metrics = _compute_backtest_metrics(train_report)
            elif train_result.train_metrics:
                train_metrics = train_result.train_metrics
                train_report = None
            else:
                train_metrics = {}
                train_report = None
            
            # Create test backtester with best parameters
            test_strat = self._slice_strategy_for_test(
                best_params, test_start, test_end
            )
            test_bt = SimpleBacktester(
                test_strat,
                cash=self.backtester.cash,
                commission=self.backtester.commission,
                commission_type=self.backtester.commission_type,
                lot_size=self.backtester.lot_size,
                leverage=self.backtester.leverage,
            )
            
            # Run backtest on test data
            test_report = test_bt.run(progress_bar=False)
            test_metrics = _compute_backtest_metrics(test_report)
            
            # Store window result
            window = WalkForwardWindow(
                window_index=window_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_periods=train_end - train_start,
                test_periods=test_end - test_start,
                best_params=best_params,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                train_report=train_report,
                test_report=test_report,
            )
            window_results.append(window)
            
            # Build row for results DataFrame
            row = {
                'window': window_idx,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            }
            row.update({f'train_{k}': v for k, v in train_metrics.items()})
            row.update({f'test_{k}': v for k, v in test_metrics.items()})
            row.update({f'param_{k}': v for k, v in best_params.items()})
            all_rows.append(row)
        
        # Create DataFrame
        results_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
        
        # Compute aggregated metrics
        aggregated = self._compute_aggregated_metrics(window_results)
        
        return WalkForwardResult(
            n_windows=n_windows,
            train_periods=self.train_periods,
            test_periods=self.test_periods,
            window_results=window_results,
            aggregated_metrics=aggregated,
            all_windows_results_df=results_df,
        )

    def _compute_aggregated_metrics(
        self,
        window_results: list[WalkForwardWindow],
    ) -> dict:
        """
        Compute aggregated statistics across all windows.
        
        Args:
            window_results: List of window results.
            
        Returns:
            Dictionary of aggregated metrics.
        """
        metrics = {}
        
        if not window_results:
            return metrics
        
        # In-sample metrics
        is_sharpe = [w.train_metrics.get('sharpe', np.nan) for w in window_results]
        is_return = [w.train_metrics.get('total_return', np.nan) for w in window_results]
        
        # Out-of-sample metrics
        oos_sharpe = [w.test_metrics.get('sharpe', np.nan) for w in window_results]
        oos_return = [w.test_metrics.get('total_return', np.nan) for w in window_results]
        
        # Compute means and stds
        metrics['in_sample_sharpe_mean'] = np.nanmean(is_sharpe)
        metrics['in_sample_sharpe_std'] = np.nanstd(is_sharpe)
        metrics['in_sample_sharpe_min'] = np.nanmin(is_sharpe)
        metrics['in_sample_sharpe_max'] = np.nanmax(is_sharpe)
        
        metrics['in_sample_return_mean'] = np.nanmean(is_return)
        metrics['in_sample_return_std'] = np.nanstd(is_return)
        
        metrics['out_of_sample_sharpe_mean'] = np.nanmean(oos_sharpe)
        metrics['out_of_sample_sharpe_std'] = np.nanstd(oos_sharpe)
        metrics['out_of_sample_sharpe_min'] = np.nanmin(oos_sharpe)
        metrics['out_of_sample_sharpe_max'] = np.nanmax(oos_sharpe)
        
        metrics['out_of_sample_return_mean'] = np.nanmean(oos_return)
        metrics['out_of_sample_return_std'] = np.nanstd(oos_return)
        
        # Stability ratio (out-of-sample / in-sample)
        # Higher ratio indicates more stable parameters
        is_sharpe_arr = np.array(is_sharpe)
        oos_sharpe_arr = np.array(oos_sharpe)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = oos_sharpe_arr / is_sharpe_arr
            # Only keep finite values where denominator is not zero
            ratios = np.where(
                np.isfinite(ratios) & ~np.isclose(is_sharpe_arr, 0),
                ratios,
                np.nan
            )
        
        metrics['oos_to_is_ratio_mean'] = np.nanmean(ratios)
        metrics['oos_to_is_ratio_std'] = np.nanstd(ratios)
        
        # Win rate (percentage of windows with positive out-of-sample return)
        metrics['oos_win_rate'] = np.mean([r > 0 for r in oos_return])
        
        return metrics


def walk_forward_analyze(
    backtester: Any,
    optimizer: OptimizerProtocol,
    params: dict[str, Any],
    train_periods: int,
    test_periods: int,
    step_periods: int | None = None,
    constraint: Callable[[dict], bool] | None = None,
    objective: str = "sharpe",
    risk_tolerance: dict[str, float] | None = None,
    min_train_periods: int = 30,
    min_test_periods: int = 10,
    progress_bar: bool = True,
    **optimizer_kwargs: Any,
) -> WalkForwardResult:
    """
    Convenience function for walk-forward analysis.
    
    This is a convenience wrapper around WalkForwardAnalyzer.analyze()
    that creates the analyzer and runs the analysis in one call.
    
    Args:
        backtester (SimpleBacktester): The backtester instance to use.
        optimizer (OptimizerProtocol): Optimizer function to use.
        params (dict[str, Any]): Parameter space for optimization.
        train_periods (int): Number of periods for each training window.
        test_periods (int): Number of periods for each test window.
        step_periods (int | None, optional): Periods to step between windows.
            If None, uses test_periods. Defaults to None.
        constraint (Callable[[dict], bool] | None, optional): Constraint function.
        objective (str, optional): Metric to optimize. Defaults to "sharpe".
        risk_tolerance (dict[str, float] | None, optional): Risk tolerance.
        min_train_periods (int, optional): Minimum training periods.
            Defaults to 30.
        min_test_periods (int, optional): Minimum test periods.
            Defaults to 10.
        progress_bar (bool, optional): Show progress bar. Defaults to True.
        **optimizer_kwargs: Additional arguments passed to optimizer.
    
    Returns:
        WalkForwardResult: Walk-forward analysis results.
    
    Example:
        >>> from quantex.backtester.walk_forward import walk_forward_analyze
        >>> result = walk_forward_analyze(
        ...     backtester=bt,
        ...     optimizer=lambda bt, params: bt.optimize(params),
        ...     params={'fast': [5, 10, 20], 'slow': [20, 50, 100]},
        ...     train_periods=252,
        ...     test_periods=63,
        ...     objective='sharpe'
        ... )
        >>> print(f"Average OOS Sharpe: {result.aggregated_metrics['out_of_sample_sharpe_mean']:.2f}")
    """
    # Handle step_periods - use test_periods if None
    actual_step = test_periods if step_periods is None else step_periods
    
    analyzer = WalkForwardAnalyzer(
        backtester=backtester,
        train_periods=train_periods,
        test_periods=test_periods,
        step_periods=actual_step,
        min_train_periods=min_train_periods,
        min_test_periods=min_test_periods,
        selection_criterion=objective,
    )
    
    return analyzer.analyze(
        optimizer=optimizer,
        params=params,
        constraint=constraint,
        objective=objective,
        risk_tolerance=risk_tolerance,
        progress_bar=progress_bar,
        **optimizer_kwargs,
    )
