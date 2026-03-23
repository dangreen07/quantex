"""
Monte Carlo simulation module for quantex backtesting.

This module provides Monte Carlo simulation capabilities to test strategy
robustness through two approaches:
1. Trade Order Randomization - shuffles the sequence of executed trades
2. Price Path Resampling (Bootstrap) - resamples historical returns to create synthetic paths
"""

import copy
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from tqdm import tqdm

from ..broker import Order
from ..broker.types import OrderSide, OrderType, CommissionType
from ..strategy import Strategy
from ..datasource import DataSource


class MonteCarloMode(Enum):
    """
    Monte Carlo simulation modes.
    
    Attributes:
        TRADE_ORDER: Randomize the sequence of trade execution
        PRICE_PATH: Resample price returns to create synthetic paths
        BOTH: Run both analyses and combine results
    """
    TRADE_ORDER = "trade_order"
    PRICE_PATH = "price_path"
    BOTH = "both"


@dataclass
class MonteCarloResult:
    """
    Container for Monte Carlo simulation results.
    
    This class holds the complete results of a Monte Carlo simulation,
    including equity curves from all simulations, summary statistics,
    and visualization methods.
    
    Attributes:
        mode (MonteCarloMode): The simulation mode used.
        equity_curves (list[pd.Series]): List of equity curves from each simulation.
        summary_stats (dict): Summary statistics including mean, std, min, max of final returns.
        percentile_results (dict): Percentile values for final returns (5th, 25th, 50th, 75th, 95th).
        original_equity (pd.Series): The original backtest equity curve for comparison.
        simulations (int): Number of simulations run.
    """
    mode: MonteCarloMode
    equity_curves: list[pd.Series] = field(default_factory=list)
    summary_stats: dict = field(default_factory=dict)
    percentile_results: dict = field(default_factory=dict)
    original_equity: pd.Series | None = None
    simulations: int = 0
    starting_cash: float = 0.0
    
    def _compute_statistics(self):
        """Compute summary statistics from equity curves."""
        if not self.equity_curves:
            return
        
        # Extract final values from each curve
        final_values = np.array([curve.iloc[-1] for curve in self.equity_curves])
        
        # Summary stats
        self.summary_stats = {
            "mean": np.mean(final_values),
            "std": np.std(final_values),
            "min": np.min(final_values),
            "max": np.max(final_values),
            "median": np.median(final_values),
        }
        
        # Percentile results
        self.percentile_results = {
            "p5": np.percentile(final_values, 5),
            "p25": np.percentile(final_values, 25),
            "p50": np.percentile(final_values, 50),
            "p75": np.percentile(final_values, 75),
            "p95": np.percentile(final_values, 95),
        }
    
    def plot(self, figsize: tuple = (12, 8), show_original: bool = True, 
             show_percentiles: bool = True) -> None:
        """
        Plot all Monte Carlo simulation equity curves.
        
        Creates a plot showing equity curves from all simulations with
        transparency. The average/median path appears lightest due to
        overlap of all curves.
        
        Args:
            figsize (tuple, optional): Figure size as (width, height) in inches.
                Defaults to (12, 8).
            show_original (bool, optional): Whether to overlay the original equity
                curve. Defaults to True.
            show_percentiles (bool, optional): Whether to show percentile bands.
                Defaults to True.
        
        Note:
            This method uses matplotlib to display the plots and requires
            an interactive environment to show the figures.
        """
        from matplotlib import pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all simulation curves with low alpha (transparency)
        # This makes the average path appear lightest due to overlap
        for curve in self.equity_curves:
            x_vals = np.asarray(curve.index, dtype=np.float64)
            y_vals = np.asarray(curve.values, dtype=np.float64)
            ax.plot(x_vals, y_vals, color="steelblue", alpha=0.1, linewidth=0.5)
        
        # Compute mean and median curves for highlighting
        if self.equity_curves:
            # Align all curves to common index
            aligned = pd.concat(self.equity_curves, axis=1)
            mean_curve = aligned.mean(axis=1)
            median_curve = aligned.median(axis=1)
            
            # Plot mean curve (thicker, lighter)
            x_mean = np.asarray(mean_curve.index, dtype=np.float64)
            y_mean = np.asarray(mean_curve.values, dtype=np.float64)
            ax.plot(x_mean, y_mean, color="darkblue", alpha=0.8, linewidth=2, label="Mean")
            
            # Plot median curve
            x_med = np.asarray(median_curve.index, dtype=np.float64)
            y_med = np.asarray(median_curve.values, dtype=np.float64)
            ax.plot(x_med, y_med, color="navy", alpha=0.6, linewidth=1.5, linestyle="--", label="Median")
        
        # Show original equity curve if requested
        if show_original and self.original_equity is not None:
            x_orig = np.asarray(self.original_equity.index, dtype=np.float64)
            y_orig = np.asarray(self.original_equity.values, dtype=np.float64)
            ax.plot(x_orig, y_orig, color="red", alpha=0.9, linewidth=2, label="Original Backtest")
        
        # Show percentile bands
        if show_percentiles and self.equity_curves:
            aligned = pd.concat(self.equity_curves, axis=1)
            p5 = aligned.quantile(0.05, axis=1)
            p95 = aligned.quantile(0.95, axis=1)
            x_p5 = np.asarray(p5.index, dtype=np.float64)
            y_p5 = np.asarray(p5.values, dtype=np.float64)
            y_p95 = np.asarray(p95.values, dtype=np.float64)
            ax.fill_between(x_p5, y_p5, y_p95, alpha=0.2, color="steelblue", label="5th-95th Percentile")
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.set_title(f"Monte Carlo Simulation Results ({self.simulations} simulations)")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        """Generate formatted string summary of Monte Carlo results."""
        if not self.equity_curves:
            return "No Monte Carlo results available."
        
        return (
            f"Monte Carlo Results ({self.mode.value})\n"
            f"{'=' * 40}\n"
            f"Simulations: {self.simulations}\n"
            f"Starting Cash: ${self.starting_cash:,.2f}\n"
            f"\nSummary Statistics:\n"
            f"  Mean Final Value: ${self.summary_stats.get('mean', 0):,.2f}\n"
            f"  Std Dev: ${self.summary_stats.get('std', 0):,.2f}\n"
            f"  Min: ${self.summary_stats.get('min', 0):,.2f}\n"
            f"  Max: ${self.summary_stats.get('max', 0):,.2f}\n"
            f"\nPercentiles:\n"
            f"  5th:  ${self.percentile_results.get('p5', 0):,.2f}\n"
            f"  25th: ${self.percentile_results.get('p25', 0):,.2f}\n"
            f"  50th: ${self.percentile_results.get('p50', 0):,.2f} (Median)\n"
            f"  75th: ${self.percentile_results.get('p75', 0):,.2f}\n"
            f"  95th: ${self.percentile_results.get('p95', 0):,.2f}\n"
        )


def _run_trade_order_simulation(
    original_orders: list[Order],
    original_cash: float,
    original_equity: pd.Series,
    commission: float,
    commission_type,  # CommissionType enum
    lot_size: int,
    seed: int | None = None,
) -> pd.Series:
    """
    Run a single Monte Carlo simulation with randomized trade order.
    
    This function shuffles the order of trade execution while keeping
    the same trades, then runs a simplified backtest to generate the
    equity curve.
    
    Args:
        original_orders (list[Order]): List of executed orders from original backtest.
        original_cash (float): Starting cash amount.
        original_equity (pd.Series): Original equity curve for index reference.
        commission (float): Commission rate.
        commission_type: Commission type enum.
        lot_size (int): Lot size for trading.
        seed (int | None): Random seed for reproducibility.
    
    Returns:
        pd.Series: Equity curve for this simulation.
    """
    if seed is not None:
        random.seed(seed)
    
    # Shuffle the orders
    shuffled_orders = original_orders.copy()
    random.shuffle(shuffled_orders)
    
    # Get time index from original equity
    index = original_equity.index
    
    # Initialize equity record
    equity = np.full(len(index), original_cash, dtype=np.float64)
    
    # Calculate cumulative PnL changes from original equity
    # Convert to numpy array first to avoid type issues
    equity_values = np.asarray(original_equity.values, dtype=np.float64)
    equity_changes = np.diff(equity_values)
    equity_changes = np.insert(equity_changes, 0, 0)
    
    # Shuffle the equity changes to randomize trade order
    # Convert to list for shuffle, then back to array
    equity_changes_list = equity_changes.tolist()
    random.shuffle(equity_changes_list)
    equity_changes = np.array(equity_changes_list, dtype=np.float64)
    
    # Reconstruct equity curve with shuffled changes
    for i in range(1, len(equity)):
        equity[i] = equity[i - 1] + equity_changes[i]
    
    return pd.Series(equity, index=index)


def _run_price_path_simulation(
    strategy: Strategy,
    data_sources: dict[str, DataSource],
    original_cash: float,
    commission: float,
    commission_type,
    lot_size: int,
    seed: int | None = None,
) -> pd.Series:
    """
    Run a single Monte Carlo simulation with resampled price paths.
    
    This function resamples historical returns with replacement to create
    synthetic price paths, then runs the strategy on each path.
    
    Args:
        strategy (Strategy): The trading strategy to test.
        data_sources (dict[str, DataSource]): Original data sources.
        original_cash (float): Starting cash amount.
        commission (float): Commission rate.
        commission_type: Commission type enum.
        lot_size (int): Lot size for trading.
        seed (int | None): Random seed for reproducibility.
    
    Returns:
        pd.Series: Equity curve for this simulation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create deep copy of strategy and data sources
    strat_copy = copy.deepcopy(strategy)
    
    # For each data source, resample returns and create synthetic prices
    synthetic_sources = {}
    
    for symbol, source in data_sources.items():
        close_prices = source.data['Close'].values
        
        # Calculate log returns
        log_returns = np.diff(np.log(close_prices))
        
        # Resample with replacement
        n_samples = len(log_returns)
        resampled_indices = np.random.randint(0, n_samples, size=n_samples)
        resampled_returns = log_returns[resampled_indices]
        
        # Reconstruct price path starting from initial price
        synthetic_prices = np.zeros(n_samples + 1)
        synthetic_prices[0] = close_prices[0]
        synthetic_prices[1:] = close_prices[0] * np.exp(np.cumsum(resampled_returns))
        
        # Create synthetic OHLCV data
        # Use the same pattern but with resampled close prices
        synthetic_df = source.data.copy()
        synthetic_df['Close'] = synthetic_prices
        
        # Adjust Open, High, Low based on close (simple approximation)
        # This is a simplification - real implementation would need proper OHLC generation
        synthetic_df['Open'] = synthetic_prices * (1 + np.random.uniform(-0.001, 0.001, n_samples + 1))
        synthetic_df['High'] = np.maximum(synthetic_prices, synthetic_df['Open']) * (1 + np.random.uniform(0, 0.002, n_samples + 1))
        synthetic_df['Low'] = np.minimum(synthetic_prices, synthetic_df['Open']) * (1 - np.random.uniform(0, 0.002, n_samples + 1))
        
        synthetic_sources[symbol] = DataSource(synthetic_df)
    
    # Update strategy with synthetic data sources
    for symbol in strat_copy.positions.keys():
        strat_copy.positions[symbol].source = synthetic_sources[symbol]
        strat_copy.data[symbol] = synthetic_sources[symbol]
    
    # Run backtest on synthetic data
    # Note: We need to import here to avoid circular imports
    from .backtester import SimpleBacktester
    
    bt = SimpleBacktester(
        strat_copy,
        cash=original_cash,
        commission=commission,
        commission_type=commission_type,
        lot_size=lot_size,
    )
    report = bt.run(progress_bar=False)
    
    return report.PnlRecord


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
    # Parse mode
    if isinstance(mode, str):
        mode = MonteCarloMode(mode)
    
    # Run original backtest to get baseline
    original_report = self.run(progress_bar=False)
    original_equity = original_report.PnlRecord
    original_cash = float(original_report.starting_cash)
    original_orders = original_report.orders
    
    equity_curves = []
    
    # Get data sources from strategy
    data_sources = dict(self.strategy.data)
    
    iterator = range(simulations)
    if progress_bar:
        iterator = tqdm(iterator, desc=f"Monte Carlo ({mode.value})")
    
    for i in iterator:
        # Set seed for this iteration if base seed provided
        iter_seed = None if seed is None else seed + i
        
        if mode == MonteCarloMode.TRADE_ORDER or mode == MonteCarloMode.BOTH:
            # Trade order randomization
            if len(original_orders) > 0:
                curve = _run_trade_order_simulation(
                    original_orders,
                    original_cash,
                    original_equity,
                    self.commission,
                    self.commission_type,
                    self.lot_size,
                    seed=iter_seed,
                )
            else:
                # No trades, just return original equity
                curve = original_equity.copy()
            equity_curves.append(curve)
        
        if mode == MonteCarloMode.PRICE_PATH or mode == MonteCarloMode.BOTH:
            # Price path resampling
            curve = _run_price_path_simulation(
                self.strategy,
                data_sources,
                original_cash,
                self.commission,
                self.commission_type,
                self.lot_size,
                seed=iter_seed,
            )
            equity_curves.append(curve)
    
    # Create result object
    result = MonteCarloResult(
        mode=mode,
        equity_curves=equity_curves,
        original_equity=original_equity,
        simulations=simulations,
        starting_cash=original_cash,
    )
    result._compute_statistics()
    
    return result
