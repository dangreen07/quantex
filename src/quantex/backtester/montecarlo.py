"""
Monte Carlo simulation module for quantex backtesting.

This module provides Monte Carlo simulation capabilities to test strategy
robustness through two approaches:
1. Trade Order Randomization - shuffles the sequence of executed trades
2. Price Path Resampling (Bootstrap) - resamples historical returns to create synthetic paths
"""

import copy
import math
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
    drawdown_stats: dict = field(default_factory=dict)
    
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

        drawdowns = []
        for curve in self.equity_curves:
            running_max = curve.cummax()
            dd = (curve / running_max) - 1.0
            drawdowns.append(float(dd.min()))

        drawdown_values = np.asarray(drawdowns, dtype=np.float64)
        self.drawdown_stats = {
            "mean": float(np.mean(drawdown_values)),
            "std": float(np.std(drawdown_values)),
            "min": float(np.min(drawdown_values)),
            "max": float(np.max(drawdown_values)),
            "median": float(np.median(drawdown_values)),
            "p5": float(np.percentile(drawdown_values, 5)),
            "p25": float(np.percentile(drawdown_values, 25)),
            "p50": float(np.percentile(drawdown_values, 50)),
            "p75": float(np.percentile(drawdown_values, 75)),
            "p95": float(np.percentile(drawdown_values, 95)),
        }

    def probabilities(
        self,
        target_return: float,
        drawdown_threshold: float,
        horizon: int | None = None,
        as_percent: bool = True,
    ) -> dict:
        """
        Calculate the probability of reaching a target return and exceeding
        a drawdown threshold within a given time horizon.

        Args:
            target_return (float): Target return threshold. If `as_percent` is
                True, this is treated as a decimal return (e.g. 0.05 for 5%).
            drawdown_threshold (float): Drawdown threshold. If `as_percent` is
                True, this is treated as a decimal drawdown (e.g. 0.05 for 5%).
            horizon (int | None, optional): Number of steps to evaluate. Defaults
                to the full length of the simulated curves.
            as_percent (bool, optional): Whether thresholds are provided as
                decimal percentages. Defaults to True.

        Returns:
            dict: Probability summary containing return and drawdown metrics.
        """
        if not self.equity_curves:
            return {
                "return_probability": 0.0,
                "drawdown_probability": 0.0,
                "horizon": horizon,
                "target_return": target_return,
                "drawdown_threshold": drawdown_threshold,
            }

        horizon = horizon or len(self.equity_curves[0])
        horizon = max(1, min(horizon, len(self.equity_curves[0])))

        if as_percent:
            target_return = float(target_return)
            drawdown_threshold = float(drawdown_threshold)

        return_hits = 0
        drawdown_hits = 0
        for curve in self.equity_curves:
            sampled = curve.iloc[:horizon]
            start_value = float(sampled.iloc[0])
            end_value = float(sampled.iloc[-1])
            achieved_return = (end_value / start_value) - 1.0 if start_value != 0 else 0.0
            max_drawdown = float(((sampled / sampled.cummax()) - 1.0).min())

            if achieved_return >= target_return:
                return_hits += 1
            if abs(max_drawdown) >= drawdown_threshold:
                drawdown_hits += 1

        total = len(self.equity_curves)
        return {
            "return_probability": return_hits / total,
            "drawdown_probability": drawdown_hits / total,
            "horizon": horizon,
            "target_return": target_return,
            "drawdown_threshold": drawdown_threshold,
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
        
        # Plot using a numeric simulation step axis to avoid date conversion
        # artifacts when equity curves share the same time index.
        if not self.equity_curves:
            ax.set_xlabel("Step")
            ax.set_ylabel("Portfolio Value")
            ax.set_title(f"Monte Carlo Simulation Results ({self.simulations} simulations)")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
            return

        step_index = np.arange(len(self.equity_curves[0]), dtype=np.float64)
        
        # Plot all simulation curves with low alpha (transparency)
        # This makes the average path appear lightest due to overlap
        for curve in self.equity_curves:
            x_vals = np.arange(len(curve), dtype=np.float64)
            y_vals = np.asarray(curve.values, dtype=np.float64)
            ax.plot(x_vals, y_vals, color="steelblue", alpha=0.1, linewidth=0.5)
        
        # Compute mean and median curves for highlighting
        if self.equity_curves:
            # Align all curves to common index
            aligned = pd.concat(self.equity_curves, axis=1)
            mean_curve = aligned.mean(axis=1)
            median_curve = aligned.median(axis=1)
            
            # Plot mean curve (thicker, lighter)
            x_mean = np.arange(len(mean_curve), dtype=np.float64)
            y_mean = np.asarray(mean_curve.values, dtype=np.float64)
            ax.plot(x_mean, y_mean, color="darkblue", alpha=0.8, linewidth=2, label="Mean")
            
            # Plot median curve
            x_med = np.arange(len(median_curve), dtype=np.float64)
            y_med = np.asarray(median_curve.values, dtype=np.float64)
            ax.plot(x_med, y_med, color="navy", alpha=0.6, linewidth=1.5, linestyle="--", label="Median")
        
        # Show original equity curve if requested
        if show_original and self.original_equity is not None:
            x_orig = np.arange(len(self.original_equity), dtype=np.float64)
            y_orig = np.asarray(self.original_equity.values, dtype=np.float64)
            ax.plot(x_orig, y_orig, color="red", alpha=0.9, linewidth=2, label="Original Backtest")
        
        # Show percentile bands
        if show_percentiles and self.equity_curves:
            aligned = pd.concat(self.equity_curves, axis=1)
            p5 = aligned.quantile(0.05, axis=1)
            p95 = aligned.quantile(0.95, axis=1)
            x_p5 = np.arange(len(p5), dtype=np.float64)
            y_p5 = np.asarray(p5.values, dtype=np.float64)
            y_p95 = np.asarray(p95.values, dtype=np.float64)
            ax.fill_between(x_p5, y_p5, y_p95, alpha=0.2, color="steelblue", label="5th-95th Percentile")
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Portfolio Value")
        ax.set_title(f"Monte Carlo Simulation Results ({self.simulations} simulations)")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        # Match the more compact spaghetti-plot look by tightening x-limits.
        ax.set_xlim(step_index[0], step_index[-1])
        
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
            f"\nDrawdown Statistics (% of peak):\n"
            f"  Mean Max DD: {self.drawdown_stats.get('mean', 0):.2%}\n"
            f"  5th: {self.drawdown_stats.get('p5', 0):.2%}\n"
            f"  25th: {self.drawdown_stats.get('p25', 0):.2%}\n"
            f"  50th: {self.drawdown_stats.get('p50', 0):.2%}\n"
            f"  75th: {self.drawdown_stats.get('p75', 0):.2%}\n"
            f"  95th: {self.drawdown_stats.get('p95', 0):.2%}\n"
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

    # Trade-order Monte Carlo must preserve the trade outcomes while changing
    # only the order in which those outcomes are realized. We therefore shuffle
    # the per-step percentage returns, not the absolute equity values.
    
    # Get time index from original equity
    index = original_equity.index
    
    # Initialize equity record
    equity = np.full(len(index), original_cash, dtype=np.float64)
    
    # Use step-wise percentage returns rather than absolute value changes.
    # This keeps the path dependent on the sequence of returns rather than
    # collapsing to the same terminal value every time.
    equity_values = np.asarray(original_equity.values, dtype=np.float64)
    equity_returns = np.zeros_like(equity_values)
    if len(equity_values) > 1:
        prev = np.where(np.arange(len(equity_values)) == 0, original_cash, equity_values[:-1])
        equity_returns[1:] = np.where(prev > 0, (equity_values[1:] / prev) - 1.0, 0.0)

    # Keep the starting cash anchored at index 0 and randomize the remaining
    # returns so the path always begins from the actual initial capital.
    shuffled_returns = equity_returns[1:].tolist()
    random.shuffle(shuffled_returns)
    equity_returns = np.concatenate(([0.0], np.asarray(shuffled_returns, dtype=np.float64)))
    
    # Reconstruct equity curve with shuffled returns
    for i in range(1, len(equity)):
        equity[i] = equity[i - 1] * (1.0 + equity_returns[i])

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
        close_prices = np.asarray(source.data["Close"].values, dtype=np.float64)
        if len(close_prices) < 2:
            synthetic_sources[symbol] = DataSource(source.data.copy())
            continue

        # Use a block-bootstrap on log returns to preserve local serial dependence
        # and then re-price the path using a geometric Brownian motion style
        # reconstruction with the sampled return distribution.
        log_returns = np.diff(np.log(close_prices))
        n_samples = len(log_returns)
        block_size = max(2, min(10, int(np.sqrt(n_samples))))
        synthetic_log_returns = []

        while len(synthetic_log_returns) < n_samples:
            start = int(np.random.randint(0, n_samples))
            block = log_returns[start : start + block_size]
            if len(block) < block_size:
                wrap = block_size - len(block)
                block = np.concatenate((block, log_returns[:wrap]))
            synthetic_log_returns.extend(block.tolist())

        synthetic_log_returns = np.asarray(synthetic_log_returns[:n_samples], dtype=np.float64)

        # Keep the simulated path realistic by matching the original return
        # center and volatility rather than letting the bootstrap drift too far.
        original_mean = float(np.mean(log_returns))
        original_std = float(np.std(log_returns))
        synthetic_mean = float(np.mean(synthetic_log_returns))
        synthetic_std = float(np.std(synthetic_log_returns))
        if synthetic_std > 0 and original_std > 0:
            synthetic_log_returns = (synthetic_log_returns - synthetic_mean) * (original_std / synthetic_std) + original_mean
        else:
            synthetic_log_returns = synthetic_log_returns - synthetic_mean + original_mean

        synthetic_close = np.empty(n_samples + 1, dtype=np.float64)
        synthetic_close[0] = close_prices[0]
        synthetic_close[1:] = synthetic_close[0] * np.exp(np.cumsum(synthetic_log_returns))
        synthetic_close = np.maximum(synthetic_close, np.finfo(np.float64).tiny)

        # Derive intraday range from the historical candle shape so OHLC remains coherent.
        source_df = source.data.copy()
        if "Open" in source_df.columns:
            open_close_gap = np.log(np.asarray(source_df["Open"].values, dtype=np.float64) / close_prices)
            open_close_gap = np.nan_to_num(open_close_gap, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            open_close_gap = np.zeros_like(synthetic_close)

        open_noise = np.random.choice(open_close_gap, size=n_samples + 1, replace=True)
        synthetic_open = synthetic_close * np.exp(open_noise)

        if {"High", "Low"}.issubset(source_df.columns):
            high_wick = np.log(np.asarray(source_df["High"].values, dtype=np.float64) / np.maximum(close_prices, np.finfo(np.float64).tiny))
            low_wick = np.log(np.asarray(source_df["Low"].values, dtype=np.float64) / np.maximum(close_prices, np.finfo(np.float64).tiny))
            high_wick = np.nan_to_num(high_wick, nan=0.0, posinf=0.0, neginf=0.0)
            low_wick = np.nan_to_num(low_wick, nan=0.0, posinf=0.0, neginf=0.0)
            synthetic_high = np.maximum(synthetic_open, synthetic_close) * np.exp(np.abs(np.random.choice(high_wick, size=n_samples + 1, replace=True)))
            synthetic_low = np.minimum(synthetic_open, synthetic_close) * np.exp(-np.abs(np.random.choice(low_wick, size=n_samples + 1, replace=True)))
        else:
            synthetic_high = np.maximum(synthetic_open, synthetic_close)
            synthetic_low = np.minimum(synthetic_open, synthetic_close)

        synthetic_df = source_df.copy()
        synthetic_df["Close"] = synthetic_close
        synthetic_df["Open"] = synthetic_open
        synthetic_df["High"] = np.maximum.reduce([synthetic_high, synthetic_open, synthetic_close])
        synthetic_df["Low"] = np.minimum.reduce([synthetic_low, synthetic_open, synthetic_close])

        if "Volume" in synthetic_df.columns:
            volume = np.asarray(source_df["Volume"].values, dtype=np.float64)
            if len(volume) == n_samples + 1:
                synthetic_df["Volume"] = np.maximum(0.0, np.random.choice(volume, size=n_samples + 1, replace=True))

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
    simulations: int | None = None,
    mode: MonteCarloMode | str = MonteCarloMode.TRADE_ORDER,
    seed: int | None = None,
    progress_bar: bool = False,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation on the strategy.
    
    This method runs multiple simulations to test strategy robustness using
    either trade order randomization, price path resampling, or both.
    
    Args:
        simulations (int | None, optional): Number of simulations to run.
            Defaults to the number of unique permutations of executed trades
            when mode is "trade_order", otherwise 100.
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

    if simulations is None:
        if mode == MonteCarloMode.TRADE_ORDER:
            simulations = math.factorial(len(original_orders)) if original_orders else 0
        else:
            simulations = 100
    
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
    
    # In BOTH mode we ran two distinct simulation families per iteration.
    # Keep the results and summary statistics separate to avoid pooling
    # different distributions into a single invalid summary.
    if mode == MonteCarloMode.BOTH:
        trade_curves = equity_curves[0::2]
        price_curves = equity_curves[1::2]

        trade_result = MonteCarloResult(
            mode=MonteCarloMode.TRADE_ORDER,
            equity_curves=trade_curves,
            original_equity=original_equity,
            simulations=simulations,
            starting_cash=original_cash,
        )
        trade_result._compute_statistics()

        price_result = MonteCarloResult(
            mode=MonteCarloMode.PRICE_PATH,
            equity_curves=price_curves,
            original_equity=original_equity,
            simulations=simulations,
            starting_cash=original_cash,
        )
        price_result._compute_statistics()

        result = MonteCarloResult(
            mode=mode,
            equity_curves=equity_curves,
            original_equity=original_equity,
            simulations=simulations,
            starting_cash=original_cash,
        )
        result.summary_stats = {
            "trade_order": trade_result.summary_stats,
            "price_path": price_result.summary_stats,
        }
        result.percentile_results = {
            "trade_order": trade_result.percentile_results,
            "price_path": price_result.percentile_results,
        }
    else:
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
