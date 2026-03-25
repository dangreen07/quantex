from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


@dataclass
class OptimizationResult:
    """
    Container for optimization results with train/validate/test splits.
    
    This class holds the complete results of an optimization run that
    includes evaluation on all three data splits, enabling proper
    model selection and generalization assessment.
    
    Attributes:
        best_params (dict): Best parameter values found.
        train_report: Backtest report for training data.
        validate_report: Backtest report for validation data.
        test_report: Backtest report for test data.
        train_metrics (dict): Computed metrics for training performance.
        validate_metrics (dict): Computed metrics for validation performance.
        test_metrics (dict): Computed metrics for test performance.
        all_results (pd.DataFrame): DataFrame with all parameter combinations
            and their metrics for each split.
    """
    best_params: dict
    train_report: Any
    validate_report: Any
    test_report: Any
    train_metrics: dict
    validate_metrics: dict
    test_metrics: dict
    all_results: pd.DataFrame


@dataclass
class BacktestReport:
    """
    Container for backtest results and performance metrics.
    
    This class encapsulates the complete results of a backtest run,
    including P&L records, orders executed, and calculated performance
    metrics such as Sharpe ratio and maximum drawdown.
    
    Attributes:
    starting_cash (np.float64): Initial cash amount at start of backtest.
    final_cash (np.float64): Final cash amount at end of backtest.
    PnlRecord (pd.Series): Time series of P&L values throughout the backtest.
    orders (list[Order]): List of all orders executed during the backtest.
    margin_call_events (list[dict]): Margin call events triggered during the run.
    """
    starting_cash: np.float64
    final_cash: np.float64
    PnlRecord: pd.Series
    orders: list
    tradeRecord: list[np.float64]
    margin_call_events: list[dict] | None = None

    @property
    def annual_rf(self):
        return 0.04

    @property
    def periods_per_year(self):
        """
        Calculate the number of trading periods per year.
        
        This property infers the appropriate number of periods per year
        from the P&L record index, useful for annualized calculations.
        
        Returns:
            int: Number of trading periods per year (e.g., 252 for daily data).
        """
        # Import here to avoid circular imports
        from .metrics import _infer_periods_per_year
        return _infer_periods_per_year(self.PnlRecord.astype(float).index, 252 * 24 * 60)
    
    @property
    def total_return(self):
        equity = self.PnlRecord.astype(float)
        tot_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        return tot_return
    
    @property
    def kelly_criterion(self):
        winning = 0
        losing = 0
        total_wins = 0
        total_losses = 0
        for trade in self.tradeRecord:
            if trade > 0:
                total_wins += abs(trade)
                winning += 1
            elif trade < 0:
                total_losses += abs(trade)
                losing += 1
        if winning + losing == 0:
            return 0.0
        if winning == 0:
            return 0.0
        if losing == 0:
            return 1.0
        W = winning / (winning + losing)
        avg_win = total_wins / winning
        avg_loss = total_losses / losing
        if avg_loss == 0:
            return 0.0
        R = avg_win / avg_loss
        if R == 0:
            return 0.0
        kelly = W - (1 - W) / R
        return kelly

    def plot(self, figsize: tuple = (10, 5)) -> None:
        """
        Plot the equity curve and drawdown charts.
        
        Creates a two-panel plot showing:
        1. The equity curve over time
        2. The drawdown curve as a percentage
        
        Args:
            figsize (tuple, optional): Figure size as (width, height) in inches.
                Defaults to (10, 5).
                
        Note:
            This method uses matplotlib to display the plots and requires
            an interactive environment to show the figures.
        """
        equity = self.PnlRecord.astype(float)
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        fig, ax = plt.subplots(
            2, figsize=figsize, sharex=True
        )

        ax_eq, ax_dd = ax

        ax_eq.plot(equity.index, equity.values, label="Equity", color="tab:blue")
        ax_eq.set_ylabel("Equity Value")
        ax_eq.set_title("Equity Curve")
        ax_eq.legend()
        ax_eq.grid(alpha=0.3)

        ax_dd.fill_between(
            drawdown.index,
            drawdown.values,
            color="tab:red",
            alpha=0.3,
            label="Drawdown",
        )
        ax_dd.set_ylabel("Drawdown")
        ax_dd.set_xlabel("Date")
        ax_dd.legend()
        ax_dd.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        """
        Generate a formatted string summary of backtest results.
        
        Returns a human-readable string containing key performance
        metrics including total return, Sharpe ratio with confidence
        intervals, maximum drawdown, and total number of trades.
        
        Returns:
            str: Formatted string with backtest summary statistics.
        """
        equity = self.PnlRecord.astype(float)
        returns = equity.pct_change().dropna()

        # Risk-free per period from an annual rate
        rf_per_period = self.annual_rf / self.periods_per_year

        # If margin calls occurred, the backtest is invalid - position was forcibly closed
        # This means the strategy was using too much leverage and the Sharpe is meaningless
        margin_calls = len(self.margin_call_events or [])
        if margin_calls > 0:
            sharpe = np.nan
            lo = np.nan
            hi = np.nan
        elif len(returns) < 2 or returns.std(ddof=1) == 0:
            sharpe = np.nan
            lo = np.nan
            hi = np.nan
        else:
            excess = returns - rf_per_period
            mean = excess.mean()
            vol = excess.std(ddof=1)
            sharpe = (mean / vol) * np.sqrt(self.periods_per_year)

            # Standard error of Sharpe (i.i.d. normal approx)
            n = len(excess)
            se = np.sqrt((1 + 0.5 * sharpe**2) / n)
            z = 1.96  # 95% CI
            lo = sharpe - z * se
            hi = sharpe + z * se

        # Max drawdown on equity curve
        running_max = equity.cummax()
        drawdown = ((equity - running_max) / running_max).min()
        mdd = float(abs(drawdown))

        tot_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        annualized_return = float((1.0 + tot_return) ** (self.periods_per_year / max(len(returns), 1)) - 1.0)
        tot_orders = len(self.orders)

        return (
            f"Starting Cash: ${self.starting_cash:,.2f}\n"
            f"Final Cash: ${self.final_cash:,.2f}\n"
            f"Total Return: {tot_return:,.2%}\n"
            f"Annualized Return: {annualized_return:,.2%}\n"
            f"Sharpe Ratio: {sharpe:.2f}" if np.isfinite(sharpe) else
            f"Sharpe Ratio: nan"
        ) + (
            f"\nSharpe Confidence Interval: [{lo:.4f}, {hi:.4f}]"
            if np.isfinite(sharpe) else "\nSharpe Confidence Interval: [nan, nan]"
        ) + (
            f"\nMax Drawdown: {mdd:.2%}\n"
            f"Kelly Fraction: {self.kelly_criterion:.3}\n"
            f"Total Trades: {tot_orders:,}\n"
            f"Margin Calls: {margin_calls:,}"
        )
