from dataclasses import dataclass
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .broker import Order, OrderSide
from .strategy import Strategy
from .enums import CommissionType
import copy
import itertools
from typing import Callable, Any
from tqdm import tqdm
import concurrent.futures
import pickle
import os
import gc

def max_drawdown(equity: pd.Series) -> float:
    """
    Calculate the maximum drawdown of an equity curve.
    
    The maximum drawdown represents the largest peak-to-trough decline
    in the equity curve, expressed as a positive percentage.
    
    Args:
        equity (pd.Series): Time series of equity values.
        
    Returns:
        float: Maximum drawdown as a positive percentage (e.g., 0.15 for 15%).
        
    Example:
        >>> equity = pd.Series([100, 110, 95, 105, 90])  
        >>> max_drawdown(equity)  
        0.18181818181818182  # ~18.18% drawdown
    """
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    return float(abs(max_dd))  # return as positive percentage

def _infer_periods_per_year(index: pd.Index, default: int = 252 * 24 * 60) -> int:
    """
    Infer the number of trading periods per year from a datetime index.
    
    This function analyzes the time differences in the index to determine
    the appropriate number of periods per year for annualized calculations.
    Falls back to minute-level trading (252 trading days * 24 hours * 60 minutes)
    if the index cannot be analyzed or contains insufficient data.
    
    Args:
        index (pd.Index): DatetimeIndex containing timestamps.
        default (int, optional): Default periods per year for minute trading.
            Defaults to 252 * 24 * 60 (minute-level data).
            
    Returns:
        int: Estimated number of trading periods per year.
        
    Example:
        >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')  
        >>> _infer_periods_per_year(dates)  
        252  # Daily trading periods
    """
    # Simple inference; falls back to minute trading year if uncertain
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return default
    dt = np.diff(index.values).astype("timedelta64[s]").astype(float)
    if not np.isfinite(dt).any():
        return default
    med_sec = np.median(dt[dt > 0])
    if not np.isfinite(med_sec) or med_sec <= 0:
        return default
    periods_per_day = 86400.0 / med_sec
    # Assume 252 trading days/year
    return int(round(252 * periods_per_day))

def _worker_init(pickled_strategy: bytes, cash: float, commision: float,
                 commision_type, lot_size: int):
    """
    Initializer for worker processes in parallel optimization.
    
    This function stores a pickled strategy and backtest configuration
    in module globals so each worker process can reuse them for
    parallel parameter optimization.
    
    Args:
        pickled_strategy (bytes): Serialized strategy instance.
        cash (float): Initial cash amount for backtesting.
        commision (float): Commission rate for trades.
        commision_type: Type of commission calculation (CommissionType enum).
        lot_size (int): Size of trading lots.
        
    Note:
        This function is designed to be called by worker processes
        during parallel optimization and should not be used directly.
    """
    global _WORKER_PICKLED_STRAT, _WORKER_BT_CONFIG
    _WORKER_PICKLED_STRAT = pickled_strategy
    _WORKER_BT_CONFIG = {
        "cash": cash,
        "commision": commision,
        "commision_type": commision_type,
        "lot_size": lot_size,
    }

def _worker_eval(param_items):
    """
    Worker evaluation function for parallel parameter optimization.
    
    This function runs in worker processes to evaluate a single
    parameter combination and return performance metrics.
    
    Args:
        param_items: Sequence of (key, value) pairs (tuple) to reconstruct dict.
                    Each tuple represents a parameter name and its value.
                    
    Returns:
        dict: Dictionary containing metrics for the evaluated parameters:
            - 'params': Dictionary of parameter values used
            - 'final_cash': Final cash amount after backtest
            - 'total_return': Total return as decimal (e.g., 0.15 for 15%)
            - 'sharpe': Sharpe ratio (or NaN if invalid)
            - 'max_drawdown': Maximum drawdown as decimal
            - 'trades': Number of trades executed
            
    Note:
        This function is designed for use in worker processes during
        parallel optimization and should not be called directly.
    """
    global _WORKER_PICKLED_STRAT, _WORKER_BT_CONFIG
    # Reconstruct params dict
    params = dict(param_items)

    # Unpickle a fresh strategy instance for this task
    strat = pickle.loads(_WORKER_PICKLED_STRAT)

    # Apply param overrides
    for k, v in params.items():
        setattr(strat, k, v)

    # Run backtest locally in worker (no progress bar)
    bt = SimpleBacktester(
        strat,
        cash=_WORKER_BT_CONFIG["cash"],
        commission=_WORKER_BT_CONFIG["commision"],
        commission_type=_WORKER_BT_CONFIG["commision_type"],
        lot_size=_WORKER_BT_CONFIG["lot_size"],
    )
    report = bt.run(progress_bar=False)

    # Compute metrics (same logic as before)
    equity = report.PnlRecord.astype(float)
    returns = equity.pct_change().dropna()

    annual_rf = 0.04
    rf_per_period = annual_rf / report.periods_per_year

    if len(returns) < 2 or returns.std(ddof=1) == 0:
        sharpe = float("nan")
    else:
        excess = returns - rf_per_period
        mean = excess.mean()
        vol = excess.std(ddof=1)
        sharpe = float((mean / vol) * (report.periods_per_year ** 0.5))

    running_max = equity.cummax()
    drawdown = ((equity - running_max) / running_max).min()
    mdd = float(abs(drawdown))

    tot_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    # Keep worker returned payload small — don't send large objects back.
    result = {
        "params": params,
        "final_cash": report.final_cash,
        "total_return": tot_return,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "trades": len(report.orders),
    }

    # Cleanup references to free memory inside worker
    del strat, bt, report, equity, returns
    gc.collect()

    return result

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
    """
    starting_cash: np.float64
    final_cash: np.float64
    PnlRecord: pd.Series
    orders: list[Order]
    tradeRecord: list[np.float64]

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

        if len(returns) < 2 or returns.std(ddof=1) == 0:
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
        tot_orders = len(self.orders)

        return (
            f"Starting Cash: ${self.starting_cash:,.2f}\n"
            f"Final Cash: ${self.final_cash:,.2f}\n"
            f"Total Return: {tot_return:,.2%}\n"
            f"Sharpe Ratio: {sharpe:.2f}" if np.isfinite(sharpe) else
            f"Sharpe Ratio: nan"
        ) + (
            f"\nSharpe Confidence Interval: [{lo:.4f}, {hi:.4f}]"
            if np.isfinite(sharpe) else "\nSharpe Confidence Interval: [nan, nan]"
        ) + (
            f"\nMax Drawdown: {mdd:.2%}\n"
            f"Kelly Fraction: {self.kelly_criterion:.3}\n"
            f"Total Trades: {tot_orders:,}"
        )

class SimpleBacktester():
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
                margin_call: float = 0.5 ## 50% of the cash lost
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
                
        Raises:
            ValueError: If strategy is None or commission rate is negative.
        """
        self.strategy = copy.deepcopy(strategy)
        self.cash = cash
        self.commission = commission
        self.commission_type = commission_type
        self.lot_size = lot_size
        self.margin_call = margin_call
        source = self.strategy.positions[list(self.strategy.positions.keys())[0]].source
        self.PnLRecord = np.zeros(len(source.data['Close']), dtype=np.float64)

    def _reset_runtime_state(self) -> None:
        source = self.strategy.positions[list(self.strategy.positions.keys())[0]].source
        self.PnLRecord = np.zeros(len(source.data['Close']), dtype=np.float64)
        self.strategy.indicators = []

        for data_source in self.strategy.data.values():
            data_source.current_index = 0

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
            broker.commision = np.float64(self.commission)
            broker.commision_type = self.commission_type

        self.strategy.init()
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
            tradeRecord=tradeRecord)
    
    def optimize(self, params: dict[str, range], constraint: Callable[[dict[str, Any]], bool] | None = None):
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
                
        Returns:
            tuple: A tuple containing (best_params, best_report, results):
                - best_params (dict[str, Any]): Dictionary of parameter values
                  that produced the best performance.
                - best_report (BacktestReport): Complete backtest report for the
                  best parameter combination.
                - results (pd.DataFrame): DataFrame with metrics for all valid
                  parameter combinations, sorted by performance.
                  
        Raises:
            ValueError: If params is empty or contains parameters with no values.
            TypeError: If any parameter values are not iterable.
            
        Note:
            The optimization uses Sharpe ratio as the primary selection criterion.
            If Sharpe ratio is invalid (NaN), it falls back to total return,
            then to final cash amount.
            
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

        total_combos = len(list(itertools.product(*value_lists)))

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
            )
            report = bt.run(progress_bar=False)

            # Compute metrics
            equity = report.PnlRecord.astype(float)
            returns = equity.pct_change().dropna()

            # Risk-free per period from an annual rate
            annual_rf = 0.04
            rf_per_period = annual_rf / report.periods_per_year

            if len(returns) < 2 or returns.std(ddof=1) == 0:
                sharpe = np.nan
                lo = np.nan
                hi = np.nan
            else:
                excess = returns - rf_per_period
                mean = excess.mean()
                vol = excess.std(ddof=1)
                sharpe = (mean / vol) * np.sqrt(report.periods_per_year)

            # Max drawdown on equity curve
            running_max = equity.cummax()
            drawdown = ((equity - running_max) / running_max).min()
            mdd = float(abs(drawdown))

            tot_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

            row = dict(row_params)
            row.update(
                {
                    "final_cash": report.final_cash,
                    "total_return": tot_return,
                    "sharpe": sharpe,
                    "max_drawdown": mdd,
                    "trades": report.orders,
                }
            )
            results_rows.append(row)

            # Selection score: prefer Sharpe, then total return, then final cash
            score = sharpe
            if not np.isfinite(score):
                score = -1000 ## Really bad

            if score > best_score:
                best_score = score
                best_params = {k: v for k, v in zip(keys, combo)}
                best_report = report

        results_df = pd.DataFrame(results_rows)

        # Sort results by composite score (Sharpe desc, then return, then cash)
        if not results_df.empty:
            def _to_score(val):
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    return None
                return v if np.isfinite(v) else None

            scores = []
            for _, r in results_df.iterrows():
                s = _to_score(r.get("sharpe"))
                if s is None:
                    s = _to_score(r.get("total_return"))
                if s is None:
                    s = _to_score(r.get("final_cash"))
                scores.append(s if s is not None else float("-inf"))
            results_df["_score"] = scores
            results_df.sort_values(by=["_score"], ascending=False, inplace=True, kind="mergesort")
            results_df.drop(columns=["_score"], inplace=True)

        return best_params or {}, best_report, results_df
    
    def optimize_parallel(self,
             params: dict[str, range],
             constraint: Callable[[dict[str, Any]], bool] | None = None,
             workers: int | None = None,
             chunksize: int = 1):
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
            workers (int | None, optional): Maximum number of worker processes to use.
                If None, defaults to min(os.cpu_count()-1, 4) to avoid overwhelming
                the system. Defaults to None.
            chunksize (int, optional): Chunk size for ProcessPoolExecutor.map.
                Smaller values provide better load balancing for many small tasks.
                Larger values reduce overhead for fewer, larger tasks.
                Defaults to 1.
                
        Returns:
            tuple: Same return format as optimize():
                (best_params, best_report, results_df)
                
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
              
        Example:
            >>> bt = SimpleBacktester(strategy)  
            >>> # Use 4 workers for parallel optimization  
            >>> best_params, best_report, results = bt.optimize_parallel(  
            ...     {'period1': range(5, 50, 5), 'period2': range(20, 100, 10)},  
            ...     workers=4  
            ... )  
        """
        if not params:
            raise ValueError("params must not be empty")

        keys = list(params.keys())
        value_lists = []
        lens = []
        for k in keys:
            vals = params[k]
            try:
                candidates = list(vals)
            except TypeError:
                raise TypeError(f"Parameter '{k}' must be iterable")
            if len(candidates) == 0:
                raise ValueError(f"Parameter '{k}' has no candidate values")
            value_lists.append(candidates)
            lens.append(len(candidates))

        # determine total combos without materializing them
        total_combos = 1
        for L in lens:
            total_combos *= L

        # prepare iterable of param dicts as sequences of items (so pickling is slightly cheaper)
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

        # choose worker count conservatively to avoid RAM hogging
        cpu_count = os.cpu_count() or 1
        if workers is None:
            workers = max(1, min(cpu_count - 1, 4))
        else:
            workers = max(1, int(workers))

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
            ),
        ) as exe:
            # map the worker over param item tuples
            # use list() on map to iterate with tqdm and collect results
            it = exe.map(_worker_eval, _param_items_iter(), chunksize=chunksize)
            # iterate with progress display
            for res in tqdm(it, total=total_combos, disable=(total_combos <= 1)):
                results_rows.append(res)

        # Build DataFrame of small metrics returned from workers
        results_df = pd.DataFrame(results_rows)
        # Compute a composite score like before: prefer sharpe, then return, then final_cash
        if not results_df.empty:
            def _to_score(val):
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    return None
                return v if np.isfinite(v) else None

            scores = []
            for _, r in results_df.iterrows():
                ret = r.get("total_return")
                s = None
                if (not ret == None and ret > 0):
                    s = _to_score(r.get("sharpe"))
                scores.append(s if s is not None else float("-inf"))
            results_df["_score"] = scores
            results_df.sort_values(by=["_score"], ascending=False, inplace=True, kind="mergesort")
            results_df.drop(columns=["_score"], inplace=True)

        # Determine best params from results_df if any
        if results_df.empty:
            return {}, None, results_df

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
        )
        best_report = bt.run(progress_bar=False)

        return best_params, best_report, results_df
