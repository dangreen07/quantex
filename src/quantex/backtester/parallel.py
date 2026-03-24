from __future__ import annotations

import gc
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .backtester import SimpleBacktester


# Global variables for worker processes (initialized by _worker_init)
_WORKER_PICKLED_STRAT: bytes | None = None
_WORKER_BT_CONFIG: dict[str, Any] | None = None


def _worker_init(
    pickled_strategy: bytes,
    cash: float,
    commision: float,
    commision_type: Any,
    lot_size: int,
    leverage: float,
) -> None:
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
        leverage (float): Leverage multiplier for position sizing.
        
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
        "leverage": leverage,
    }


def _compute_metrics_numpy(
    equity: np.ndarray,
    periods_per_year: float,
    n_trades: int,
) -> dict[str, Any]:
    """
    Compute performance metrics using numpy arrays directly.
    
    This is more efficient than using pandas operations for the
    inner loop of optimization since we avoid pandas overhead.
    
    Args:
        equity: Numpy array of equity values over time.
        periods_per_year: Number of periods in a year for annualization.
        n_trades: Number of trades executed.
        
    Returns:
        Dictionary with computed metrics.
    """
    # Calculate returns using numpy (avoid pandas overhead)
    equity_arr = equity.astype(np.float64)
    
    # Handle edge cases
    if len(equity_arr) < 2:
        return {
            "final_cash": float(equity_arr[-1]) if len(equity_arr) > 0 else 0.0,
            "total_return": 0.0,
            "sharpe": float("nan"),
            "max_drawdown": 0.0,
            "trades": n_trades,
        }
    
    # Compute returns using numpy
    returns = np.diff(equity_arr) / equity_arr[:-1]
    
    # Remove NaN/Inf values
    valid_returns = returns[np.isfinite(returns)]
    
    # Total return
    tot_return = float(equity_arr[-1] / equity_arr[0] - 1.0) if equity_arr[0] != 0 else 0.0
    
    # Sharpe ratio
    annual_rf = 0.04
    rf_per_period = annual_rf / periods_per_year
    
    if len(valid_returns) < 2:
        sharpe = float("nan")
    else:
        excess = valid_returns - rf_per_period
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)
        if std_excess == 0:
            sharpe = float("nan")
        else:
            sharpe = float((mean_excess / std_excess) * (periods_per_year ** 0.5))
    
    # Maximum drawdown using numpy
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / running_max
    mdd = float(abs(np.min(drawdowns)))
    
    return {
        "final_cash": float(equity_arr[-1]),
        "total_return": tot_return,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "trades": n_trades,
    }


def _worker_eval(param_items: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """
    Worker evaluation function for parallel parameter optimization.
    
    This function runs in worker processes to evaluate a single
    parameter combination and return performance metrics.
    
    Optimizations applied:
    1. Uses numpy for metric computation instead of pandas (faster)
    2. Returns only essential metrics (reduces IPC overhead)
    3. Explicit cleanup of references to help GC
    
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
    assert _WORKER_PICKLED_STRAT is not None, "Worker not initialized"
    strat = pickle.loads(_WORKER_PICKLED_STRAT)

    # Apply param overrides
    for k, v in params.items():
        setattr(strat, k, v)

    # Import here to avoid circular imports
    from .backtester import SimpleBacktester
    
    # Access config (should be initialized by _worker_init)
    assert _WORKER_BT_CONFIG is not None, "Worker not initialized"
    config = _WORKER_BT_CONFIG
    
    # Run backtest locally in worker (no progress bar)
    bt = SimpleBacktester(
        strat,
        cash=config["cash"],
        commission=config["commision"],
        commission_type=config["commision_type"],
        lot_size=config["lot_size"],
        leverage=config["leverage"],
    )
    report = bt.run(progress_bar=False)

    # Compute metrics using optimized numpy version
    # This avoids pandas overhead for metric computation
    # Use to_numpy() with copy=False for efficiency, convert to float64
    equity_values = np.asarray(report.PnlRecord, dtype=np.float64)
    metrics = _compute_metrics_numpy(
        equity=equity_values,
        periods_per_year=report.periods_per_year,
        n_trades=len(report.orders),
    )

    # Build result with params
    result: dict[str, Any] = {
        "params": params,
        **metrics,
    }

    # Cleanup references to free memory inside worker
    del strat, bt, report
    gc.collect()

    return result
