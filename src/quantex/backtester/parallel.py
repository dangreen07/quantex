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
    lot_size: int
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


def _worker_eval(param_items: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
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
    )
    report = bt.run(progress_bar=False)

    # Compute metrics
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
    result: dict[str, Any] = {
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
