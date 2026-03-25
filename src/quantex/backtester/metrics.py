from __future__ import annotations

import pandas as pd
import numpy as np
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .reports import BacktestReport


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


def _compute_backtest_metrics(report: BacktestReport) -> dict[str, Any]:
    """Compute performance metrics from a BacktestReport."""
    equity = report.PnlRecord.astype(float)
    returns = equity.pct_change().dropna()

    annual_rf = report.annual_rf
    rf_per_period = annual_rf / report.periods_per_year

    # If margin calls occurred, the backtest is invalid - position was forcibly closed
    # This means the strategy was using too much leverage and the Sharpe is meaningless
    margin_calls = len(report.margin_call_events or [])
    if margin_calls > 0:
        sharpe = float("nan")
    elif len(returns) < 2 or returns.std(ddof=1) == 0:
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

    return {
        "final_cash": report.final_cash,
        "total_return": tot_return,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "trades": len(report.orders),
    }


def _extract_metric_value(report: BacktestReport, metric: str) -> Any:
    """Extract a metric value from a BacktestReport."""
    value = getattr(report, metric, None)
    if callable(value):
        value = value()
    return value


def _risk_tolerance_passes(report: BacktestReport, risk_tolerance: dict[str, float] | None) -> bool:
    """Check if a report passes risk tolerance constraints."""
    if not risk_tolerance:
        return True

    metrics = _compute_backtest_metrics(report)
    for metric, max_value in risk_tolerance.items():
        if max_value is None:
            continue
        current_value = metrics.get(metric, _extract_metric_value(report, metric))
        if current_value is None:
            raise AttributeError(f"BacktestReport does not expose metric '{metric}'")
        if not np.isfinite(float(current_value)):
            return False
        if float(current_value) > float(max_value):
            return False

    return True
