"""Backtest orchestration utilities.

This module exposes:
    * ``BacktestResult`` – a dataclass aggregating NAV, orders, fills and metrics.
    * ``BacktestRunner`` – a convenience wrapper that wires together a Strategy,
      one or more DataSource objects, an execution simulator and the internal
      EventBus. End-users typically instantiate ``BacktestRunner`` once per
      test and call `run()` to obtain a `BacktestResult`.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from quantex.engine import EventBus
from quantex.execution import ImmediateFillSimulator
from quantex.sources import BacktestingDataSource
from quantex.strategy import Strategy
from quantex.models import Order, Fill


@dataclass
class BacktestResult:
    """Contains the results of a backtest.

    Attributes:
        nav: A pandas Series representing the Net Asset Value (NAV) over time.
        orders: A list of all orders generated during the backtest.
        fills: A list of all fills executed during the backtest.
        metrics: A dictionary of performance metrics.
    """

    nav: pd.Series
    orders: list[Order]
    fills: list[Fill]
    metrics: dict


class BacktestRunner:
    """User-facing helper that wires Strategy, EventBus, and Simulator."""

    def __init__(
        self,
        strategy: Strategy,
        data_sources: Mapping[str, BacktestingDataSource],
        risk_free_rate: float = 0.043,
        periods_per_year: int = 98_280,
    ):
        """Initializes the BacktestRunner.

        Args:
            strategy: The trading strategy to be backtested.
            data_sources: A dictionary of data sources for the backtest.
            risk_free_rate: The risk-free rate to use for the Sharpe ratio.
            periods_per_year: The number of periods per year to use for the Sharpe ratio.
        """
        self.strategy = strategy
        self.data_sources = data_sources
        self.simulator = ImmediateFillSimulator(self.strategy.portfolio)
        self.event_bus = EventBus(strategy, data_sources, self.simulator)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def run(self) -> BacktestResult:
        """Runs the back-test.

        Returns:
            A `BacktestResult` object containing the results.
        """
        self.event_bus.run()
        nav_series = pd.Series(
            self.event_bus.nav, index=self.event_bus.timestamps, name="NAV"
        )

        metrics: dict = {}

        if not nav_series.empty and nav_series.iloc[0] != 0:
            metrics["total_return"] = nav_series.iloc[-1] / nav_series.iloc[0] - 1

            # Sharpe ratio -------------------------------------------------
            if len(nav_series) > 1:
                metrics["sharpe_ratio"] = _annualised_sharpe(
                    nav_series, self.risk_free_rate, self.periods_per_year
                )

        return BacktestResult(
            nav_series, self.event_bus.orders, self.event_bus.fills, metrics
        )


def _annualised_sharpe(
    nav: pd.Series, risk_free_rate: float = 0.043, periods_per_year: int = 98_280
) -> float:
    """Compute the annualised Sharpe ratio for a NAV series.

    Args:
        nav: Series of portfolio values indexed by timestamp.
        risk_free_rate: Annual risk-free rate expressed as a decimal. Defaults to
            4.3 % (0.043).
        periods_per_year: Number of return observations in a year. For US
            equity market minutes this is ``252 * 390 = 98_280``. For assets
            trading 24/7 (e.g. crypto) use ``365 * 1_440``.

    Returns:
        Annualised Sharpe ratio. If the standard deviation of excess returns is
        zero the function returns ``np.nan``.
    """

    # Minute-to-minute returns
    returns = nav.pct_change().dropna()
    if returns.empty:
        return float("nan")

    # Per-period risk-free rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess = returns - rf_per_period

    std = excess.std(ddof=0)
    if std == 0:
        return float("nan")

    return np.sqrt(periods_per_year) * excess.mean() / std
