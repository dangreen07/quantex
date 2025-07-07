from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from quantex.engine import EventBus
from quantex.execution import ImmediateFillSimulator
from quantex.sources import DataSource
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

    def __init__(self, strategy: Strategy, data_sources: Dict[str, DataSource]):
        """Initializes the BacktestRunner.

        Args:
            strategy: The trading strategy to be backtested.
            data_sources: A dictionary of data sources for the backtest.
        """
        self.strategy = strategy
        self.data_sources = data_sources
        self.simulator = ImmediateFillSimulator(self.strategy.portfolio)
        self.event_bus = EventBus(strategy, data_sources, self.simulator)

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
        if not nav_series.empty:
            metrics["total_return"] = nav_series.iloc[-1] / nav_series.iloc[0] - 1

        return BacktestResult(
            nav_series, self.event_bus.orders, self.event_bus.fills, metrics
        )
