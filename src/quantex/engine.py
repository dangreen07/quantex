from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from quantex.models import Fill, Order
from quantex.sources import DataSource
from quantex.strategy import Strategy
from quantex.execution import ImmediateFillSimulator


class EventBus:
    """Lightweight dispatcher that coordinates data, strategy, and execution.

    This class is the central coordinator of the backtesting engine. It fetches
    data from data sources, passes it to the strategy for processing, and
    sends any generated orders to the execution simulator.
    """

    def __init__(
        self,
        strategy: Strategy,
        data_sources: Dict[str, DataSource],
        simulator: ImmediateFillSimulator,
    ) -> None:
        """Initializes the EventBus.

        Args:
            strategy: The trading strategy to be executed.
            data_sources: A dictionary of data sources.
            simulator: The execution simulator.
        """
        self.strategy = strategy
        self.data_sources = data_sources
        self.simulator = simulator

        self.orders: list[Order] = []
        self.fills: list[Fill] = []
        self.nav: list[float] = []
        self.timestamps: list[datetime] = []

    def _all_sources_have_data(self) -> bool:
        """Checks if all data sources have more data.

        Returns:
            True if all data sources have more data, False otherwise.
        """
        for ds in self.data_sources.values():
            if hasattr(ds, "__len__") and ds.index >= len(ds):  # type: ignore[arg-type]
                return False
        return True

    def step(self) -> bool:
        """Processes one bar of data across all data sources.

        Returns:
            True if the simulation should continue, False if data is exhausted.
        """
        if not self._all_sources_have_data():
            return False

        # Snapshot current bars and prices
        price_dict: Dict[str, float] = {}
        bar_timestamps: List[datetime] = []
        for ds in self.data_sources.values():
            bar = ds.get_current_bar()
            # Symbol may be *None* (should not happen in practice) – fallback to empty string
            symbol = bar.symbol or ""
            price_dict[symbol] = bar.close
            bar_timestamps.append(bar.timestamp)

        # Run strategy logic – it may queue orders
        self.strategy.run()

        # Execute any queued orders via simulator
        new_orders = self.strategy._pop_pending_orders()
        self.orders.extend(new_orders)
        for order in new_orders:
            execution_price = price_dict[order.symbol or ""]
            ts = bar_timestamps[0]  # first timestamp as canonical
            fill = self.simulator.execute(order, execution_price, ts)
            self.fills.append(fill)

        # Record NAV after fills
        nav = self.strategy.portfolio.net_asset_value(price_dict)
        self.nav.append(nav)
        self.timestamps.append(bar_timestamps[0])

        # Advance indices for next bar
        for ds in self.data_sources.values():
            ds._increment_index()
        self.strategy._increment_index()

        return True

    def run(self) -> None:
        """Runs the simulation until all data is exhausted."""
        while self.step():
            continue
