from __future__ import annotations

from datetime import datetime
from typing import Mapping, cast
import pandas as pd

from quantex.execution import ImmediateFillSimulator
from quantex.models import Fill, Order
from quantex.sources import BacktestingDataSource
from quantex.strategy import Strategy


class EventBus:
    """Lightweight dispatcher that coordinates data, strategy, and execution.

    This class is the central coordinator of the backtesting engine. It fetches
    data from data sources, passes it to the strategy for processing, and
    sends any generated orders to the execution simulator.
    """

    def __init__(
        self,
        strategy: Strategy,
        data_sources: Mapping[str, BacktestingDataSource],
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
        # Pre-computed event timeline
        self._timeline: list[datetime] = []
        self._price_df: pd.DataFrame | None = None

    def _precompute_timeline(self) -> None:
        """Gathers all timestamps from all sources and creates a unified timeline."""
        all_timestamps = []
        for ds in self.data_sources.values():
            all_timestamps.extend(ds.get_raw_data().index)

        self._timeline = sorted(list(set(all_timestamps)))

    def _precompute_price_data(self) -> None:
        """Creates a unified dataframe of close prices for all symbols."""
        price_dfs = {}
        for name, ds in self.data_sources.items():
            raw_data = ds.get_raw_data()
            if "close" in raw_data.columns and ds.symbol:
                price_dfs[ds.symbol] = raw_data["close"]

        self._price_df = pd.DataFrame(price_dfs).ffill()

    def run(self) -> None:
        """Runs the simulation until all data is exhausted.

        This method orchestrates the event loop, which proceeds in timestamp
        order. At each step, it determines the earliest timestamp among all
        data sources, processes the data for that moment, executes strategy
        logic, and advances the data sources that produced the event. This
        ensures that data from multiple sources is handled chronologically.
        """
        self._precompute_timeline()
        self._precompute_price_data()

        if self._price_df is None:
            return  # No data to process

        self.strategy.precompute_signals(self._price_df)

        for ts in self._timeline:
            self.strategy.timestamp = ts

            # Get the current prices for all symbols at this timestamp
            current_prices = self._price_df.loc[ts].to_dict()

            # The strategy now needs to be adapted to not rely on get_current_bar
            # This is a major change, for now we will stub the data access
            # so the engine can run.
            # In a real scenario, the Strategy class would be refactored to
            # accept the current price data directly.
            for ds in self.data_sources.values():
                # We need to manually update the index of each data source
                # to keep it in sync with the main timeline.
                if ts in ds.get_raw_data().index:
                    ds.index = cast(int, ds.get_raw_data().index.get_loc(ts))

            # Run strategy logic
            self.strategy.run()

            # Execute orders
            new_orders = self.strategy._pop_pending_orders()
            self.orders.extend(new_orders)
            for order in new_orders:
                execution_price = current_prices.get(order.symbol)
                if execution_price is None:
                    continue
                fill = self.simulator.execute(order, execution_price, ts)
                self.fills.append(fill)

            # Record NAV
            nav = self.strategy.portfolio.net_asset_value(
                {str(k): float(v) for k, v in current_prices.items()}
            )
            self.nav.append(nav)
            self.timestamps.append(ts)

            # The index of the strategy is now the main loop's concern
            self.strategy._increment_index()
