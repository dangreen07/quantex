"""Core event loop orchestrating data flow between sources, strategy and simulator.

The `EventBus` pre-computes a global timeline, vectorises price data into
NumPy arrays for speed, injects market snapshots into the running
`quantex.strategy.Strategy` and records NAV, orders and fills.

Google-style docstrings are used throughout for consistency.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping
import pandas as pd

from quantex.execution import ImmediateFillSimulator, NextBarSimulator
from quantex.models import Fill, Order
from quantex.sources import BacktestingDataSource
from quantex.strategy import Strategy
from tqdm import tqdm


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
        simulator: ImmediateFillSimulator | NextBarSimulator,
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

        # Expose EventBus to the simulator (used by NextBarSimulator to fetch
        # open prices).
        setattr(self.simulator, "event_bus", self)

        setattr(self.strategy, "event_bus", self)

        self.orders: list[Order] = []
        self.fills: list[Fill] = []
        self.nav: list[float] = []
        self.timestamps: list[datetime] = []
        # Pre-computed event timeline
        self._timeline: list[datetime] = []
        self._price_df: pd.DataFrame | None = None

    def _precompute_timeline(self) -> None:
        """Computes a *synchronised* global timeline.

        The timeline now contains **only** those timestamps that are present in
        *all* data sources (i.e. the set intersection). This guarantees that
        every bar processed by the engine has a corresponding observation for
        every symbol, removing the need for forward-filling or other alignment
        work-arounds downstream.
        """

        timeline: pd.Index | None = None
        for ds in self.data_sources.values():
            idx = ds.get_raw_data().index
            timeline = idx if timeline is None else timeline.intersection(idx)

        # No common timestamps → empty timeline
        if timeline is None:
            self._timeline = []
        else:
            # ``intersection`` preserves order of the left operand, but we
            # explicitly sort to ensure monotonically increasing timestamps.
            self._timeline = timeline.sort_values().to_list()

    def _precompute_price_data(self) -> None:
        """Builds a price matrix strictly aligned to the global timeline."""

        # Gather close price series for each symbol
        price_series: dict[str, pd.Series] = {}
        for _, ds in self.data_sources.items():
            raw_data = ds.get_raw_data()
            if "close" in raw_data.columns and ds.symbol:
                price_series[ds.symbol] = raw_data["close"]

        # Assemble into a single DataFrame and forward-fill within each column
        price_df = pd.DataFrame(price_series).sort_index().ffill()

        # Restrict to *only* the timestamps present in ``self._timeline`` so the
        # row index aligns 1-to-1 with the event loop.
        if self._timeline:
            price_df = price_df.loc[self._timeline]

        self._price_df = price_df

        if not self._price_df.empty:
            self._symbols = list(self._price_df.columns)
            self._price_array = self._price_df.to_numpy(dtype="float64", copy=False)
            self._symbol_idx = {sym: i for i, sym in enumerate(self._symbols)}

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

        for row_idx, ts in tqdm(enumerate(self._timeline), total=len(self._timeline)):
            self.strategy.timestamp = ts

            price_row = self._price_array[row_idx]

            # Flush any orders queued for execution at *this* timestamp.
            if isinstance(self.simulator, NextBarSimulator):
                new_fills = self.simulator.flush_pending(
                    ts, price_row, self._symbol_idx
                )
            else:
                new_fills = []
            self.fills.extend(new_fills)

            # Inject current market snapshot into strategy (very low overhead)
            self.strategy._update_market_data(
                price_row, self._symbols, self._symbol_idx
            )

            # Align each data source to the global timeline & advance pointer
            for ds in self.data_sources.values():
                if ds.peek_timestamp() == ts:
                    ds._increment_index()

            # Run strategy logic
            self.strategy.run()

            # Execute orders
            new_orders = self.strategy._pop_pending_orders()
            self.orders.extend(new_orders)
            for order in new_orders:
                # Guard: if symbol not in price mapping, skip execution
                idx = self._symbol_idx.get(order.symbol)
                if idx is None:
                    continue
                execution_price = float(price_row[idx])
                fill = self.simulator.execute(order, execution_price, ts)
                if fill is not None:
                    self.fills.append(fill)

            # Record NAV (vectorised prices dict → float conversion once)
            nav = self.strategy.portfolio.net_asset_value_array(
                price_row, self._symbol_idx
            )
            self.nav.append(nav)
            self.timestamps.append(ts)

            # Advance strategy book-keeping pointer
            self.strategy._increment_index()
