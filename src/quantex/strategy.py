from quantex import DataSource
from quantex.models import Position, Portfolio, Order
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Mapping
import pandas as pd


class Strategy(ABC):
    """Base class for all trading strategies.

    This class owns a `quantex.models.Portfolio` instance which keeps
    track of cash and `Position` objects. For convenience and backward
    compatibility, the underlying `positions` mapping is exposed directly so
    that existing strategy implementations that reference
    `self.positions[<symbol>]` continue to work unchanged.
    """

    # The engine (or outer loop) advances *all* data sources by calling
    # :pycode{_increment_index} on them **and** on the strategy to allow for
    # strategy-level book-keeping.
    index: int = 0
    timestamp: datetime | None = None
    _ind_state: dict[tuple, tuple] = {}

    def __init__(
        self,
        data_sources: Mapping[str, DataSource] | None = None,
        symbols: list[str] | None = None,
        *,
        initial_cash: float = 0.0,
    ) -> None:
        """Initializes a new strategy instance.

        Args:
            data_sources: Mapping from a source name to a concrete
                `quantex.sources.DataSource` implementation.
            symbols: List of tradable symbols to initialize `Position` objects
                for. If None, positions are created lazily.
            initial_cash: Starting cash for the internal `Portfolio`.
        """

        # Store references to market data sources (may be empty – engine injects)
        self.data_sources = data_sources or {}
        self.timestamp = None

        # Maintain a portfolio to aggregate cash & PnL
        self.portfolio: Portfolio = Portfolio(cash=initial_cash)

        # Expose positions dict for backward compatibility
        self.positions = self.portfolio.positions

        # Register provided symbols (if any)
        self.symbols = symbols or []
        for sym in self.symbols:
            # Pre-create empty Position objects so that strategy code can rely
            # on their existence.
            if sym not in self.positions:
                self.positions[sym] = Position(sym)

        # Queue of orders submitted during the current bar – cleared each step
        self._pending_orders: list[Order] = []
        self.signals = pd.DataFrame()

        # Cache for computed indicator Series keyed by (func_name, symbol, param_tuple)
        self._indicator_cache: dict[tuple, pd.Series] = {}

        # --- Internals set by the engine each bar ---
        self._price_row: list[float] | None = None
        self._symbols: list[str] | None = None
        self._symbol_idx: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # Convenience Properties
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        """Current available cash in the underlying Portfolio.

        Example:
            ```python
            if self.cash > 10_000:
                self.buy("AAPL", 5)
            ```

        Returns:
            float: The cash balance that can be deployed for new positions.
        """

        return self.portfolio.cash

    def _increment_index(self) -> None:
        """Advances the internal bar pointer by one.

        This should be called by the backtesting engine after all logic
        for the current bar has executed.
        """
        self.index += 1

    @abstractmethod
    def run(self):
        """Executes the strategy logic for the current bar.

        Concrete strategies must override this method. It should inspect
        the available data sources and make trading decisions.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def buy(
        self,
        symbol: str,
        quantity: float | None = None,
        *,
        cash: float | None = None,
        limit_price: float | None = None,
    ) -> None:
        """Creates and submits a buy order.

        The method is flexible:

        1. **Quantity-based** (classic):
            ```python
            self.buy("AAPL", quantity=10)
            ```

        2. **Cash-based** – buy as many shares as *cash* allows:
            ```python
            self.buy("AAPL", cash=5_000)
            ```

        3. **Max-size** – omit both *quantity* and *cash* to invest all
           available cash:
            ```python
            self.buy("AAPL")
            ```

        Args:
            symbol: The instrument to purchase.
            quantity: Number of shares to buy. If ``None`` the method will
                derive it from *cash* or the account's entire cash balance.
            cash: Dollar amount to deploy. Ignored when *quantity* is
                provided. Mutually exclusive with *quantity*.
            limit_price: Optional limit order price. If omitted, a market
                order is created.
        """

        if self.timestamp is None:
            raise RuntimeError("Cannot place order: strategy timestamp is not set.")

        if quantity is None:
            # Determine execution price reference for sizing.
            price_ref = (
                limit_price if limit_price is not None else self.get_price(symbol)
            )

            deployable_cash = self.cash if cash is None else min(cash, self.cash)
            if deployable_cash <= 0:
                raise ValueError("Insufficient cash to size order.")

            # Allow fractional sizing (e.g. for crypto) – use exact ratio
            quantity = deployable_cash / price_ref

        if quantity <= 0:
            raise ValueError("Quantity must be > 0 after sizing.")

        order = Order(
            id=f"buy-{symbol}-{self.timestamp}",
            symbol=symbol,
            side="buy",
            quantity=quantity,
            order_type="limit" if limit_price else "market",
            limit_price=limit_price,
            timestamp=self.timestamp,
        )
        self.submit_order(order)

    def sell(
        self,
        symbol: str,
        quantity: float | None = None,
        *,
        cash: float | None = None,
        limit_price: float | None = None,
    ) -> None:
        """Creates and submits a sell order using the current strategy timestamp.

        Args:
            symbol: The symbol to sell.
            quantity: The quantity to sell. Must be positive.
            limit_price: If provided, a limit order is created. Otherwise, a
                market order is created.
        """
        if self.timestamp is None:
            raise RuntimeError("Cannot place order: strategy timestamp is not set.")

        # ------------------------------------------------------------------
        # Flexible sizing logic (mirrors buy())
        # ------------------------------------------------------------------

        if quantity is None:
            # Reference price for sizing – fall back to current market price
            price_ref = (
                limit_price if limit_price is not None else self.get_price(symbol)
            )

            if cash is not None:
                # Size based on desired cash proceeds
                quantity = cash / price_ref
            else:
                # Neither quantity nor cash provided – default to closing the
                # existing long position (if any)
                position = self.positions.get(symbol)
                if position is None or position.is_closed or position.position <= 0:
                    raise ValueError(
                        "Unable to infer sell quantity – provide quantity or cash, or ensure a long position is open."
                    )
                quantity = position.position  # sell entire long exposure

        if quantity <= 0:
            raise ValueError("Quantity must be > 0 after sizing.")

        order = Order(
            id=f"sell-{symbol}-{self.timestamp}",
            symbol=symbol,
            side="sell",
            quantity=quantity,  # type: ignore[arg-type]
            order_type="limit" if limit_price else "market",
            limit_price=limit_price,
            timestamp=self.timestamp,
        )
        self.submit_order(order)

    def close_position(self, symbol: str) -> None:
        """Creates and submits an order to close the entire position for a symbol.

        This is a helper method that checks if a position is open for the
        given symbol and, if so, creates a market order to close it at the
        current strategy timestamp.

        Args:
            symbol: The symbol of the position to close.
        """
        if self.timestamp is None:
            raise RuntimeError("Cannot place order: strategy timestamp is not set.")

        position = self.positions.get(symbol)
        if not position or position.is_closed:
            return

        if position.is_long:
            side = "sell"
            quantity = position.position
        else:  # is_short
            side = "buy"
            quantity = abs(position.position)

        order = Order(
            id=f"close-{symbol}-{self.timestamp}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            timestamp=self.timestamp,
        )
        self.submit_order(order)

    def submit_order(self, order: Order) -> None:
        """Queues an order to be executed by the engine.

        Strategies should call this method to simulate realistic order routing.

        Args:
            order: The `Order` to be submitted.
        """

        self._pending_orders.append(order)

    def _pop_pending_orders(self) -> list[Order]:
        """Returns and clears the list of queued orders. For internal use."""
        orders, self._pending_orders = self._pending_orders, []
        return orders

    def get_price(self, symbol: str) -> float:
        """Returns the latest price for *symbol*.

        This accesses the numpy price row injected by the engine and is
        therefore O(1) without any pandas overhead. Raises ``KeyError`` if the
        symbol is not part of the backtest universe.
        """

        if self._price_row is None or self._symbol_idx is None:
            raise RuntimeError("Market data not yet initialised for this bar.")
        idx = self._symbol_idx.get(symbol)
        if idx is None:
            raise KeyError(symbol)
        return float(self._price_row[idx])

    @property
    def prices(self) -> dict[str, float]:
        """Lazy dict of symbol→price for the current bar (lightweight)."""

        if self._price_row is None or self._symbols is None:
            raise RuntimeError("Market data not yet initialised for this bar.")
        return {sym: float(price) for sym, price in zip(self._symbols, self._price_row)}

    def _get_price_history_df(self) -> pd.DataFrame:
        """Internal helper to fetch the full, forward-filled price DataFrame.

        Returns:
            pd.DataFrame: Price data for *all* symbols in the backtest universe
                (columns) indexed by the global event timeline. Values are
                forward-filled by the engine ensuring there are no missing
                timestamps – ideal for multi-asset lookback computations.

        Raises:
            RuntimeError: If the strategy is not attached to an *EventBus* or
                if the price DataFrame has not been initialised yet (i.e. the
                first bar has not been processed).
        """

        event_bus = getattr(self, "event_bus", None)
        if event_bus is None:
            raise RuntimeError(
                "Strategy is not attached to an EventBus; price history unavailable."
            )

        price_df: pd.DataFrame | None = (
            event_bus._price_df
        )  # pylint: disable=protected-access
        if price_df is None:
            raise RuntimeError(
                "Price history not yet initialised – wait until the first bar has been processed."
            )

        return price_df

    @property
    def price_history(self) -> pd.DataFrame:  # noqa: D401 – property describing data
        """Price history up to *and including* the current bar.

        Example:
            > history = self.price_history
            > latest_btc = history["BTC"].iloc[-1]
        """

        df = self._get_price_history_df()
        # ``self.index`` corresponds to the *current* bar position
        return df.iloc[: self.index + 1]

    def get_lookback_prices(self, lookback_period: int) -> pd.DataFrame:
        """Returns an *aligned* lookback window for all symbols.

        Args:
            lookback_period: Number of bars (inclusive of the current bar) to
                return.

        Returns:
            pd.DataFrame: DataFrame with the last *lookback_period* rows from
            :pyattr:`price_history`. If there are fewer than *lookback_period*
            observations available (e.g. at the beginning of a backtest), the
            entire available history is returned instead. The returned frame
            is guaranteed to be free of missing timestamps across symbols due
            to the forward-filling performed by the engine.
        """

        history = self.price_history
        if len(history) < lookback_period:
            return history.copy()
        return history.iloc[-lookback_period:]

    # Called by EventBus – should be considered *private*
    def _update_market_data(self, price_row, symbols, symbol_idx):
        self._price_row = price_row
        self._symbols = symbols
        self._symbol_idx = symbol_idx
