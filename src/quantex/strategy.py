from quantex import DataSource
from quantex.models import Position, Portfolio, Order
from abc import ABC, abstractmethod


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

    def __init__(
        self,
        data_sources: dict[str, DataSource],
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

        # Store references to market data sources
        self.data_sources = data_sources

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

    def _increment_index(self) -> None:
        """Advances the internal bar pointer by one.

        This should be called by the backtesting engine after all logic
        for the current bar has executed.
        """
        self.index += 1

    @abstractmethod
    def run(self):  # pragma: no cover
        """Executes the strategy logic for the current bar.

        Concrete strategies must override this method. It should inspect
        the available data sources and make trading decisions.
        """
        raise NotImplementedError("Subclasses must implement this method.")

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
