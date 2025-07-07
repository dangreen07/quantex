from quantex import DataSource
from quantex.models import Position, Portfolio


class Strategy:
    """Base class for all trading strategies.

    The class now owns a :class:`quantex.models.Portfolio` instance which keeps
    track of cash and *Position* objects.  For convenience and backward
    compatibility, the underlying ``positions`` mapping is exposed directly so
    that existing strategy implementations that reference
    ``self.positions[<symbol>]`` continue to work unchanged.
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
        """Create a new strategy instance.

        Parameters
        ----------
        data_sources
            Mapping from a *source name* to a concrete
            :class:`quantex.sources.DataSource` implementation.
        symbols
            List of tradable symbols the strategy should initialise
            :class:`quantex.models.Position` objects for.  If *None*, no
            positions are pre-populated; they will be created lazily when the
            first trade occurs.
        initial_cash
            Starting cash that will be assigned to the internal
            :class:`quantex.models.Portfolio`.
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

    def _increment_index(self) -> None:
        """Advance the internal bar pointer by **one**.

        This should be called by the back-testing engine **after** all logic
        tied to the current bar has executed.
        """
        self.index += 1

    def run(self):  # pragma: no cover
        """Execute the strategy logic for the current bar.

        Concrete strategies **must** override this method.  It should inspect
        the available data sources, make trading decisions, and update the
        portfolio (typically via the exposed :pycode{self.positions}) before
        the engine advances to the next bar.
        """
        raise NotImplementedError("Subclasses must implement this method.")
