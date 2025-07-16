"""Core data models for QuantEx.

This module defines immutable market-data records (`Bar`, `Tick`), trading
objects (`Order`, `Fill`), and stateful position-keeping helpers
(`Position`, `Portfolio`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict


@dataclass(frozen=True)
class Bar:
    """OHLCV bar for a single symbol and timestamp.

    Attributes:
        timestamp: The timestamp of the bar (usually end-of-period).
        open: The opening price.
        high: The highest price.
        low: The lowest price.
        close: The closing price.
        volume: The trading volume.
        symbol: The symbol of the instrument.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str | None = None


@dataclass(frozen=True)
class Tick:
    """Single tick (trade) quote.

    Attributes:
        timestamp: The timestamp of the tick.
        price: The price of the tick.
        volume: The volume of the tick.
        symbol: The symbol of the instrument.
    """

    timestamp: datetime
    price: float
    volume: float
    symbol: str | None = None


"""Trading Order & Fill"""


@dataclass
class Order:
    """Represents an order submitted by a strategy.

    Attributes:
        id: The unique identifier for the order.
        symbol: The symbol of the instrument to trade.
        side: The side of the order, either 'buy' or 'sell'.
        quantity: The quantity of the instrument to trade.
        order_type: The type of order, e.g., 'market' or 'limit'.
        limit_price: The limit price for a limit order.
        timestamp: The time the order was created.
    """

    id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: str = "market"  # e.g. market / limit
    limit_price: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit_price required for limit order")


@dataclass
class Fill:
    """Represents the execution of (part of) an Order.

    Attributes:
        order_id: The ID of the order that was filled.
        symbol: The symbol of the instrument that was traded.
        quantity: The quantity of the instrument that was traded. Positive
            for a buy, negative for a sell.
        price: The price at which the trade was executed.
        timestamp: The time of the execution.
        commission: The commission paid for the trade.
    """

    order_id: str
    symbol: str
    quantity: float  # positive for buy, negative for sell
    price: float
    timestamp: datetime
    commission: float = 0.0

    def value(self) -> float:
        """Calculates the cash impact of the fill.

        Returns:
            The signed cash impact of the fill. A buy decreases cash, while a
            sell increases it.
        """
        return -self.quantity * self.price  # buy decreases cash, sell increases


@dataclass
class Trade:
    """Represents a single trade.

    Attributes:
        symbol: The symbol of the instrument traded.
        price: The price of the trade.
        quantity: The quantity of the trade (signed).
        timestamp: The timestamp of the trade.
    """

    symbol: str
    price: float
    quantity: float  # signed (+ buy, - sell)
    timestamp: datetime

    def __str__(self):
        return (
            f"Trade(symbol={self.symbol}, price={self.price:.2f}, "
            f"quantity={self.quantity:.1f})"
        )


"""Position & Portfolio"""


class Position:
    """Tracks position and P&L for a single symbol."""

    def __init__(self, symbol: str):
        """Initializes a new Position.

        Args:
            symbol: The symbol for this position.
        """
        self.symbol = symbol
        self.position: float = 0.0  # signed quantity
        self.trades: list["Trade"] = []
        self.average_price: float = 0.0
        self.realized_pnl: float = 0.0
        # Timestamp when the current exposure was opened. None if flat.
        self.open_timestamp: datetime | None = None

    @property
    def is_long(self) -> bool:
        """Returns True if the position is long."""
        return self.position > 0

    @property
    def is_short(self) -> bool:
        """Returns True if the position is short."""
        return self.position < 0

    @property
    def is_closed(self) -> bool:
        """Returns True if the position is closed."""
        return abs(self.position) < 1e-8  # Account for floating point errors

    def _apply_trade(self, quantity: float, price: float, timestamp: datetime):
        prev_pos = self.position
        new_pos = prev_pos + quantity

        if prev_pos == 0:
            """Opening a new position."""
            self.average_price = price
        elif prev_pos * quantity > 0:
            """Increasing position in same direction."""
            total_size = abs(prev_pos) + abs(quantity)
            self.average_price = (
                self.average_price * abs(prev_pos) + price * abs(quantity)
            ) / total_size
        else:
            """Reducing or flipping position."""
            closing_size = min(abs(quantity), abs(prev_pos))
            sign_prev = 1 if prev_pos > 0 else -1
            self.realized_pnl += (price - self.average_price) * closing_size * sign_prev

            if new_pos == 0:
                self.average_price = 0.0
            elif abs(quantity) > abs(prev_pos):
                """Direction flip: cost basis reset."""
                self.average_price = price

        # Update position quantity
        self.position = new_pos

        # --- Maintain open_timestamp ----------------------------------
        if prev_pos == 0 and new_pos != 0:
            # Transition from flat to open – record entry time
            self.open_timestamp = timestamp
        elif new_pos == 0:
            # Flat again → reset
            self.open_timestamp = None
        elif prev_pos * new_pos < 0:
            # Direction flip (crossed through zero) → new exposure starts now
            self.open_timestamp = timestamp

        self.trades.append(Trade(self.symbol, price, quantity, timestamp))

    def buy(self, quantity: float, price: float, timestamp: datetime):
        """Increases long exposure.

        Args:
            quantity: The amount to buy. Must be positive.
            price: The price of the purchase.
            timestamp: The time of the purchase.
        """

        if quantity <= 0:
            raise ValueError("quantity must be positive for buy")
        self._apply_trade(quantity, price, timestamp)

    def sell(self, quantity: float, price: float, timestamp: datetime):
        """Decreases or flips exposure by selling.

        Args:
            quantity: The amount to sell. Must be positive.
            price: The price of the sale.
            timestamp: The time of the sale.
        """

        if quantity <= 0:
            raise ValueError("quantity must be positive for sell")
        self._apply_trade(-quantity, price, timestamp)

    def calculate_total_pnl(self, current_price: float) -> float:
        """Calculates the total P&L for this position.

        Args:
            current_price: The current market price of the symbol.

        Returns:
            The total P&L (realized + unrealized).
        """
        unrealized = (current_price - self.average_price) * self.position
        return self.realized_pnl + unrealized


class Portfolio:
    """Aggregates cash and multiple Position objects."""

    def __init__(self, cash: float = 0.0):
        """Initializes the Portfolio.

        Args:
            cash: The starting cash balance.
        """
        self.starting_cash = cash
        self.cash = cash
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0

    def process_fill(self, fill: Fill):
        """Updates the portfolio based on a fill.

        This method updates cash and the relevant position.

        Args:
            fill: The Fill object to process.
        """
        self.cash -= fill.quantity * fill.price + fill.commission

        pos = self.positions.get(fill.symbol)
        if pos is None:
            pos = Position(fill.symbol)
            self.positions[fill.symbol] = pos

        prev_realized = pos.realized_pnl
        pos._apply_trade(fill.quantity, fill.price, fill.timestamp)
        self.realized_pnl += pos.realized_pnl - prev_realized

    def net_asset_value(self, price_dict: Dict[str, float]) -> float:
        """Calculates the total value of the portfolio.

        Args:
            price_dict: A dictionary mapping symbols to their current prices.

        Returns:
            The Net Asset Value (NAV) of the portfolio.
        """
        nav = self.cash
        for sym, pos in self.positions.items():
            current_price = price_dict[sym]
            unrealized = (current_price - pos.average_price) * pos.position
            nav += unrealized + pos.average_price * pos.position
        return nav

    def net_asset_value_array(self, price_array, symbol_idx: Dict[str, int]) -> float:
        """Compute NAV using a NumPy row for maximum performance.

        Args:
            price_array (numpy.ndarray): 1-D array *aligned with* ``symbol_idx``
                representing the latest prices for the entire universe.
            symbol_idx (dict[str, int]): Mapping from symbol to its position in
                ``price_array``.

        Returns:
            float: Total net-asset-value (cash + market value of positions).
        """
        nav = self.cash
        for sym, pos in self.positions.items():
            idx = symbol_idx[sym]
            current_price = float(price_array[idx])
            unrealized = (current_price - pos.average_price) * pos.position
            nav += unrealized + pos.average_price * pos.position
        return nav

    def unrealized_pnl(self, price_dict: Dict[str, float]) -> float:
        """Calculates the unrealized P&L of the portfolio.

        Args:
            price_dict: A dictionary mapping symbols to their current prices.

        Returns:
            The total unrealized P&L across all positions.
        """
        total = 0.0
        for sym, pos in self.positions.items():
            total += (price_dict[sym] - pos.average_price) * pos.position
        return total

    def __repr__(self):
        return (
            f"Portfolio(cash={self.cash:.2f}, realized_pnl={self.realized_pnl:.2f}, "
            f"positions={len(self.positions)})"
        )
