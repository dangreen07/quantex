from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

"""Core data models for QuantEx.

This module defines immutable market-data records (`Bar`, `Tick`), trading
objects (`Order`, `Fill`), and stateful position-keeping helpers
(`Position`, `Portfolio`)."""

"""Market Data Models"""


@dataclass(frozen=True)
class Bar:
    """OHLCV bar for a single symbol and timestamp (usually end-of-period)."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str | None = None


@dataclass(frozen=True)
class Tick:
    """Single tick (trade) quote."""

    timestamp: datetime
    price: float
    volume: float
    symbol: str | None = None


"""Trading Order & Fill"""


@dataclass
class Order:
    """Represents an order submitted by a strategy."""

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
    """Execution of (part of) an Order."""

    order_id: str
    symbol: str
    quantity: float  # positive for buy, negative for sell
    price: float
    timestamp: datetime
    commission: float = 0.0

    def value(self) -> float:
        """Cash impact of the fill (signed)."""
        return -self.quantity * self.price  # buy decreases cash, sell increases


@dataclass
class Trade:
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
        self.symbol = symbol
        self.position: float = 0.0  # signed quantity
        self.trades: list["Trade"] = []
        self.average_price: float = 0.0
        self.realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.position > 0

    @property
    def is_short(self) -> bool:
        return self.position < 0

    @property
    def is_closed(self) -> bool:
        return (self.position - 0) < 1e-8  ## Account for floating point errors

    # Re-use calculation logic from previous Positions implementation
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

        self.position = new_pos
        self.trades.append(Trade(self.symbol, price, quantity, timestamp))

    # Public helpers
    def buy(self, price: float, timestamp: datetime, quantity: float = 1):
        if quantity <= 0:
            raise ValueError("quantity must be positive for buy")
        self._apply_trade(quantity, price, timestamp)

    def sell(self, price: float, timestamp: datetime, quantity: float = 1):
        if quantity <= 0:
            raise ValueError("quantity must be positive for sell")
        self._apply_trade(-quantity, price, timestamp)

    def calculate_total_pnl(self, current_price: float) -> float:
        unrealized = (current_price - self.average_price) * self.position
        return self.realized_pnl + unrealized


class Portfolio:
    """Aggregates cash and multiple Position objects."""

    def __init__(self, cash: float = 0.0):
        self.starting_cash = cash
        self.cash = cash
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0

    def process_fill(self, fill: Fill):
        """Update cash first (commission reduces cash)."""
        self.cash -= fill.quantity * fill.price + fill.commission

        pos = self.positions.get(fill.symbol)
        if pos is None:
            pos = Position(fill.symbol)
            self.positions[fill.symbol] = pos

        prev_realized = pos.realized_pnl
        pos._apply_trade(fill.quantity, fill.price, fill.timestamp)
        self.realized_pnl += pos.realized_pnl - prev_realized

    def net_asset_value(self, price_dict: Dict[str, float]):
        nav = self.cash
        for sym, pos in self.positions.items():
            current_price = price_dict[sym]
            unrealized = (current_price - pos.average_price) * pos.position
            nav += unrealized + pos.average_price * pos.position
        return nav

    def unrealized_pnl(self, price_dict: Dict[str, float]):
        total = 0.0
        for sym, pos in self.positions.items():
            total += (price_dict[sym] - pos.average_price) * pos.position
        return total

    def __repr__(self):
        return (
            f"Portfolio(cash={self.cash:.2f}, realized_pnl={self.realized_pnl:.2f}, "
            f"positions={len(self.positions)})"
        )
