from datetime import datetime
from quantex import DataSource

class Trade:
    """A single trade."""
    def __init__(self, symbol: str, price: float, quantity: float, timestamp: datetime):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp
    
    def __str__(self):
        return f"Trade(symbol={self.symbol}, price={self.price}, quantity={self.quantity})"

## Revised implementation of Positions to support trading and P&L tracking
class Positions:
    """Tracks the position, trades, and P&L for a single symbol."""

    def __init__(self, symbol: str):
        self.symbol: str = symbol

        # Current position size. Positive => long, Negative => short.
        self.position: float = 0.0

        # Volume-weighted average entry price of the *open* position.
        # When flat, this will be 0.0.
        self.avg_price: float = 0.0

        # Accumulated realised P&L from all closed trades.
        self.realized_pnl: float = 0.0

        # Store every trade that flows through the object (optional but useful for audit).
        self.trades: list[Trade] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def buy(self, quantity: float, price: float, timestamp: datetime | None = None):
        """Execute a *buy* order (positive quantity)."""
        self._validate_order(quantity, price)
        trade = Trade(self.symbol, price, quantity, timestamp or datetime.now())
        self.trades.append(trade)
        self._update_position(side="buy", quantity=quantity, price=price)

    def sell(self, quantity: float, price: float, timestamp: datetime | None = None):
        """Execute a *sell* order (positive quantity)."""
        self._validate_order(quantity, price)
        trade = Trade(self.symbol, price, quantity, timestamp or datetime.now())
        self.trades.append(trade)
        self._update_position(side="sell", quantity=quantity, price=price)

    def unrealized_pnl(self, market_price: float) -> float:
        """Current unrealised P&L given the latest *market_price*."""
        return (market_price - self.avg_price) * self.position if self.position else 0.0

    def total_pnl(self, market_price: float) -> float:
        """Realised + unrealised P&L."""
        return self.realized_pnl + self.unrealized_pnl(market_price)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_order(quantity: float, price: float):
        if quantity <= 0:
            raise ValueError("Quantity must be positive.")
        if price <= 0:
            raise ValueError("Price must be positive.")

    def _update_position(self, *, side: str, quantity: float, price: float):
        """Core position & P&L bookkeeping.

        This method supports both long and short positioning using an average cost
        approach. Realised P&L is computed whenever an opposite-side trade
        reduces or flips the current position.
        """

        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")

        # Determine the *signed* trade quantity (positive for buys, negative for sells)
        signed_qty = quantity if side == "buy" else -quantity

        # Case 1: Same direction – just scale position & adjust VWAP
        if self.position == 0 or (self.position > 0 and signed_qty > 0) or (self.position < 0 and signed_qty < 0):
            # New aggregate position after trade
            new_position = self.position + signed_qty

            # Update VWAP if we will still have an open position
            if new_position != 0:
                # Weighted average price calculation
                total_cost = self.avg_price * abs(self.position) + price * abs(signed_qty)
                self.avg_price = total_cost / abs(new_position)
            else:
                # Flat – reset avg_price
                self.avg_price = 0.0

            self.position = new_position
            return  # No realised P&L generated in this scenario

        # Case 2: Opposite direction – we are closing/reversing part or all of the existing position.
        # Determine how much of the trade quantity offsets the existing position
        offset_qty = min(abs(signed_qty), abs(self.position))

        # Realised P&L from the offset portion
        if self.position > 0:  # We were long, selling to close
            realised = (price - self.avg_price) * offset_qty
        else:  # We were short, buying to cover
            realised = (self.avg_price - price) * offset_qty

        self.realized_pnl += realised

        # Remaining quantity after offset determines if we flip direction
        remaining_qty = abs(signed_qty) - offset_qty

        if remaining_qty == 0:
            # Flat after this trade
            self.position = 0.0
            self.avg_price = 0.0
        else:
            # We have flipped to the opposite direction with *remaining_qty*
            self.position = (remaining_qty if side == "buy" else -remaining_qty)
            self.avg_price = price  # New position opened entirely at the trade price

    # ------------------------------------------------------------------
    # Misc / dunder helpers
    # ------------------------------------------------------------------
    def __str__(self):
        return (
            f"Positions(symbol={self.symbol}, position={self.position}, "
            f"avg_price={self.avg_price:.4f}, realized_pnl={self.realized_pnl:.2f})"
        )

## The primitive for a basic strategy implementation. This will be used by backtesting and live trading to run the strategy.
class Strategy:
    """A strategy class that will be inherited by all strategies."""
    def __init__(self, data_sources: dict[str, DataSource]):
        self.data_sources = data_sources
    
    def run_strategy(self, idx: int):
        """Run the strategy."""
        raise NotImplementedError("Subclasses must implement this method.")
    