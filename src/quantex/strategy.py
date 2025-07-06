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
        return f"Trade(symbol={self.symbol}, price={self.price:.2f}, quantity={self.quantity:.1f})"

class Positions:
    """Tracks the position, trades, and P&L for a single symbol."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.position = 0
        self.trades = []
        self.average_price = 0.0  # average cost basis of current position
        self.realized_pnl = 0.0   # cumulative realized profit & loss
    
    def _apply_trade(self, quantity: float, price: float, timestamp: datetime):
        """Internal helper to process a trade.

        A positive quantity represents a buy (adding to / reducing a short position).
        A negative quantity represents a sell (reducing a long / adding to a short position).
        This method updates position size, average cost basis, realized P&L and records the trade.
        """
        prev_pos = self.position
        new_pos = prev_pos + quantity

        if prev_pos == 0:
            # Opening a brand-new position
            self.average_price = price
        elif prev_pos * quantity > 0:
            # Increasing position in the same direction (adding to long or short)
            total_size = abs(prev_pos) + abs(quantity)
            self.average_price = (
                (self.average_price * abs(prev_pos) + price * abs(quantity)) / total_size
            )
        else:
            # Reducing or flipping the position
            closing_size = min(abs(quantity), abs(prev_pos))
            sign_prev = 1 if prev_pos > 0 else -1  # +1 for long, -1 for short
            # Realized P&L for the closed portion
            self.realized_pnl += (price - self.average_price) * closing_size * sign_prev

            if new_pos == 0:
                # Fully closed the position
                self.average_price = 0.0
            elif abs(quantity) > abs(prev_pos):
                # Direction flip: set new cost basis to trade price
                self.average_price = price
                # Otherwise (partial reduction) keep existing average_price

        self.position = new_pos
        self.trades.append(Trade(self.symbol, price, quantity, timestamp))

    def buy(self, quantity: float, price: float, timestamp: datetime):
        """Buy quantity of the asset (may close short or open/increase long)."""
        self._apply_trade(quantity, price, timestamp)

    def sell(self, quantity: float, price: float, timestamp: datetime):
        """Sell quantity of the asset (may close long or open/increase short)."""
        self._apply_trade(-quantity, price, timestamp)

    def calculate_total_pnl(self, current_price: float) -> float:
        """Return total (realized + unrealized) PnL given the current market price."""
        unrealized = (current_price - self.average_price) * self.position
        return self.realized_pnl + unrealized

## The primitive for a basic strategy implementation. This will be used by backtesting and live trading to run the strategy.
class Strategy:
    """A strategy class that will be inherited by all strategies."""
    def __init__(self, data_sources: dict[str, DataSource]):
        self.data_sources = data_sources
    
    def run_strategy(self, idx: int):
        """Run the strategy."""
        raise NotImplementedError("Subclasses must implement this method.")
    