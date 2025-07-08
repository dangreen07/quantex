# Writing Strategies

QuantEx strategies inherit from `quantex.strategy.Strategy`.

```python
from quantex.strategy import Strategy

class SMACross(Strategy):
    def run(self):
        # Access last price fast
        price = self.get_price("BTC-USD")

        # Example: very naive SMA crossover using last N cached values you
        # maintain yourself (live-safe – no full-history look-ahead).
        if not hasattr(self, "_prices"):
            self._prices = []
        self._prices.append(price)
        window_fast, window_slow = 10, 20
        if len(self._prices) >= window_slow:
            sma_fast = sum(self._prices[-window_fast:]) / window_fast
            sma_slow = sum(self._prices[-window_slow:]) / window_slow
            if sma_fast > sma_slow and self.positions["BTC-USD"].is_closed:
                self.buy("BTC-USD", 1)
            elif sma_fast < sma_slow:
                self.close_position("BTC-USD")
```

## Helper Methods

| Method | Purpose |
|--------|---------|
| `buy(symbol, qty, limit_price=None)` | Places a buy order. |
| `sell(symbol, qty, limit_price=None)` | Places a sell order. |
| `close_position(symbol)` | Market-closes any open position. |
| `get_price(symbol)` | O(1) latest price lookup via NumPy. |
| `prices` | Dict of symbol→price for current bar. |
| `price_history` | **DataFrame** of aligned prices for *all* symbols up to the current bar. |
| `get_lookback_prices(n)` | Returns the last *n* aligned rows (auto-truncated if history is shorter). |

> **No NaNs by design** – Because the engine processes only timestamps that are present **in all data sources**, every row returned by `price_history` or `get_lookback_prices()` contains a value for each symbol.  You can safely rely on these helper methods being free of missing data without additional forward-filling.

## Lifecycle

Each bar the engine:
1. Injects timestamp & prices.
2. Calls `run()`.
3. Executes queued orders and advances the strategy index.

## Minimal Template

```python
class MyStrategy(Strategy):
    def run(self):
        if self.index == 0:
            self.buy("ETH-USD", 2)
```

That's all you need – no manual timestamps or boilerplate. 