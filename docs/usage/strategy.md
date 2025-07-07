# Writing Strategies

QuantEx strategies inherit from `quantex.strategy.Strategy`.

```python
from quantex.strategy import Strategy

class SMACross(Strategy):
    def precompute_signals(self, prices):
        # vectorised indicator calculation – runs **once**
        self.signals = prices.rolling(10).mean() > prices.rolling(20).mean()

    def run(self):
        # use helpers injected by the engine
        price = self.get_price("BTC-USD")

        # custom logic – here a simple SMAs condition
        if self.signals.loc[self.timestamp, "BTC-USD"]:
            if self.positions["BTC-USD"].is_closed:
                self.buy("BTC-USD", 1)
        else:
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

## Lifecycle

1. `precompute_signals(prices_df)` – optional, called once before the event loop.
2. Each bar:
   a. Engine injects timestamp & prices.
   b. `run()` executes.
   c. Orders are routed/filled.

## Minimal Template

```python
class MyStrategy(Strategy):
    def run(self):
        if self.index == 0:
            self.buy("ETH-USD", 2)
```

That's all you need – no manual timestamps or boilerplate. 