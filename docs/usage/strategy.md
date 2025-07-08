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
| `cash` | Current available cash (read-only shortcut for `portfolio.cash`). |
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

## Building a Strategy – End-to-End

Below is a **complete, minimal** example that you can copy-paste and run. It shows how to

1. Load price data from a Parquet file.
2. Implement a strategy that relies on a *lookback window*.
3. Run the back-test and inspect the results.

```python
from quantex.backtest import BacktestRunner
from quantex.sources import ParquetDataSource
from quantex.strategy import Strategy

class MeanReversion(Strategy):
    """Buys when price drops below the rolling 20-bar mean by 2×σ, sells on mean re-touch."""

    def run(self):
        lookback = 20
        # Aligned history (no NaNs) for *all* symbols
        hist = self.get_lookback_prices(lookback)
        price = hist["BTC-USD"].iloc[-1]
        mean = hist["BTC-USD"].mean()
        std = hist["BTC-USD"].std()

        # Entry
        if price < mean - 2 * std and self.positions["BTC-USD"].is_closed:
            self.buy("BTC-USD", 1)

        # Exit
        if not self.positions["BTC-USD"].is_closed and price >= mean:
            self.close_position("BTC-USD")

# 1. Market data source (Parquet is fastest)
ds = ParquetDataSource("btc.parquet", symbol="BTC-USD")

# 2. Instantiate the strategy (start with $10k cash)
strat = MeanReversion(symbols=["BTC-USD"], initial_cash=10_000)

# 3. Run the back-test
result = BacktestRunner(strat, {"btc": ds}).run()
print(result.metrics)
```

### Cash Safety Check

QuantEx enforces a *cash sufficiency rule* at execution time. A `ValueError` is raised if you try to buy more than your available cash:

```python
>>> strat.buy("BTC-USD", 1)  # costs $100 but we have $50
ValueError: Insufficient cash to execute buy order: required=100.00, available=50.00
```

No more accidental negative balances!

## Advanced Tips

* **Multi-Asset Trading** – Because the engine feeds *aligned* price rows, slicing `hist[["BTC", "ETH"]]` gives you a *synchronised* DataFrame ready for vectorised NumPy/Pandas operations.
* **Lagging Indicators** – Compute them on `self.price_history` inside `run()`; no look-ahead bias because the latest row corresponds to the current bar.
* **Position Sizing** – Use `self.cash` (or `self.portfolio.cash`) to implement Kelly-style sizing.
* **Debugging** – Insert `print(self.timestamp, self.portfolio)` to log cash & positions every bar.

---

### Planned Extensions

QuantEx aims to add:

1. **Pluggable Execution Handlers** (latency, partial fills)
2. **Built-in Risk Checks** (max drawdown, position limits)
3. **CLI Launcher** for one-liner back-tests: `quant backtest strategy.py btc.parquet`

Contributions welcome! Pull requests & feature requests are tracked on GitHub. 