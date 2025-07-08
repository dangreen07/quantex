# QuantEx

Welcome to **QuantEx**, a minimal yet powerful Python library for building, back-testing and documenting quantitative trading strategies.

QuantEx provides five key building blocks:

1. **Market Data Sources (`quantex.sources`)** – consistent interfaces for historical (back-testing) or live data feeds.
2. **Core Data Models (`quantex.models`)** – immutable market-data records and stateful helpers such as `Position` and `Portfolio` for tracking P&L.
3. **Strategy Harness (`quantex.strategy`)** – a thin, extendable base-class that ties data sources and portfolio management together.
4. **Event Bus & Execution Simulator (`quantex.engine`, `quantex.execution`)** – dispatches bars to the strategy, converts `Order`s into `Fill`s, and keeps the portfolio in sync.
5. **Backtest Runner (`quantex.backtest`)** – one-stop helper that wires everything together and returns results/metrics ready for analysis.

The package is intentionally lightweight; it aims to give you just the right abstractions so you can focus on research rather than plumbing.

---

## Installation

```bash
git clone https://github.com/dangreen07/quantex.git
cd quantex
poetry install
```

*(QuantEx is published for illustrative purposes only – adjust according to your own distribution method.)*

---

## Quick-Start

Get up and running with **QuantEx** in less than five minutes.

### 1. Prepare Some Data

QuantEx does **not** bundle market data. For a first test, create a tiny CSV named `prices.csv`:

```csv
timestamp,open,high,low,close,volume
2024-01-01 09:30:00,100,101,99,100.5,1000
2024-01-01 09:31:00,100.5,101,100,100.7,500
2024-01-01 09:32:00,100.7,100.8,100.2,100.3,750
2024-01-01 09:33:00,100.3,100.5,99.8,99.9,1200
2024-01-01 09:34:00,99.9,100.2,99.7,100.1,900
```

### 2. Create a Strategy

```python
from quantex.strategy import Strategy
from quantex.sources import CSVDataSource
from quantex.backtest import BacktestRunner

class BuyAndHold(Strategy):
    """Buys on the first bar and holds."""

    def run(self):
        if self.index == 0:
            self.buy("TEST", 100)

# --- Back-test Execution ---
if __name__ == "__main__":
    data_source = CSVDataSource("prices.csv", symbol="TEST")
    strategy = BuyAndHold(symbols=["TEST"], initial_cash=100_000)
    result = BacktestRunner(strategy, {"csv": data_source}).run()

    print("--- METRICS ---")
    print(f"Total Return: {result.metrics['total_return']:.2%}")
    print(f"Total Cash Return: {result.metrics['total_return']*100_000:.2f}")
    print("--- NAV ---")
    print(result.nav)
```

### 3. Run It

```bash
python my_strategy.py
```

You should see output similar to:

```
--- METRICS ---
Total Return: -0.04%
Total Cash Return: -40.00
--- NAV ---
2024-01-01 09:30:00    100000.0
2024-01-01 09:31:00    100020.0
2024-01-01 09:32:00     99980.0
2024-01-01 09:33:00     99940.0
2024-01-01 09:34:00     99960.0
Name: NAV, dtype: float64
```

---

## Next Steps

* Browse the [**Guides**](usage/strategy.md) for tutorials and patterns.
* Dive into the [**API Reference**](reference/quantex.models.md) for full docstrings.
* Contribute ideas or improvements on GitHub.

Happy trading 🐍💹

--- 