# Quick-Start

Get up and running with **QuantEx** in less than five minutes.

---

## 1. Install

Currently installing this library requires poetry

```bash
git clone https://github.com/dangreen07/quantex.git
cd quantex
poetry install
```

## 2. Prepare Some Data

QuantEx does **not** bundle market data. For this tutorial, create a simple CSV file named `prices.csv`.

```csv
# prices.csv
timestamp,open,high,low,close,volume
2024-01-01 09:30:00,100,101,99,100.5,1000
2024-01-01 09:31:00,100.5,101,100,100.7,500
2024-01-01 09:32:00,100.7,100.8,100.2,100.3,750
2024-01-01 09:33:00,100.3,100.5,99.8,99.9,1200
2024-01-01 09:34:00,99.9,100.2,99.7,100.1,900
```

> Any OHLCV CSV with the required headers works.

## 3. Create a Strategy

Create a file named `my_strategy.py`.

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

## 4. Run the Back-test

Execute your strategy from the terminal:

```bash
python my_strategy.py
```

You should see output similar to this:

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

That's it! You've successfully run a back-test.

---

### Where to Go Next

*   Check out [**Guides**](usage/strategy.md) for more tutorials.
*   Explore the [**API Reference**](reference/quantex.models.md) for per-class docstrings.
*   Contribute ideas or issues on GitHub. 