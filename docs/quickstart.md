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
    """A simple strategy that buys on the first bar and holds."""

    def run(self):
        # On the first data bar, buy 100 units and then do nothing.
        if self.index == 0:
            bar = self.data_sources["csv"].get_current_bar()
            self.buy("TEST", 100, bar.timestamp)

# --- Back-test Execution ---
if __name__ == "__main__":
    # 1. Instantiate your data source
    data_source = CSVDataSource("prices.csv", symbol="TEST")
    
    # 2. Instantiate your strategy
    strategy = BuyAndHold(
        data_sources={"csv": data_source}, 
        symbols=["TEST"], 
        initial_cash=100_000
    )

    # 3. Run the back-test
    result = BacktestRunner(strategy, {"csv": data_source}).run()

    # 4. Print the results
    print("--- METRICS ---")
    print(result.metrics)
    print("\n--- FINAL NAV ---")
    print(result.nav.tail())
```

## 4. Run the Back-test

Execute your strategy from the terminal:

```bash
python my_strategy.py
```

You should see output similar to this:

```
--- METRICS ---
{'total_return': 0.0009950248756218988}

--- FINAL NAV ---
timestamp
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

*   Check out **Key Concepts** for deeper architectural details.
*   Explore the **API Reference** for per-class docstrings.
*   Contribute ideas or issues on GitHub. 