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

QuantEx does **not** bundle market data. For this tutorial we'll create a tiny CSV with two columns:

```csv
# prices.csv
timestamp,open,high,low,close,volume
2024-01-01 09:30:00,100,101,99,100.5,1000
2024-01-01 09:31:00,100.5,101,100,100.7,500
...
```

> Any OHLCV CSV (with the header above) works – longer datasets make the back-test more interesting.

## 3. Build a Data Source

```python
import pandas as pd
from quantex.sources import BacktestingDataSource
from quantex.models import Bar

class CSVSource(BacktestingDataSource):
    def __init__(self, path: str, symbol="TEST"):
        self._df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
        self.index = 0
        self.symbol = symbol

    def __len__(self):
        return len(self._df)

    def get_current_bar(self):
        row = self._df.iloc[self.index]
        ts = self._df.index[self.index]
        return Bar(ts, row.open, row.high, row.low, row.close, row.volume, self.symbol)

    def get_lookback_data(self, lookback):
        start = max(0, self.index - lookback + 1)
        return self._df.iloc[start : self.index + 1]
```

## 4. Create a Strategy

```python
from quantex.strategy import Strategy
from quantex.models import Order

class BuyDip(Strategy):
    """Buy when price falls below 10-period moving average."""

    def run(self):
        if self.index < 10:
            return  # need history
        window = self.data_sources["csv"].get_lookback_data(10)
        ma10 = window.close.mean()
        bar = self.data_sources["csv"].get_current_bar()
        if bar.close < ma10 and self.positions["TEST"].is_closed:
            self.submit_order(Order("buy-1", "TEST", "buy", 100))
        elif bar.close > ma10 and self.positions["TEST"].is_long:
            self.submit_order(Order("sell-1", "TEST", "sell", 100))
```

## 5. Run the Back-test

```python
from quantex.backtest import BacktestRunner

src = CSVSource("prices.csv")
strat = BuyDip({"csv": src}, symbols=["TEST"], initial_cash=10_000)
res = BacktestRunner(strat, {"csv": src}).run()

print(res.metrics)
print(res.nav.tail())
```

That's it! You've executed a full strategy lifecycle using EventBus, ImmediateFillSimulator, and the data models.

---

### Where to Go Next

* Check **Key Concepts** for deeper architectural details
* Explore the **API Reference** for per-class docstrings
* Contribute ideas or issues on GitHub 