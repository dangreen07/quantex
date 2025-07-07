# QuantEx

Welcome to **QuantEx**, a minimal yet powerful Python library for building, back-testing and documenting quantitative trading strategies.

QuantEx provides three key building blocks:

1. **Market Data Sources (`quantex.sources`)** – consistent interfaces for historical (back-testing) or live data feeds.
2. **Core Data Models (`quantex.models`)** – immutable market-data records and stateful helpers such as `Position` and `Portfolio` for tracking P&L.
3. **Strategy Harness (`quantex.strategy`)** – a thin, extendable base-class that ties data sources and portfolio management together.

The package is intentionally lightweight; it aims to give you just the right abstractions so you can focus on research rather than plumbing.

---

## Installation

```bash
pip install quantex  # or use Poetry / hatch / pip-tools as you prefer
```

*(QuantEx is published for illustrative purposes only – adjust according to your own distribution method.)*

---

## Quick Example

Below is a condensed example that showcases how the main pieces fit together:

```python
import pandas as pd
from quantex.sources import BacktestingDataSource
from quantex.strategy import Strategy

class CSVSource(BacktestingDataSource):
    def __init__(self, path):
        self._df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
        self.index = 0

    def __len__(self):
        return len(self._df)

    def get_current_bar(self):
        row = self._df.iloc[self.index]
        ts = self._df.index[self.index]
        return Bar(ts, row.open, row.high, row.low, row.close, row.volume, symbol="TEST")

    def get_lookback_data(self, lookback_period):
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1]

class MeanReversion(Strategy):
    def run(self):
        if self.index < 20:
            return  # not enough history
        window = self.data_sources["csv"].get_lookback_data(20)
        if window.close.iloc[-1] < window.close.mean() and self.positions["TEST"].is_closed:
            bar = self.data_sources["csv"].get_current_bar()
            self.positions["TEST"].buy(bar.close, bar.timestamp, quantity=100)
```

See the full example in the `examples/` section (coming soon) for a complete trading loop.

---

## Next Steps

* Browse the **API Reference** for full docstrings of each sub-module.
* Check out the test-driven examples in `quantex/tests/`.
* Contribute improvements or raise issues on GitHub!

Happy trading 🐍💹 