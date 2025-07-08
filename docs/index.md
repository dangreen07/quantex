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

## Next Steps

* Browse the [**API Reference**](reference/quantex.models.md) for full docstrings of each sub-module.
* Contribute improvements or raise issues on GitHub!

Happy trading 🐍💹 