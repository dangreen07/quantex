# Backtesting Engine

At runtime the **EventBus** orchestrates the data flow between:
1. Data sources → provide market data.
2. Strategy → consumes data & produces orders.
3. Execution simulator → turns orders into fills.

The `BacktestRunner` wires these pieces and produces a `BacktestResult`.

```python
from quantex.backtest import BacktestRunner
from quantex.sources import ParquetDataSource
from my_strategy import SMACross

ds = ParquetDataSource("btc.parquet", symbol="BTC-USD")
strategy = SMACross(symbols=["BTC-USD"], initial_cash=10_000)
result = BacktestRunner(strategy, {"btc": ds}).run()
print(result.metrics)
```

## BacktestResult Fields
| Field | Type | Description |
|-------|------|-------------|
| `nav` | `pd.Series` | Net-asset-value over time. |
| `orders` | `list[Order]` | All generated orders. |
| `fills` | `list[Fill]` | Executed trades. |
| `metrics` | `dict` | Basic summary stats. |

> Extend by post-processing the `nav` or `fills` lists. 