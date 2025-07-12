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
| `metrics` | `dict` | Performance metrics (see below). |

## Performance Metrics
The `metrics` dictionary contains key performance statistics:

| Metric | Description |
|--------|-------------|
| `total_return` | Total return as a percentage (e.g., 0.15 = 15% return) |
| `max_drawdown` | Maximum drawdown as a percentage (e.g., -0.25 = 25% drawdown) |
| `sharpe_ratio` | Annualized Sharpe ratio |

```python
# Example accessing metrics
result = BacktestRunner(strategy, data_sources).run()
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.4f}")
```

> Extend by post-processing the `nav` or `fills` lists. 