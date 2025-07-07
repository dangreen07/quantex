# Data Sources

QuantEx separates **market data ingestion** from strategy logic. All data providers implement the abstract `DataSource` interface.

| Class | Use-case |
|-------|----------|
| `CSVDataSource` | Local OHLCV CSV files. |
| `ParquetDataSource` | Columnar Parquet files (fast). |
| _(abstract)_ `BacktestingDataSource` | Contract for fixed-length sources. |

## CSV Example
```python
from quantex.sources import CSVDataSource

ds = CSVDataSource("prices.csv", symbol="AAPL")
print(len(ds))        # number of rows
print(ds.peek_timestamp())  # first timestamp
```

## Parquet Example
```python
from quantex.sources import ParquetDataSource

ds = ParquetDataSource("btc.parquet", symbol="BTC-USD")
bar = ds.get_current_bar()
print(bar.close)
```

### Required Columns
All OHLCV sources must provide: `timestamp`, `open`, `high`, `low`, `close`, `volume`.

### Custom Sources
Implement:
```python
class MySource(BacktestingDataSource):
    def __len__(self): ...
    def get_raw_data(self): ...  # pandas.DataFrame indexed by timestamp
    def get_current_bar(self): ...
    def get_lookback_data(self, n): ...
    def peek_timestamp(self): ...
```

Return types mirror the built-ins so the engine remains agnostic. 