# Data Sources Guide

This guide explains how market data enters Quantex and how strategy code reads it during a backtest.

If you are new to the library, start with this mental model:

- a [`DataSource`](../../src/quantex/datasource.py:6) wraps historical OHLCV data
- a strategy reads that data one bar at a time
- the backtester moves the visible point forward during [`SimpleBacktester.run()`](../../src/quantex/backtester.py:414)

## The three data source classes

The current codebase defines:

- [`DataSource`](../../src/quantex/datasource.py:6) for in-memory pandas DataFrames
- [`CSVDataSource`](../../src/quantex/datasource.py:194) for CSV files
- [`ParquetDataSource`](../../src/quantex/datasource.py:228) for Parquet files

## Required data format

Every data source must provide these columns:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

This requirement is enforced in [`DataSource.__init__()`](../../src/quantex/datasource.py:60).

The index should be time-based if you want meaningful annualization in backtest metrics, because [`BacktestReport.periods_per_year`](../../src/quantex/backtester.py:214) infers data frequency from the index.

## Loading CSV data

The simplest entry point is [`CSVDataSource`](../../src/quantex/datasource.py:194).

```python
from quantex import CSVDataSource

source = CSVDataSource("data.csv")

print(len(source))
print(source.Index[0])
print(source.Index[-1])
```

[`CSVDataSource.__init__()`](../../src/quantex/datasource.py:215) reads the file with pandas using the first column as the parsed index by default.

Example CSV layout:

```csv
Date,Open,High,Low,Close,Volume
2024-01-01,100,101,99,100.5,1500
2024-01-02,100.5,102,100,101.8,1700
2024-01-03,101.8,103,101,102.6,1650
```

## Loading Parquet data

Use [`ParquetDataSource`](../../src/quantex/datasource.py:228) when your historical data is stored as Parquet.

```python
from quantex import ParquetDataSource

source = ParquetDataSource("data.parquet")
print(len(source))
```

Unlike some earlier documentation, you should pass a Parquet file path here, not a CSV path. [`ParquetDataSource.__init__()`](../../src/quantex/datasource.py:250) calls pandas `read_parquet` directly.

## Creating a data source from a pandas DataFrame

If you already have data in memory, create a plain [`DataSource`](../../src/quantex/datasource.py:6).

```python
import pandas as pd
from quantex import DataSource

df = pd.DataFrame(
    {
        "Open": [100, 101, 102],
        "High": [101, 102, 103],
        "Low": [99, 100, 101],
        "Close": [100.5, 101.8, 102.6],
        "Volume": [1500, 1700, 1650],
    },
    index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
)

source = DataSource(df)
```

## Train/test split support

[`DataSource.__init__()`](../../src/quantex/datasource.py:32) has a built-in 80/20 split option.

```python
train_source = DataSource(df, train_test_split=True, mode="train")
test_source = DataSource(df, train_test_split=True, mode="test")
```

This is simple dataset slicing inside the constructor, as tested in [`tests/test_datasource.py`](../../tests/test_datasource.py).

## What you can read from a data source

### Historical arrays

The following properties expose the visible history up to the current step:

- [`DataSource.Open`](../../src/quantex/datasource.py:90)
- [`DataSource.High`](../../src/quantex/datasource.py:101)
- [`DataSource.Low`](../../src/quantex/datasource.py:112)
- [`DataSource.Close`](../../src/quantex/datasource.py:123)
- [`DataSource.Volume`](../../src/quantex/datasource.py:134)

Example:

```python
recent_closes = source.Close[-20:]
latest_visible_high = source.High[-1]
```

### Current-bar values

The following properties expose the single current bar:

- [`DataSource.COpen`](../../src/quantex/datasource.py:145)
- [`DataSource.CHigh`](../../src/quantex/datasource.py:155)
- [`DataSource.CLow`](../../src/quantex/datasource.py:165)
- [`DataSource.CClose`](../../src/quantex/datasource.py:175)
- [`DataSource.CVolume`](../../src/quantex/datasource.py:185)

Example:

```python
current_open = source.COpen
current_close = source.CClose
```

### Index access

The full index is available through [`DataSource.Index`](../../src/quantex/datasource.py:80).

```python
timestamp = source.Index[source.current_index]
```

## How data becomes visible during a backtest

This is one of the most important pieces of the library.

During [`SimpleBacktester.run()`](../../src/quantex/backtester.py:414), each source receives a new [`DataSource.current_index`](../../src/quantex/datasource.py:62) on every step.

That means:

- [`DataSource.CClose`](../../src/quantex/datasource.py:175) points to the current bar
- [`DataSource.Close`](../../src/quantex/datasource.py:123) includes only data up to that bar
- future bars are not exposed through the normal visible-history properties

This is why strategy code can safely use slices like `self.data["TEST"].Close[-20:]` during a backtest.

## Using data sources inside a strategy

```python
from quantex import Strategy, CSVDataSource


class MyStrategy(Strategy):
    def init(self):
        self.add_data(CSVDataSource("eurusd.csv"), "EURUSD")

    def next(self):
        price_now = self.data["EURUSD"].CClose
        recent_prices = self.data["EURUSD"].Close[-10:]

        if price_now > recent_prices.mean():
            self.positions["EURUSD"].buy(quantity=0.1)
```

[`Strategy.add_data()`](../../src/quantex/strategy.py:98) stores the source in [`Strategy.data`](../../src/quantex/strategy.py:48) and creates a matching [`Broker`](../../src/quantex/broker.py:113) in [`Strategy.positions`](../../src/quantex/strategy.py:47).

## Multiple data sources in one strategy

### Multiple symbols

```python
def init(self):
    self.add_data(CSVDataSource("eurusd.csv"), "EURUSD")
    self.add_data(CSVDataSource("gbpusd.csv"), "GBPUSD")

def next(self):
    if self.data["EURUSD"].CClose > self.data["GBPUSD"].CClose:
        self.positions["EURUSD"].buy(quantity=0.1)
```

### Multiple timeframes

```python
def init(self):
    self.add_data(CSVDataSource("eurusd_m1.csv"), "EURUSD_M1")
    self.add_data(CSVDataSource("eurusd_h1.csv"), "EURUSD_H1")

def next(self):
    fast_price = self.data["EURUSD_M1"].CClose
    slower_context = self.data["EURUSD_H1"].CClose
```

When you attach multiple sources, remember that [`SimpleBacktester.run()`](../../src/quantex/backtester.py:443) splits starting cash evenly across the brokers created for those sources.

## Data validation tips

Before backtesting, check:

1. required columns exist
2. index ordering makes sense
3. numeric values are numeric
4. timestamps are what you expect

Example:

```python
def validate_dataframe(df):
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"missing columns: {missing}")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    return df
```

## Limitations to be aware of

The current data source API does **not** include:

- a `from_dataframe` convenience constructor on [`CSVDataSource`](../../src/quantex/datasource.py:194)
- built-in resampling helpers
- built-in synchronization utilities for mismatched timelines

If you need those features, preprocess the pandas DataFrame first and then pass the result into [`DataSource`](../../src/quantex/datasource.py:6).

## Minimal preprocessing workflow

```python
import pandas as pd
from quantex import DataSource


df = pd.read_csv("raw.csv", index_col=0, parse_dates=[0])
df = df.sort_index()
df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

source = DataSource(df)
```

## Summary

Use:

- [`CSVDataSource`](../../src/quantex/datasource.py:194) for CSV files
- [`ParquetDataSource`](../../src/quantex/datasource.py:228) for Parquet files
- [`DataSource`](../../src/quantex/datasource.py:6) for prepared pandas DataFrames

For next steps, see [Strategy guide](./strategy.md) and [Backtesting guide](./backtesting.md).

