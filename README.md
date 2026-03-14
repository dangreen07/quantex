# Quantex

Quantex is a Python library for writing trading strategies and replaying them on historical market data.

It gives you a small set of building blocks:

- [`Strategy`](src/quantex/strategy.py:9) for your trading rules
- [`DataSource`](src/quantex/datasource.py:6) plus [`CSVDataSource`](src/quantex/datasource.py:194) and [`ParquetDataSource`](src/quantex/datasource.py:228) for OHLCV market data
- [`Broker`](src/quantex/broker.py:113) objects, created automatically per symbol, for order placement and position state
- [`SimpleBacktester`](src/quantex/backtester.py:356) for simulation and parameter search
- [`BacktestReport`](src/quantex/backtester.py:188) for results, equity history, and summary statistics

The project is intentionally small. It does not try to be a full research platform, portfolio database, or live-trading engine. Instead, it focuses on a straightforward workflow: load historical bars, define strategy logic, simulate orders, and inspect the results.

## Installation

Quantex requires Python 3.10 or newer and is published as [`quantex`](pyproject.toml).

```bash
pip install quantex
```

## What the library actually does

At runtime, a typical Quantex workflow looks like this:

1. Create a strategy class by inheriting from [`Strategy`](src/quantex/strategy.py:9).
2. In [`Strategy.init()`](src/quantex/strategy.py:52), attach one or more data sources with [`Strategy.add_data()`](src/quantex/strategy.py:98).
3. Still in [`Strategy.init()`](src/quantex/strategy.py:52), build indicator arrays with [`Strategy.Indicator()`](src/quantex/strategy.py:126).
4. In [`Strategy.next()`](src/quantex/strategy.py:71), read the current bar through properties such as [`DataSource.COpen`](src/quantex/datasource.py:145) and [`DataSource.CClose`](src/quantex/datasource.py:175), then place orders through the broker stored in [`Strategy.positions`](src/quantex/strategy.py:47).
5. Run the strategy with [`SimpleBacktester.run()`](src/quantex/backtester.py:414).
6. Inspect the returned [`BacktestReport`](src/quantex/backtester.py:188), including [`BacktestReport.total_return`](src/quantex/backtester.py:227), [`BacktestReport.periods_per_year`](src/quantex/backtester.py:214), [`BacktestReport.plot()`](src/quantex/backtester.py:252), and the printable summary from [`BacktestReport.__str__()`](src/quantex/backtester.py:299).

## Usage

The example below assumes no prior knowledge of the library.

### 1. Prepare a data file

Quantex expects historical bar data with these columns:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

The first column in a CSV file is typically the timestamp index used by [`CSVDataSource`](src/quantex/datasource.py:215).

Example:

```csv
Date,Open,High,Low,Close,Volume
2024-01-01,100,101,99,100.5,1500
2024-01-02,100.5,102,100,101.8,1700
2024-01-03,101.8,103,101,102.6,1650
```

### 2. Write a strategy

```python
from quantex import Strategy, CSVDataSource, SimpleBacktester
import pandas as pd


class MovingAverageCross(Strategy):
    def __init__(self, fast_period=5, slow_period=20):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def init(self):
        data = CSVDataSource("data.csv")
        self.add_data(data, "TEST")

        close = self.data["TEST"].Close
        self.fast_ma = self.Indicator(
            pd.Series(close).rolling(window=self.fast_period).mean().to_numpy()
        )
        self.slow_ma = self.Indicator(
            pd.Series(close).rolling(window=self.slow_period).mean().to_numpy()
        )

    def next(self):
        if len(self.fast_ma) < 2 or len(self.slow_ma) < 2:
            return

        broker = self.positions["TEST"]

        crossed_up = self.fast_ma[-2] <= self.slow_ma[-2] and self.fast_ma[-1] > self.slow_ma[-1]
        crossed_down = self.fast_ma[-2] >= self.slow_ma[-2] and self.fast_ma[-1] < self.slow_ma[-1]

        if crossed_up and broker.is_closed():
            broker.buy(quantity=1.0)
        elif crossed_down and broker.is_long():
            broker.close()
```

### 3. Run a backtest

```python
strategy = MovingAverageCross(fast_period=5, slow_period=20)
backtester = SimpleBacktester(strategy, cash=10_000)
report = backtester.run()

print(report)
report.plot()
```

### 4. Understand what happens during the backtest

When you call [`SimpleBacktester.run()`](src/quantex/backtester.py:414):

- the backtester deep-copies your strategy in [`SimpleBacktester.__init__()`](src/quantex/backtester.py:380)
- starting cash is split evenly across all attached symbols in [`SimpleBacktester.run()`](src/quantex/backtester.py:439)
- each data source advances one bar at a time by updating [`DataSource.current_index`](src/quantex/datasource.py:62)
- each symbol's broker processes pending orders through [`Broker._iterate()`](src/quantex/broker.py:483)
- market orders execute at the current bar's open price through [`DataSource.COpen`](src/quantex/datasource.py:145)
- equity is tracked into the final [`BacktestReport.PnlRecord`](src/quantex/backtester.py:205)

### 5. Understand order sizing

Order sizing in Quantex is simple but important:

- [`Broker.buy()`](src/quantex/broker.py:159) treats `quantity` as a fraction of available cash unless you pass `amount`
- [`Broker.sell()`](src/quantex/broker.py:235) uses the same sizing calculation and can open or increase a short position
- [`Broker.close()`](src/quantex/broker.py:307) places a market order that offsets the current position

Because of this design, `buy(0.5)` means “use roughly half of the broker cash for this symbol”, not “buy half a share”.

## Core concepts

### Strategy

Your strategy is a class that implements two methods:

- [`Strategy.init()`](src/quantex/strategy.py:52): called once before the bar-by-bar loop starts
- [`Strategy.next()`](src/quantex/strategy.py:71): called on every step of the simulation

### Data sources

[`DataSource`](src/quantex/datasource.py:6) wraps a pandas DataFrame and exposes both full history and current-bar values.

- historical arrays: [`DataSource.Open`](src/quantex/datasource.py:90), [`DataSource.High`](src/quantex/datasource.py:101), [`DataSource.Low`](src/quantex/datasource.py:112), [`DataSource.Close`](src/quantex/datasource.py:123), [`DataSource.Volume`](src/quantex/datasource.py:134)
- current bar values: [`DataSource.COpen`](src/quantex/datasource.py:145), [`DataSource.CHigh`](src/quantex/datasource.py:155), [`DataSource.CLow`](src/quantex/datasource.py:165), [`DataSource.CClose`](src/quantex/datasource.py:175), [`DataSource.CVolume`](src/quantex/datasource.py:185)

### Indicators

[`Strategy.Indicator()`](src/quantex/strategy.py:126) wraps a NumPy array in [`TimeNDArray`](src/quantex/helpers.py:7), which only exposes data up to the current bar. That helps you avoid accidentally reading future values if you index indicators correctly.

### Broker and orders

Each call to [`Strategy.add_data()`](src/quantex/strategy.py:98) also creates a [`Broker`](src/quantex/broker.py:113) for that symbol.

Supported order behavior in the current codebase:

- market orders and limit orders via [`OrderType`](src/quantex/broker.py:24)
- pending, active, and complete order states via [`OrderStatus`](src/quantex/broker.py:37)
- optional stop-loss and take-profit triggers stored on [`Order`](src/quantex/broker.py:52)
- percentage or cash commissions via [`CommissionType`](src/quantex/enums.py:4)

## Optimization

[`SimpleBacktester.optimize()`](src/quantex/backtester.py:485) runs a grid search over every parameter combination you provide.

[`SimpleBacktester.optimize_parallel()`](src/quantex/backtester.py:659) does the same work in multiple processes, then re-runs the best parameter set locally to produce a full [`BacktestReport`](src/quantex/backtester.py:188).

Minimal example:

```python
params = {
    "fast_period": range(5, 11),
    "slow_period": range(15, 31, 5),
}

best_params, best_report, results = backtester.optimize(
    params,
    constraint=lambda p: p["fast_period"] < p["slow_period"],
)

print(best_params)
print(best_report)
print(results.head())
```

## Documentation

Project documentation is built with MkDocs. The main docs entry point is [`docs/index.md`](docs/index.md), and usage guides live under [`docs/usage/`](docs/usage/).

Published documentation: [https://dangreen07.github.io/quantex/](https://dangreen07.github.io/quantex/)

## Development

The package metadata and dependencies are defined in [`pyproject.toml`](pyproject.toml).

Typical local workflow:

```bash
poetry install
poetry run pytest
poetry run pre-commit install
```

Tests in [`tests/test_backtester.py`](tests/test_backtester.py), [`tests/test_broker.py`](tests/test_broker.py), [`tests/test_datasource.py`](tests/test_datasource.py), and [`tests/test_strategy.py`](tests/test_strategy.py) are useful references for the library's current behavior.

## License

See [`LICENSE.md`](LICENSE.md).

