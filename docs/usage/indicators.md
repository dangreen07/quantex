# Indicators Guide

Quantex now includes a built-in technical indicator catalog for common stock-trading workflows and more advanced quantitative studies.

You can access the catalog in two ways:

- package level: [`quantex.indicators`](../../src/quantex/indicators.py)
- strategy level: `self.ta` from [`Strategy.__init__()`](../../src/quantex/strategy.py:40)

Every indicator returns NumPy arrays. To make those arrays time-aware during a backtest, register them with [`Strategy.Indicator()`](../../src/quantex/strategy.py:126).

## Quick start

```python
from quantex import Strategy, CSVDataSource


class MomentumTrendStrategy(Strategy):
    def init(self):
        self.add_data(CSVDataSource("data.csv"), "TEST")
        source = self.data["TEST"]

        self.sma_20 = self.Indicator(self.ta.sma(source.Close, 20))
        self.ema_50 = self.Indicator(self.ta.ema(source.Close, 50))
        self.rsi_14 = self.Indicator(self.ta.rsi(source.Close, 14))

        macd_line, macd_signal, macd_hist = self.ta.macd(source.Close)
        self.macd_line = self.Indicator(macd_line)
        self.macd_signal = self.Indicator(macd_signal)
        self.macd_hist = self.Indicator(macd_hist)

    def next(self):
        if len(self.sma_20) < 2 or len(self.macd_signal) < 2:
            return

        bullish_cross = self.macd_line[-2] <= self.macd_signal[-2] and self.macd_line[-1] > self.macd_signal[-1]
        if bullish_cross and self.rsi_14[-1] < 70:
            self.positions["TEST"].buy(quantity=0.5)
```

## Indicator categories

### Moving averages and trend filters

- `sma(values, period)`
- `ema(values, period)`
- `wma(values, period)`
- `dema(values, period)`
- `tema(values, period)`
- `kama(values, er_period=10, fast_period=2, slow_period=30)`
- `linear_regression_slope(values, period=20)`

### Momentum and oscillators

- `momentum(values, period=1)`
- `roc(values, period=1)`
- `rsi(values, period=14)`
- `stochastic_oscillator(high, low, close, k_period=14, d_period=3)`
- `cci(high, low, close, period=20)`
- `williams_r(high, low, close, period=14)`
- `macd(values, fast_period=12, slow_period=26, signal_period=9)`
- `trix(values, period=15)`
- `ultimate_oscillator(high, low, close, short_period=7, medium_period=14, long_period=28)`
- `fisher_transform(values, period=10)`

### Volatility and channel indicators

- `true_range(high, low, close)`
- `atr(high, low, close, period=14)`
- `volatility(values, period, ddof=0)`
- `bollinger_bands(values, period=20, std_dev=2.0)`
- `keltner_channels(high, low, close, ema_period=20, atr_period=10, multiplier=2.0)`
- `donchian_channels(high, low, period=20)`

### Volume and trend-strength indicators

- `obv(close, volume)`
- `mfi(high, low, close, volume, period=14)`
- `adx(high, low, close, period=14)`
- `aroon(high, low, period=25)`
- `vortex(high, low, close, period=14)`

### Advanced and research-oriented indicators

- `ichimoku_cloud(high, low, close, conversion_period=9, base_period=26, span_b_period=52, displacement=26)`
- `zscore(values, period=20)`
- `sharpe_ratio(values, period=20, risk_free_rate=0.0)`
- `sortino_ratio(values, period=20, target_return=0.0)`
- `hurst_exponent(values, period=100)`

## Multi-output indicators

Some indicators return several arrays. Register each returned series separately if you want time-aware access in `next()`.

### MACD

```python
macd_line, macd_signal, macd_hist = self.ta.macd(self.data["TEST"].Close)
self.macd_line = self.Indicator(macd_line)
self.macd_signal = self.Indicator(macd_signal)
self.macd_hist = self.Indicator(macd_hist)
```

### Bollinger Bands

```python
lower, middle, upper = self.ta.bollinger_bands(self.data["TEST"].Close, period=20)
self.bb_lower = self.Indicator(lower)
self.bb_middle = self.Indicator(middle)
self.bb_upper = self.Indicator(upper)
```

### ADX and directional indicators

```python
adx, plus_di, minus_di = self.ta.adx(
    self.data["TEST"].High,
    self.data["TEST"].Low,
    self.data["TEST"].Close,
)
self.adx = self.Indicator(adx)
self.plus_di = self.Indicator(plus_di)
self.minus_di = self.Indicator(minus_di)
```

### Ichimoku Cloud

```python
conversion, base, span_a, span_b, lagging = self.ta.ichimoku_cloud(
    self.data["TEST"].High,
    self.data["TEST"].Low,
    self.data["TEST"].Close,
)
```

## Time-aware behavior

[`Strategy.Indicator()`](../../src/quantex/strategy.py:126) converts a plain NumPy array into [`TimeNDArray`](../../src/quantex/helpers.py:7), which hides future values during the backtest loop.

That means:

- `indicator[-1]` is the latest visible value
- `indicator[-2]` is the prior visible value
- `indicator[-5:]` only exposes visible history

This keeps built-in indicators safe to use in the same way as custom arrays.

## Combining built-in and custom indicators

You can mix built-in indicators with your own NumPy calculations.

```python
close = self.data["TEST"].Close
returns = np.diff(close, prepend=close[0]) / close[0]

self.custom_signal = self.Indicator(returns.cumsum())
self.atr_14 = self.Indicator(
    self.ta.atr(self.data["TEST"].High, self.data["TEST"].Low, close, 14)
)
```

## Notes on inputs and outputs

- single-series indicators expect a one-dimensional array-like input
- OHLC-based indicators require same-length `high`, `low`, and `close` arrays
- volume indicators require `volume` with the same length as price data
- rolling indicators begin with `NaN` values until enough history exists

## Typical trading patterns

### Trend-following crossover

```python
fast = self.Indicator(self.ta.ema(self.data["TEST"].Close, 20))
slow = self.Indicator(self.ta.ema(self.data["TEST"].Close, 50))
```

### Mean reversion

```python
zscore = self.Indicator(self.ta.zscore(self.data["TEST"].Close, 20))
bb_lower, bb_mid, bb_upper = self.ta.bollinger_bands(self.data["TEST"].Close, 20)
```

### Breakout systems

```python
donchian_lower, donchian_mid, donchian_upper = self.ta.donchian_channels(
    self.data["TEST"].High,
    self.data["TEST"].Low,
    20,
)
```

### Regime filtering

```python
adx, plus_di, minus_di = self.ta.adx(
    self.data["TEST"].High,
    self.data["TEST"].Low,
    self.data["TEST"].Close,
    14,
)
hurst = self.ta.hurst_exponent(self.data["TEST"].Close, 100)
```

## Summary

Use built-in indicators in Quantex like this:

1. compute arrays with `self.ta` or `quantex.indicators`
2. wrap each series with [`Strategy.Indicator()`](../../src/quantex/strategy.py:126)
3. read only visible values inside [`Strategy.next()`](../../src/quantex/strategy.py:71)

For broader strategy design, see [Strategy guide](./strategy.md) and [Backtesting guide](./backtesting.md).
