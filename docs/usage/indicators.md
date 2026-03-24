# Indicators Guide

Quantex provides a comprehensive library of technical indicators for common stock-trading workflows and advanced quantitative analysis. This guide covers all available indicators, their parameters, and practical examples for building trading strategies.

## Overview

The indicator catalog is accessible in two ways:

- **Package level**: Import directly from [`quantex.indicators`](../../src/quantex/indicators.py) for standalone calculations
- **Strategy level**: Use `self.ta` from [`Strategy.__init__()`](../../src/quantex/strategy.py:40) for time-aware backtesting

All indicators return NumPy arrays. To prevent look-ahead bias during backtesting, wrap arrays with [`Strategy.Indicator()`](../../src/quantex/strategy.py:126) to create time-aware [`TimeNDArray`](../../src/quantex/helpers.py:7) objects that hide future values.

## Quick Start

### Basic Usage

Data sources are added inside the `init()` method:

```python
from quantex import Strategy, CSVDataSource


class MomentumTrendStrategy(Strategy):
    def init(self):
        self.add_data(CSVDataSource("data.csv"), "TEST")
        source = self.data["TEST"]

        # Simple moving averages
        self.sma_20 = self.Indicator(self.ta.sma(source.Close, 20))
        self.ema_50 = self.Indicator(self.ta.ema(source.Close, 50))

        # Momentum oscillator
        self.rsi_14 = self.Indicator(self.ta.rsi(source.Close, 14))

        # MACD with all three components
        macd_line, macd_signal, macd_hist = self.ta.macd(source.Close)
        self.macd_line = self.Indicator(macd_line)
        self.macd_signal = self.Indicator(macd_signal)
        self.macd_hist = self.Indicator(macd_hist)

    def next(self):
        # Wait for indicators to warm up
        if len(self.sma_20) < 2 or len(self.macd_signal) < 2:
            return

        # Golden cross with RSI filter
        bullish_cross = self.macd_line[-2] <= self.macd_signal[-2] and self.macd_line[-1] > self.macd_signal[-1]
        if bullish_cross and self.rsi_14[-1] < 70:
            self.positions["TEST"].buy(quantity=0.5)
```

### Package-Level Usage

You can also use indicators directly without a strategy context:

```python
import numpy as np
from quantex import indicators

# Standalone indicator calculations
close_prices = np.array([100, 102, 101, 104, 103, 107, 109, 108, 112, 115])
sma_5 = indicators.sma(close_prices, 5)
rsi_14 = indicators.rsi(close_prices, 14)

# Multi-output indicators
lower, middle, upper = indicators.bollinger_bands(close_prices, 20, 2.0)
```

## Indicator Categories

### Moving Averages

Moving averages smooth price data to identify trends and filter noise.

| Function | Description | Parameters |
|----------|-------------|------------|
| `sma(values, period)` | Simple Moving Average - equal-weighted average over the period | `period`: Window size (default: varies) |
| `ema(values, period)` | Exponential Moving Average - exponentially weighted with newest values emphasized | `period`: Smoothing period |
| `wma(values, period)` | Weighted Moving Average - linearly weighted from oldest (1) to newest (period) | `period`: Window size |
| `dema(values, period)` | Double Exponential Moving Average - EMA of EMA for reduced lag | `period`: Base period |
| `tema(values, period)` | Triple Exponential Moving Average - EMA of EMA of EMA | `period`: Base period |
| `kama(values, er_period, fast_period, slow_period)` | Kaufman Adaptive Moving Average - adjusts speed based on market volatility | `er_period`: Efficiency ratio period (default: 10), `fast_period`: 2, `slow_period`: 30 |

**Example: Moving Average Crossover System**

```python
class EMACrossoverStrategy(Strategy):
    def init(self):
        close = self.data["TEST"].Close
        # Fast and slow EMAs for trend detection
        self.ema_fast = self.Indicator(self.ta.ema(close, 9))
        self.ema_slow = self.Indicator(self.ta.ema(close, 21))
        
        # KAMA for adaptive trend following
        self.kama = self.Indicator(self.ta.kama(close, 10, 2, 30))

    def next(self):
        if len(self.ema_fast) < 2:
            return
        
        # Golden cross (bullish signal)
        if self.ema_fast[-1] > self.ema_slow[-1] and self.ema_fast[-2] <= self.ema_slow[-2]:
            self.positions["TEST"].buy()
        
        # Death cross (bearish signal)
        elif self.ema_fast[-1] < self.ema_slow[-1] and self.ema_fast[-2] >= self.ema_slow[-2]:
            self.positions["TEST"].close()


strat = EMACrossoverStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Momentum and Oscillators

Momentum indicators measure the rate of price change and identify overbought/oversold conditions.

| Function | Description | Parameters |
|----------|-------------|------------|
| `momentum(values, period)` | Price change over the period | `period`: Lookback (default: 1) |
| `roc(values, period)` | Rate of Change as percentage | `period`: Lookback (default: 1) |
| `rsi(values, period)` | Relative Strength Index (0-100) | `period`: RSI period (default: 14) |
| `stochastic_oscillator(high, low, close, k_period, d_period)` | Fast (%K) and slow (%D) stochastic | `k_period`: 14, `d_period`: 3 |
| `cci(high, low, close, period)` | Commodity Channel Index | `period`: 20 |
| `williams_r(high, low, close, period)` | Williams %R (0 to -100) | `period`: 14 |
| `macd(values, fast_period, slow_period, signal_period)` | MACD line, signal, histogram | `fast`: 12, `slow`: 26, `signal`: 9 |
| `trix(values, period)` | Triple smooth exponential moving average derivative | `period`: 15 |
| `ultimate_oscillator(high, low, close, short, medium, long)` | Weighted average of three timeframes | `short`: 7, `medium`: 14, `long`: 28 |
| `fisher_transform(values, period)` | Transform to Gaussian normal distribution | `period`: 10 |

**Example: RSI-Based Mean Reversion Strategy**

```python
class RSIMeanReversionStrategy(Strategy):
    def init(self):
        close = self.data["TEST"].Close
        self.rsi_14 = self.Indicator(self.ta.rsi(close, 14))
        
    def next(self):
        if len(self.rsi_14) < 1:
            return
        
        # Oversold - potential long entry
        if self.rsi_14[-1] < 30 and not self.positions["TEST"].is_long:
            self.positions["TEST"].buy()
        
        # Overbought - take profits
        elif self.rsi_14[-1] > 70 and self.positions["TEST"].is_long:
            self.positions["TEST"].close()


strat = RSIMeanReversionStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

**Example: Stochastic Oscillator Trading**

```python
class StochasticStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        k, d = self.ta.stochastic_oscillator(ohlc.High, ohlc.Low, ohlc.Close, 14, 3)
        self.stoch_k = self.Indicator(k)
        self.stoch_d = self.Indicator(d)

    def next(self):
        if len(self.stoch_k) < 2:
            return
        
        # %K crosses above %D in oversold territory
        if self.stoch_k[-1] > self.stoch_d[-1] and self.stoch_k[-2] <= self.stoch_d[-2]:
            if self.stoch_k[-1] < 20:
                self.positions["TEST"].buy()
        
        # %K crosses below %D in overbought territory
        elif self.stoch_k[-1] < self.stoch_d[-1] and self.stoch_k[-2] >= self.stoch_d[-2]:
            if self.stoch_k[-1] > 80:
                self.positions["TEST"].close()


strat = StochasticStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Volatility Indicators

Volatility measures the degree of variation in prices, useful for risk management and breakout detection.

| Function | Description | Parameters |
|----------|-------------|------------|
| `true_range(high, low, close)` | Maximum of: H-L, |H-PC|, |L-PC| | Requires OHLC |
| `atr(high, low, close, period)` | Average True Range - smoothed volatility | `period`: 14 |
| `volatility(values, period, ddof)` | Rolling standard deviation of returns | `period`: lookback, `ddof`: 0 |
| `bollinger_bands(values, period, std_dev)` | Lower, middle, upper bands | `period`: 20, `std_dev`: 2.0 |
| `keltner_channels(high, low, close, ema_period, atr_period, multiplier)` | Channels based on EMA and ATR | `ema_period`: 20, `atr_period`: 10, `multiplier`: 2.0 |
| `donchian_channels(high, low, period)` | Upper, middle, lower from highest/lowest | `period`: 20 |

**Example: Bollinger Band Mean Reversion**

```python
class BollingerReversionStrategy(Strategy):
    def init(self):
        close = self.data["TEST"].Close
        lower, middle, upper = self.ta.bollinger_bands(close, 20, 2.0)
        self.bb_lower = self.Indicator(lower)
        self.bb_middle = self.Indicator(middle)
        self.bb_upper = self.Indicator(upper)

    def next(self):
        if len(self.bb_lower) < 1:
            return
        
        price = self.data["TEST"].Close[-1]
        
        # Price touches lower band - potential long
        if price <= self.bb_lower[-1] and not self.positions["TEST"].is_long:
            self.positions["TEST"].buy()
        
        # Price touches upper band - potential short
        elif price >= self.bb_upper[-1] and not self.positions["TEST"].is_short:
            self.positions["TEST"].sell()


strat = BollingerReversionStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

**Example: ATR-Based Position Sizing**

```python
class ATRPositionSizingStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        self.atr_14 = self.Indicator(self.ta.atr(ohlc.High, ohlc.Low, ohlc.Close, 14))

    def next(self):
        if len(self.atr_14) < 1:
            return
        
        # Use ATR for dynamic stop loss
        atr_value = self.atr_14[-1]
        current_price = self.data["TEST"].Close[-1]
        
        # Position size based on 2% risk per trade
        risk_amount = self.portfolio.cash * 0.02
        position_size = risk_amount / atr_value
        
        if self.should_buy() and not self.positions["TEST"].is_long:
            self.positions["TEST"].buy(quantity=position_size)
            # Set stop loss at 2 ATRs below entry
            self.stop_loss = current_price - (2 * atr_value)


strat = ATRPositionSizingStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Volume and Trend Strength Indicators

These indicators incorporate volume data and measure trend strength.

| Function | Description | Parameters |
|----------|-------------|------------|
| `obv(close, volume)` | On-Balance Volume - cumulative volume flow | Requires same-length arrays |
| `mfi(high, low, close, volume, period)` | Money Flow Index (0-100) | `period`: 14 |
| `adx(high, low, close, period)` | Average Directional Index and DI+/- | `period`: 14 |
| `aroon(high, low, period)` | AROON up, down, and oscillator | `period`: 25 |
| `vortex(high, low, close, period)` | VI+ and VI- trend indicators | `period`: 14 |
| `vwap(high, low, close, volume)` | Volume Weighted Average Price | Requires volume |

**Example: ADX Trend Strength Filter**

```python
class ADXTrendStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        adx, plus_di, minus_di = self.ta.adx(ohlc.High, ohlc.Low, ohlc.Close, 14)
        self.adx = self.Indicator(adx)
        self.plus_di = self.Indicator(plus_di)
        self.minus_di = self.Indicator(minus_di)

    def next(self):
        if len(self.adx) < 1:
            return
        
        # Strong trend filter (ADX > 25)
        if self.adx[-1] > 25:
            # Uptrend
            if self.plus_di[-1] > self.minus_di[-1] and not self.positions["TEST"].is_long:
                self.positions["TEST"].buy()
            
            # Downtrend
            elif self.minus_di[-1] > self.plus_di[-1] and not self.positions["TEST"].is_short:
                self.positions["TEST"].sell()


strat = ADXTrendStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

**Example: Money Flow Index Divergence**

```python
class MFIDivergenceStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        self.mfi_14 = self.Indicator(self.ta.mfi(ohlc.High, ohlc.Low, ohlc.Close, ohlc.Volume, 14))
        self.rsi_14 = self.Indicator(self.ta.rsi(ohlc.Close, 14))

    def next(self):
        if len(self.mfi_14) < 2:
            return
        
        # Bullish divergence: price makes lower low, MFI makes higher low
        price_low = self.data["TEST"].Low[-1] < self.data["TEST"].Low[-2]
        mfi_higher = self.mfi_14[-1] > self.mfi_14[-2]
        
        if price_low and mfi_higher and self.mfi_14[-1] < 30:
            self.positions["TEST"].buy()


strat = MFIDivergenceStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Advanced and Research-Oriented Indicators

These indicators are useful for more sophisticated quantitative analysis.

| Function | Description | Parameters |
|----------|-------------|------------|
| `ichimoku_cloud(high, low, close, conversion, base, span_b, displacement)` | Tenkan-sen, Kijun-sen, cloud spans | `conversion`: 9, `base`: 26, `span_b`: 52, `displacement`: 26 |
| `zscore(values, period)` | Standardized score (mean=0, std=1) | `period`: 20 |
| `sharpe_ratio(values, period, risk_free_rate)` | Risk-adjusted return measure | `period`: 20, `risk_free_rate`: 0.0 |
| `sortino_ratio(values, period, target_return)` | Downside risk-adjusted return | `period`: 20, `target_return`: 0.0 |
| `hurst_exponent(values, period)` | Measures mean-reversion vs trend (H<0.5: mean-revert, H>0.5: trend) | `period`: 100 |
| `linear_regression_slope(values, period)` | Slope of linear regression line | `period`: 20 |

**Example: Z-Score Statistical Arbitrage**

```python
class ZScoreStrategy(Strategy):
    def init(self):
        close = self.data["TEST"].Close
        self.zscore_20 = self.Indicator(self.ta.zscore(close, 20))

    def next(self):
        if len(self.zscore_20) < 1:
            return
        
        z = self.zscore_20[-1]
        
        # Enter when z-score exceeds threshold (potential mean reversion)
        if z > 2.0 and not self.positions["TEST"].is_short:
            self.positions["TEST"].sell()  # Price likely to revert down
        
        elif z < -2.0 and not self.positions["TEST"].is_long:
            self.positions["TEST"].buy()  # Price likely to revert up
        
        # Exit at mean
        elif abs(z) < 0.5:
            self.positions["TEST"].close()


strat = ZScoreStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

**Example: Hurst Exponent Regime Detection**

```python
class HurstRegimeStrategy(Strategy):
    def init(self):
        close = self.data["TEST"].Close
        self.hurst = self.Indicator(self.ta.hurst_exponent(close, 100))

    def next(self):
        if len(self.hurst) < 1 or np.isnan(self.hurst[-1]):
            return
        
        h = self.hurst[-1]
        
        if h < 0.5:
            self.regime = "mean_reversion"  # Use range-bound strategies
        else:
            self.regime = "trend_following"  # Use momentum strategies


strat = HurstRegimeStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

## Multi-Output Indicators

Some indicators return multiple arrays. Unpack and register each series separately.

### MACD (Moving Average Convergence Divergence)

Returns: `(macd_line, signal_line, histogram)`

```python
class MACDStrategy(Strategy):
    def init(self):
        close = self.data["TEST"].Close
        macd_line, macd_signal, macd_hist = self.ta.macd(
            close,
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        self.macd_line = self.Indicator(macd_line)
        self.macd_signal = self.Indicator(macd_signal)
        self.macd_hist = self.Indicator(macd_hist)

    def next(self):
        # MACD crossover
        if len(self.macd_line) < 2:
            return
        
        if self.macd_line[-1] > self.macd_signal[-1] and self.macd_line[-2] <= self.macd_signal[-2]:
            # Bullish crossover
            pass


strat = MACDStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Bollinger Bands

Returns: `(lower_band, middle_band, upper_band)`

```python
class BollingerBandsStrategy(Strategy):
    def init(self):
        close = self.data["TEST"].Close
        lower, middle, upper = self.ta.bollinger_bands(
            close,
            period=20,
            std_dev=2.0
        )
        self.bb_lower = self.Indicator(lower)
        self.bb_middle = self.Indicator(middle)
        self.bb_upper = self.Indicator(upper)


strat = BollingerBandsStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### ADX (Average Directional Index)

Returns: `(adx_value, plus_di, minus_di)`

```python
class ADXStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        adx, plus_di, minus_di = self.ta.adx(
            ohlc.High,
            ohlc.Low,
            ohlc.Close,
            period=14
        )
        self.adx = self.Indicator(adx)
        self.plus_di = self.Indicator(plus_di)
        self.minus_di = self.Indicator(minus_di)


strat = ADXStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Stochastic Oscillator

Returns: `(percent_k, percent_d)`

```python
class StochasticOscillatorStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        k, d = self.ta.stochastic_oscillator(
            ohlc.High,
            ohlc.Low,
            ohlc.Close,
            k_period=14,
            d_period=3
        )
        self.stoch_k = self.Indicator(k)
        self.stoch_d = self.Indicator(d)


strat = StochasticOscillatorStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Ichimoku Cloud

Returns: `(conversion_line, base_line, span_a, span_b, lagging_span)`

```python
class IchimokuStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        conversion, base, span_a, span_b, lagging = self.ta.ichimoku_cloud(
            ohlc.High,
            ohlc.Low,
            ohlc.Close,
            conversion_period=9,
            base_period=26,
            span_b_period=52,
            displacement=26
        )
        self.tenkan_sen = self.Indicator(conversion)
        self.kijun_sen = self.Indicator(base)
        self.span_a = self.Indicator(span_a)
        self.span_b = self.Indicator(span_b)
        self.chikou_span = self.Indicator(lagging)


strat = IchimokuStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Keltner Channels

Returns: `(lower_channel, middle_channel, upper_channel)`

```python
class KeltnerChannelStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        lower, middle, upper = self.ta.keltner_channels(
            ohlc.High,
            ohlc.Low,
            ohlc.Close,
            ema_period=20,
            atr_period=10,
            multiplier=2.0
        )
        self.kc_lower = self.Indicator(lower)
        self.kc_middle = self.Indicator(middle)
        self.kc_upper = self.Indicator(upper)


strat = KeltnerChannelStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Donchian Channels

Returns: `(lower_channel, middle_channel, upper_channel)`

```python
class DonchianChannelStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        lower, middle, upper = self.ta.donchian_channels(
            ohlc.High,
            ohlc.Low,
            period=20
        )
        self.dc_lower = self.Indicator(lower)
        self.dc_middle = self.Indicator(middle)
        self.dc_upper = self.Indicator(upper)


strat = DonchianChannelStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### AROON

Returns: `(aroon_up, aroon_down, aroon_oscillator)`

```python
class AROONStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        up, down, oscillator = self.ta.aroon(
            ohlc.High,
            ohlc.Low,
            period=25
        )
        self.aroon_up = self.Indicator(up)
        self.aroon_down = self.Indicator(down)
        self.aroon_osc = self.Indicator(oscillator)


strat = AROONStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Vortex Indicator

Returns: `(vi_plus, vi_minus)`

```python
class VortexStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        vi_plus, vi_minus = self.ta.vortex(
            ohlc.High,
            ohlc.Low,
            ohlc.Close,
            period=14
        )
        self.vi_plus = self.Indicator(vi_plus)
        self.vi_minus = self.Indicator(vi_minus)


strat = VortexStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

## Time-Aware Behavior

[`Strategy.Indicator()`](../../src/quantex/strategy.py:126) wraps NumPy arrays into [`TimeNDArray`](../../src/quantex/helpers.py:7), which enforces look-ahead bias protection:

| Syntax | Description |
|--------|-------------|
| `indicator[-1]` | Current (latest visible) value |
| `indicator[-2]` | Previous visible value |
| `indicator[-5:]` | Last 5 visible values |
| `len(indicator)` | Number of visible values |

**Important**: Always check if the indicator has warmed up before using values:

```python
def next(self):
    # Wait for the indicator to have valid values
    if len(self.sma_20) < 20:
        return
    
    # Safe to use SMA values now
    if self.data["TEST"].Close[-1] > self.sma_20[-1]:
        pass
```

## Combining Built-in and Custom Indicators

You can mix built-in indicators with your own NumPy calculations:

```python
class CombinedIndicatorStrategy(Strategy):
    def init(self):
        close = self.data["TEST"].Close
        volume = self.data["TEST"].Volume
        
        # Built-in indicators
        self.atr_14 = self.Indicator(
            self.ta.atr(self.data["TEST"].High, self.data["TEST"].Low, close, 14)
        )
        self.rsi_14 = self.Indicator(self.ta.rsi(close, 14))
        
        # Custom: price momentum
        self.price_momentum = self.Indicator(
            (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else np.nan
        )
        
        # Custom: volume-weighted returns
        returns = np.diff(close, prepend=close[0]) / close[0]
        self.volume_signal = self.Indicator((returns * volume).cumsum())
        
        # Custom: cross-asset correlation proxy
        self.relative_strength = self.Indicator(
            (close / self.ta.sma(close, 20)) - 1.0
        )


strat = CombinedIndicatorStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

## Common Trading Strategies

### Trend-Following: EMA Crossover

```python
class EMACrossoverStrategy(Strategy):
    def init(self):
        close = self.data["TEST"].Close
        self.ema_fast = self.Indicator(self.ta.ema(close, 9))
        self.ema_slow = self.Indicator(self.ta.ema(close, 21))

    def next(self):
        if len(self.ema_fast) < 2:
            return
        
        # Golden cross
        if self.ema_fast[-1] > self.ema_slow[-1] and self.ema_fast[-2] <= self.ema_slow[-2]:
            self.positions["TEST"].buy()
        
        # Death cross
        elif self.ema_fast[-1] < self.ema_slow[-1] and self.ema_fast[-2] >= self.ema_slow[-2]:
            self.positions["TEST"].close()


strat = EMACrossoverStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Mean Reversion: Bollinger Band + RSI

```python
class BollingerReversionStrategy(Strategy):
    def init(self):
        close = self.data["TEST"].Close
        lower, middle, upper = self.ta.bollinger_bands(close, 20, 2.0)
        self.bb_lower = self.Indicator(lower)
        self.bb_upper = self.Indicator(upper)
        self.rsi_14 = self.Indicator(self.ta.rsi(close, 14))

    def next(self):
        if len(self.bb_lower) < 1:
            return
        
        price = self.data["TEST"].Close[-1]
        
        # Oversold at lower band
        if price <= self.bb_lower[-1] and self.rsi_14[-1] < 30:
            self.positions["TEST"].buy()
        
        # Overbought at upper band
        elif price >= self.bb_upper[-1] and self.rsi_14[-1] > 70:
            self.positions["TEST"].close()


strat = BollingerReversionStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Breakout: Donchian Channel

```python
class DonchianBreakoutStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        lower, middle, upper = self.ta.donchian_channels(
            ohlc.High,
            ohlc.Low,
            20
        )
        self.dc_upper = self.Indicator(upper)
        self.dc_lower = self.Indicator(lower)

    def next(self):
        if len(self.dc_upper) < 1:
            return
        
        price = self.data["TEST"].Close[-1]
        
        # Breakout above upper band
        if price > self.dc_upper[-1]:
            self.positions["TEST"].buy()
        
        # Breakdown below lower band
        elif price < self.dc_lower[-1]:
            self.positions["TEST"].close()


strat = DonchianBreakoutStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Momentum: MACD with RSI Filter

```python
class MACDMomentumStrategy(Strategy):
    def init(self):
        close = self.data["TEST"].Close
        macd_line, macd_signal, macd_hist = self.ta.macd(close)
        self.macd_line = self.Indicator(macd_line)
        self.macd_signal = self.Indicator(macd_signal)
        self.macd_hist = self.Indicator(macd_hist)
        self.rsi_14 = self.Indicator(self.ta.rsi(close, 14))

    def next(self):
        if len(self.macd_line) < 2:
            return
        
        # MACD bullish crossover with RSI confirmation
        macd_bullish = self.macd_line[-1] > self.macd_signal[-1] and self.macd_line[-2] <= self.macd_signal[-2]
        rsi_confirm = 50 < self.rsi_14[-1] < 65
        
        if macd_bullish and rsi_confirm:
            self.positions["TEST"].buy()
        
        # Exit on MACD bearish crossover or RSI overbought
        macd_bearish = self.macd_line[-1] < self.macd_signal[-1] and self.macd_line[-2] >= self.macd_signal[-2]
        if macd_bearish or self.rsi_14[-1] > 75:
            self.positions["TEST"].close()


strat = MACDMomentumStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

### Regime-Based: ADX + AROON

```python
class RegimeSwitchingStrategy(Strategy):
    def init(self):
        ohlc = self.data["TEST"]
        adx, plus_di, minus_di = self.ta.adx(ohlc.High, ohlc.Low, ohlc.Close, 14)
        up, down, osc = self.ta.aroon(ohlc.High, ohlc.Low, 25)
        
        self.adx = self.Indicator(adx)
        self.plus_di = self.Indicator(plus_di)
        self.minus_di = self.Indicator(minus_di)
        self.aroon_osc = self.Indicator(osc)

    def next(self):
        if len(self.adx) < 1:
            return
        
        # Strong trend regime
        if self.adx[-1] > 25:
            if self.plus_di[-1] > self.minus_di[-1]:
                self.positions["TEST"].buy()
            elif self.minus_di[-1] > self.plus_di[-1]:
                self.positions["TEST"].close()
        
        # No trend / ranging regime - use mean reversion
        else:
            if self.aroon_osc[-1] > 50:
                self.positions["TEST"].close()
            elif self.aroon_osc[-1] < -50:
                self.positions["TEST"].buy()


strat = RegimeSwitchingStrategy()
strat.add_data(CSVDataSource("data.csv"), "TEST")
```

## Best Practices

### 1. Data Source Initialization Pattern

Data sources are added inside the `init()` method:

```python
class MyStrategy(Strategy):
    def init(self):
        self.add_data(datasource, "NAME")
```

This ensures data is available when the backtest starts.

### 2. Warm-Up Period

Most rolling indicators require a warm-up period before producing valid values:

```python
def next(self):
    # Check minimum history required
    if len(self.indicator) < indicator_period:
        return
```

### 3. Look-Ahead Bias Prevention

Always wrap indicators with `self.Indicator()`:

```python
# WRONG - raw array, no look-ahead protection
self.sma = self.ta.sma(close, 20)

# CORRECT - time-aware array
self.sma = self.Indicator(self.ta.sma(close, 20))
```

### 4. Multi-Timeframe Analysis

Combine short and long period indicators for robust signals:

```python
def init(self):
    close = self.data["TEST"].Close
    ohlc = self.data["TEST"]
    # Short-term trend
    self.sma_10 = self.Indicator(self.ta.sma(close, 10))
    # Long-term trend
    self.sma_50 = self.Indicator(self.ta.sma(close, 50))
    # Trend strength
    self.adx = self.Indicator(self.ta.adx(ohlc.High, ohlc.Low, close, 14))
```

### 5. Signal Confirmation

Use multiple indicators to confirm signals:

```python
def next(self):
    # Multiple confirmations for a buy signal
    rsi_oversold = self.rsi_14[-1] < 30
    macd_bullish = self.macd_hist[-1] > 0
    adx_strong = self.adx[-1] > 25
    price_above_sma = self.data["TEST"].Close[-1] > self.sma_50[-1]
    
    if all([rsi_oversold, macd_bullish, adx_strong, price_above_sma]):
        self.positions["TEST"].buy()
```

## Input Requirements

| Indicator Type | Required Inputs | Notes |
|---------------|-----------------|-------|
| Single-series | `values` (1D array) | Price data (close, open, etc.) |
| OHLC-based | `high`, `low`, `close` | Same-length arrays |
| Volume-based | `close`, `volume` | Same-length arrays |
| Full OHLCV | `high`, `low`, `close`, `volume` | For VWAP, MFI, etc. |

All inputs must be one-dimensional and have the same length (except for OHLC-specific indicators).

## Summary

To use indicators effectively:

1. **Create** your strategy class with `init()` and `next()` methods
2. **Add** data sources inside the `init()` method: `self.add_data(datasource, "NAME")`
3. **Compute** arrays using `self.ta` (in strategy) or `quantex.indicators` (standalone)
4. **Wrap** each series with `self.Indicator()` for time-aware access
5. **Check** for warm-up periods before using values in `next()`
6. **Combine** multiple indicators for robust signal generation

For broader strategy design, see the [Strategy guide](./strategy.md) and [Backtesting guide](./backtesting.md).
