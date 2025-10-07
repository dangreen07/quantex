# Technical Indicators Guide

This guide covers how to create, use, and manage technical indicators in QuantEx strategies.

## Overview

QuantEx provides a flexible system for working with technical indicators:

- **Time-aware indicators**: Only use data available up to current time step
- **Memory efficient**: Optimized for large datasets
- **Easy integration**: Simple API for custom indicator creation
- **Built-in helpers**: Common indicators and calculation utilities

## Basic Indicator Usage

### Creating Indicators

```python
class IndicatorStrategy(Strategy):
    def init(self):
        # Load data
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Create indicators using the Indicator helper
        close_prices = self.data['EURUSD'].Close

        # Simple Moving Average
        self.sma_20 = self.Indicator(self.sma(close_prices, 20))

        # Exponential Moving Average
        self.ema_12 = self.Indicator(self.ema(close_prices, 12))

        # RSI
        self.rsi_14 = self.Indicator(self.rsi(close_prices, 14))

    def sma(self, prices, period):
        """Simple Moving Average"""
        return pd.Series(prices).rolling(window=period).mean().values

    def ema(self, prices, period):
        """Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period).mean().values

    def rsi(self, prices, period):
        """Relative Strength Index"""
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).fillna(50).values
```

### Using Indicators in Trading Logic

```python
def next(self):
    # Ensure we have enough data
    if len(self.sma_20) < 2:
        return

    # Access current and previous values
    current_sma = self.sma_20[-1]
    previous_sma = self.sma_20[-2]

    current_price = self.data['EURUSD'].CClose

    # Simple moving average crossover strategy
    if current_price > current_sma and previous_price <= previous_sma:
        # Bullish crossover
        self.positions['EURUSD'].buy(0.5)

    elif current_price < current_sma and previous_price >= previous_sma:
        # Bearish crossover
        self.positions['EURUSD'].sell(0.5)
```

## Common Technical Indicators

### Trend Indicators

```python
def calculate_trend_indicators(self, prices, short_period=10, long_period=20):
    """Calculate common trend indicators"""

    indicators = {}

    # Simple Moving Averages
    indicators['sma_short'] = self.sma(prices, short_period)
    indicators['sma_long'] = self.sma(prices, long_period)

    # Exponential Moving Averages
    indicators['ema_short'] = self.ema(prices, short_period)
    indicators['ema_long'] = self.ema(prices, long_period)

    # Moving Average Convergence Divergence (MACD)
    ema_short = pd.Series(prices).ewm(span=12).mean()
    ema_long = pd.Series(prices).ewm(span=26).mean()
    indicators['macd'] = (ema_short - ema_long).values
    indicators['macd_signal'] = ema_short.ewm(span=9).mean().values
    indicators['macd_hist'] = (indicators['macd'] - indicators['macd_signal'])

    return indicators

def sma(self, prices, period):
    return pd.Series(prices).rolling(window=period).mean().values

def ema(self, prices, period):
    return pd.Series(prices).ewm(span=period).mean().values
```

### Momentum Indicators

```python
def calculate_momentum_indicators(self, prices, periods=[14, 28]):
    """Calculate common momentum indicators"""

    indicators = {}

    for period in periods:
        # Rate of Change (ROC)
        indicators[f'roc_{period}'] = (
            (prices - np.roll(prices, period)) / np.roll(prices, period) * 100
        )

        # Momentum
        indicators[f'momentum_{period}'] = prices - np.roll(prices, period)

    # Stochastic Oscillator
    high_14 = pd.Series(self.data['EURUSD'].High).rolling(window=14).max()
    low_14 = pd.Series(self.data['EURUSD'].Low).rolling(window=14).min()
    indicators['stoch_k'] = 100 * (prices - low_14) / (high_14 - low_14)
    indicators['stoch_d'] = indicators['stoch_k'].rolling(window=3).mean()

    return indicators
```

### Volatility Indicators

```python
def calculate_volatility_indicators(self, prices, periods=[10, 20]):
    """Calculate common volatility indicators"""

    indicators = {}

    for period in periods:
        # Bollinger Bands
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()

        indicators[f'bb_upper_{period}'] = (sma + 2 * std).values
        indicators[f'bb_middle_{period}'] = sma.values
        indicators[f'bb_lower_{period}'] = (sma - 2 * std).values
        indicators[f'bb_width_{period}'] = ((indicators[f'bb_upper_{period}'] -
                                           indicators[f'bb_lower_{period}']) / sma).values

        # Average True Range (ATR)
        high = self.data['EURUSD'].High
        low = self.data['EURUSD'].Low
        close = prices

        tr1 = high - low
        tr2 = abs(high - np.roll(close, 1))
        tr3 = abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        indicators[f'atr_{period}'] = pd.Series(tr).rolling(window=period).mean().values

    return indicators
```

## Advanced Indicator Techniques

### Multi-Timeframe Indicators

```python
class MultiTimeframeStrategy(Strategy):
    def init(self):
        # Load same symbol on different timeframes
        self.add_data(CSVDataSource('data/EURUSD_M1.csv'), 'EURUSD_M1')
        self.add_data(CSVDataSource('data/EURUSD_H1.csv'), 'EURUSD_H1')

        # Indicators on different timeframes
        self.m1_sma = self.Indicator(self.sma(self.data['EURUSD_M1'].Close, 20))
        self.h1_trend = self.Indicator(self.sma(self.data['EURUSD_H1'].Close, 50))

    def next(self):
        # Use H1 trend for direction, M1 for timing
        if len(self.h1_trend) < 1 or len(self.m1_sma) < 2:
            return

        h1_trend_up = self.data['EURUSD_H1'].Close[-1] > self.h1_trend[-1]
        m1_bullish = (self.data['EURUSD_M1'].CClose > self.m1_sma[-1] and
                     self.data['EURUSD_M1'].Close[-2] <= self.m1_sma[-2])

        if h1_trend_up and m1_bullish:
            self.positions['EURUSD_M1'].buy(0.3)
```

### Dynamic Indicator Parameters

```python
class AdaptiveStrategy(Strategy):
    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Adaptive parameters based on market volatility
        self.base_period = 20
        self.volatility_period = self.Indicator(self.calculate_volatility_period())

    def calculate_volatility_period(self):
        """Calculate adaptive period based on volatility"""
        prices = self.data['EURUSD'].Close
        returns = pd.Series(prices).pct_change()

        # Higher volatility = shorter period
        volatility = returns.rolling(window=50).std().values
        normalized_vol = volatility / volatility.mean()

        # Adaptive period between 10 and 30
        adaptive_period = 20 * (2 - normalized_vol)  # Inverse relationship
        adaptive_period = np.clip(adaptive_period, 10, 30)

        return adaptive_period.astype(int)

    def next(self):
        if len(self.volatility_period) < 1:
            return

        # Use adaptive period for indicator calculation
        current_period = max(10, int(self.volatility_period[-1]))
        sma = self.sma(self.data['EURUSD'].Close, current_period)

        # Use for trading decisions
        current_price = self.data['EURUSD'].CClose

        if current_price > sma[-1]:
            self.positions['EURUSD'].buy(0.5)
```

### Indicator Combinations

```python
def calculate_combined_indicators(self, high, low, close, volume):
    """Calculate indicators that combine multiple price series"""

    indicators = {}

    # Average Directional Index (ADX)
    high_diff = pd.Series(high).diff()
    low_diff = pd.Series(low).diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    atr = self.calculate_atr(high, low, close, 14)
    indicators['plus_di'] = 100 * pd.Series(plus_dm).rolling(window=14).mean() / atr
    indicators['minus_di'] = 100 * pd.Series(minus_dm).rolling(window=14).mean() / atr

    # ADX calculation
    dx = 100 * abs(indicators['plus_di'] - indicators['minus_di']) / (indicators['plus_di'] + indicators['minus_di'])
    indicators['adx'] = pd.Series(dx).rolling(window=14).mean().values

    # On Balance Volume (OBV)
    obv = np.zeros_like(close)
    obv[0] = volume[0]

    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]

    indicators['obv'] = obv

    return indicators

def calculate_atr(self, high, low, close, period):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - np.roll(close, 1))
    tr3 = abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return pd.Series(tr).rolling(window=period).mean().values
```

## Indicator Helper Functions

### Price Transform Indicators

```python
def calculate_price_transforms(self, prices):
    """Calculate common price transformations"""

    transforms = {}

    # Log prices
    transforms['log_prices'] = np.log(prices)

    # Price differences
    transforms['price_diff'] = np.diff(prices)

    # Percentage changes
    transforms['pct_change'] = (np.diff(prices) / prices[:-1]) * 100

    # Normalized prices (0-1 scale)
    price_min = np.min(prices)
    price_max = np.max(prices)
    transforms['normalized'] = (prices - price_min) / (price_max - price_min)

    return transforms
```

### Statistical Indicators

```python
def calculate_statistical_indicators(self, prices, window=20):
    """Calculate statistical indicators"""

    stats = {}

    # Rolling statistics
    price_series = pd.Series(prices)

    stats['rolling_mean'] = price_series.rolling(window=window).mean().values
    stats['rolling_std'] = price_series.rolling(window=window).std().values
    stats['rolling_var'] = price_series.rolling(window=window).var().values

    # Z-score
    stats['z_score'] = (prices - stats['rolling_mean']) / stats['rolling_std']

    # Rolling skewness
    stats['skewness'] = price_series.rolling(window=window).skew().values

    # Rolling kurtosis
    stats['kurtosis'] = price_series.rolling(window=window).kurt().values

    return stats
```

## Custom Indicator Library

### Creating Reusable Indicators

```python
class IndicatorLibrary:
    """Library of reusable technical indicators"""

    @staticmethod
    def sma(prices, period):
        """Simple Moving Average"""
        return pd.Series(prices).rolling(window=period).mean().values

    @staticmethod
    def ema(prices, period):
        """Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period).mean().values

    @staticmethod
    def rsi(prices, period=14):
        """Relative Strength Index"""
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).fillna(50).values

    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = pd.Series(prices).ewm(span=fast).mean()
        ema_slow = pd.Series(prices).ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line.values,
            'signal': signal_line.values,
            'histogram': histogram.values
        }

    @staticmethod
    def bollinger_bands(prices, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()

        return {
            'upper': (sma + std_dev * std).values,
            'middle': sma.values,
            'lower': (sma - std_dev * std).values,
            'width': ((sma + std_dev * std) - (sma - std_dev * std)) / sma.values
        }
```

### Using the Indicator Library

```python
class LibraryStrategy(Strategy):
    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Use indicator library
        lib = IndicatorLibrary()
        close_prices = self.data['EURUSD'].Close

        # Create multiple indicators
        self.sma_20 = self.Indicator(lib.sma(close_prices, 20))
        self.ema_12 = self.Indicator(lib.ema(close_prices, 12))

        macd_data = lib.macd(close_prices)
        self.macd = self.Indicator(macd_data['macd'])
        self.macd_signal = self.Indicator(macd_data['signal'])

        bb_data = lib.bollinger_bands(close_prices)
        self.bb_upper = self.Indicator(bb_data['upper'])
        self.bb_lower = self.Indicator(bb_data['lower'])

    def next(self):
        # Use indicators for trading decisions
        current_price = self.data['EURUSD'].CClose

        # Bollinger Band mean reversion
        if current_price < self.bb_lower[-1]:
            self.positions['EURUSD'].buy(0.3)
        elif current_price > self.bb_upper[-1]:
            self.positions['EURUSD'].sell(0.3)

        # MACD signal
        if (self.macd[-1] > self.macd_signal[-1] and
            self.macd[-2] <= self.macd_signal[-2]):
            self.positions['EURUSD'].buy(0.2)
```

## Indicator Performance Analysis

### Indicator Evaluation

```python
def evaluate_indicator_performance(self, indicator_values, price_direction, look_ahead=1):
    """Evaluate how well an indicator predicts price direction"""

    # Align indicator values with future price direction
    indicator_aligned = indicator_values[:-look_ahead]  # Remove last N values
    future_direction = price_direction[look_ahead:]     # Shift forward by N

    # Calculate indicator-based signals
    signals = np.where(indicator_aligned > 0, 1, -1)  # Simple threshold

    # Calculate accuracy
    correct_predictions = np.sum(signals == future_direction)
    total_predictions = len(signals)

    accuracy = correct_predictions / total_predictions

    # Calculate other metrics
    true_positives = np.sum((signals == 1) & (future_direction == 1))
    false_positives = np.sum((signals == 1) & (future_direction == -1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'total_signals': total_predictions,
        'correct_signals': correct_predictions
    }
```

### Indicator Optimization

```python
def optimize_indicator_parameters(self, prices, param_ranges):
    """Optimize indicator parameters for best performance"""

    best_params = None
    best_score = -np.inf
    results = []

    for params in self.generate_parameter_combinations(param_ranges):
        # Calculate indicator with current parameters
        indicator = self.calculate_indicator(prices, **params)

        # Evaluate performance
        score = self.evaluate_indicator(indicator, prices)

        results.append({**params, 'score': score})

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, pd.DataFrame(results)

def generate_parameter_combinations(self, param_ranges):
    """Generate all combinations of parameters"""
    import itertools

    keys = list(param_ranges.keys())
    values = list(param_ranges.values())

    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))
```

## Visualization and Plotting

### Indicator Plotting

```python
def plot_indicators(self, data_source, indicators, figsize=(12, 8)):
    """Plot price data with indicators"""

    fig, axes = plt.subplots(len(indicators) + 1, 1, figsize=figsize,
                            sharex=True)

    # Plot price
    axes[0].plot(data_source.Index[:len(data_source.Close)],
                data_source.Close, label='Price', color='black')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True)

    # Plot indicators
    for i, (name, indicator_data) in enumerate(indicators.items()):
        if len(indicator_data) > 0:
            axes[i+1].plot(data_source.Index[:len(indicator_data)],
                          indicator_data, label=name)
            axes[i+1].set_ylabel(name)
            axes[i+1].legend()
            axes[i+1].grid(True)

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()
```

### Interactive Indicator Analysis

```python
class InteractiveIndicatorStrategy(Strategy):
    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Create multiple indicators for comparison
        close_prices = self.data['EURUSD'].Close

        self.indicators = {
            'SMA_10': self.Indicator(self.sma(close_prices, 10)),
            'SMA_20': self.Indicator(self.sma(close_prices, 20)),
            'EMA_10': self.Indicator(self.ema(close_prices, 10)),
            'EMA_20': self.Indicator(self.ema(close_prices, 20)),
            'RSI_14': self.Indicator(self.rsi(close_prices, 14))
        }

    def next(self):
        # Log indicator values for analysis
        if len(self.data['EURUSD'].Close) % 100 == 0:  # Every 100 steps
            self.log_indicator_values()

    def log_indicator_values(self):
        """Log current indicator values for analysis"""

        current_price = self.data['EURUSD'].CClose

        indicator_values = {}
        for name, indicator in self.indicators.items():
            if len(indicator) > 0:
                indicator_values[name] = indicator[-1]

        print(f"Price: {current_price:.5f}")
        for name, value in indicator_values.items():
            print(f"{name}: {value:.5f}")
```

## Best Practices

### 1. Indicator Validation

```python
def validate_indicator(self, indicator_data, min_length=50):
    """Validate indicator data quality"""

    if len(indicator_data) < min_length:
        return False, "Insufficient data"

    if np.isnan(indicator_data).any():
        return False, "Contains NaN values"

    if np.isinf(indicator_data).any():
        return False, "Contains infinite values"

    # Check for reasonable value ranges
    if np.ptp(indicator_data) == 0:  # No variation
        return False, "No variation in indicator values"

    return True, "Valid"
```

### 2. Memory Management

```python
def manage_indicator_memory(self, max_indicators=10):
    """Manage memory usage for multiple indicators"""

    if len(self.indicators) > max_indicators:
        # Remove oldest indicators
        oldest_keys = list(self.indicators.keys())[:len(self.indicators)-max_indicators]
        for key in oldest_keys:
            del self.indicators[key]

    # Force garbage collection periodically
    if len(self.data['EURUSD'].Close) % 1000 == 0:
        import gc
        gc.collect()
```

### 3. Indicator Synchronization

```python
def synchronize_indicators(self, indicators):
    """Ensure all indicators have the same length"""

    min_length = min(len(indicator) for indicator in indicators.values())

    synchronized = {}
    for name, indicator in indicators.items():
        synchronized[name] = indicator[:min_length]

    return synchronized
```

## Common Patterns

### Indicator Crossover Detection

```python
def detect_crossovers(self, fast_indicator, slow_indicator):
    """Detect crossovers between two indicators"""

    if len(fast_indicator) < 2 or len(slow_indicator) < 2:
        return None

    current_fast = fast_indicator[-1]
    current_slow = slow_indicator[-1]
    previous_fast = fast_indicator[-2]
    previous_slow = slow_indicator[-2]

    # Bullish crossover
    if (previous_fast <= previous_slow and current_fast > current_slow):
        return 'bullish'

    # Bearish crossover
    elif (previous_fast >= previous_slow and current_fast < current_slow):
        return 'bearish'

    return None
```

### Indicator Divergence

```python
def detect_divergence(self, price, indicator, lookback=20):
    """Detect divergence between price and indicator"""

    if len(price) < lookback or len(indicator) < lookback:
        return None

    recent_prices = price[-lookback:]
    recent_indicator = indicator[-lookback:]

    # Price making higher highs but indicator making lower highs (bearish divergence)
    price_higher_high = recent_prices[-1] > recent_prices.max() * 0.99
    indicator_lower_high = recent_indicator[-1] < recent_indicator.max() * 0.99

    if price_higher_high and indicator_lower_high:
        return 'bearish'

    # Price making lower lows but indicator making higher lows (bullish divergence)
    price_lower_low = recent_prices[-1] < recent_prices.min() * 1.01
    indicator_higher_low = recent_indicator[-1] > recent_indicator.min() * 1.01

    if price_lower_low and indicator_higher_low:
        return 'bullish'

    return None
```

## Next Steps

Now that you understand technical indicators in QuantEx, explore these related topics:

- **[Strategy Guide](./strategy.md)**: Learn how to integrate indicators into trading strategies
- **[Backtesting Guide](./backtesting.md)**: Understand how indicators perform in backtests
- **[Data Sources Guide](./data-sources.md)**: Work with different data types for indicators

For complete API reference, see the [Technical Indicators API documentation](../../reference/quantex.indicators.md).