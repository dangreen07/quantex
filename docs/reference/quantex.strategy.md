# Strategy API Reference

This page provides comprehensive API documentation for QuantEx's strategy framework.

## Strategy Class

The `Strategy` class is the base class for all trading strategies in QuantEx.

```python
class Strategy(ABC):
    def __init__(self):
        self.positions: dict[str, Broker] = {}
        self.data: dict[str, DataSource] = {}
        self.indicators: list[TimeNDArray] = {}
```

### Abstract Methods

#### `init(self)`

Initialize your strategy before backtesting begins.

```python
@abstractmethod
def init(self):
    """Initialize strategy, load data, and set up indicators"""
    pass
```

**Usage:**
```python
def init(self):
    # Load market data
    data = CSVDataSource('data/EURUSD.csv')
    self.add_data(data, 'EURUSD')

    # Set up indicators
    close_prices = self.data['EURUSD'].Close
    self.sma_20 = self.Indicator(self.calculate_sma(close_prices, 20))
```

#### `next(self)`

Execute trading logic for each time step.

```python
@abstractmethod
def next(self):
    """Execute trading logic for current time step"""
    pass
```

**Usage:**
```python
def next(self):
    # Ensure we have enough data
    if len(self.sma_20) < 2:
        return

    # Generate trading signals
    if self.should_buy():
        self.positions['EURUSD'].buy(0.5)
    elif self.should_sell():
        self.positions['EURUSD'].sell(0.5)
```

### Instance Methods

#### `add_data(self, source, symbol)`

Register a data source for a symbol.

```python
def add_data(self, source: DataSource, symbol: str):
    """Add a data source for the specified symbol"""
    pass
```

**Parameters:**
- `source` (DataSource): The data source object
- `symbol` (str): Symbol name (e.g., 'EURUSD')

**Usage:**
```python
def init(self):
    data = CSVDataSource('data/EURUSD.csv')
    self.add_data(data, 'EURUSD')
```

#### `Indicator(self, arr)`

Create a time-aware indicator from a numpy array.

```python
def Indicator(self, arr: np.typing.NDArray):
    """Create a time-aware indicator"""
    pass
```

**Parameters:**
- `arr` (np.ndarray): Indicator values as numpy array

**Returns:**
- `TimeNDArray`: Time-aware indicator object

**Usage:**
```python
def init(self):
    close_prices = self.data['EURUSD'].Close
    self.sma_20 = self.Indicator(self.calculate_sma(close_prices, 20))

def calculate_sma(self, prices, period):
    return pd.Series(prices).rolling(window=period).mean().values
```

## TimeNDArray Class

Time-aware numpy array that only exposes data up to the current time step.

### Properties

#### Visible Data Access

```python
# Access current value (most recent)
current_value = indicator[-1]

# Access previous value
previous_value = indicator[-2]

# Access historical values (up to current time)
historical_values = indicator[:]

# Get length (up to current time)
length = len(indicator)
```

### Methods

#### `from_array(cls, arr)`

Create a TimeNDArray from a numpy array.

```python
@classmethod
def from_array(cls, arr: npt.NDArray[T]) -> "TimeNDArray[T]":
    """Create TimeNDArray from numpy array"""
    pass
```

**Usage:**
```python
# Create indicator from calculation
sma_values = self.calculate_sma(prices, 20)
indicator = TimeNDArray.from_array(sma_values)
```

#### `visible(self)`

Return a plain numpy array view of visible data.

```python
def visible(self) -> npt.NDArray[T]:
    """Return visible portion as plain numpy array"""
    pass
```

**Usage:**
```python
# Get visible data for calculations
visible_sma = self.sma_20.visible()
```

## Complete Strategy Example

```python
from quantex import Strategy, CSVDataSource
import pandas as pd
import numpy as np

class MovingAverageStrategy(Strategy):
    """
    Moving Average Crossover Strategy

    This strategy generates buy/sell signals based on moving average crossovers.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30, symbol: str = 'EURUSD'):
        """
        Initialize strategy parameters

        Args:
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            symbol: Trading symbol
        """
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.symbol = symbol

    def init(self):
        """Initialize strategy with data and indicators"""

        # Validate parameters
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")

        # Load market data
        data = CSVDataSource(f'data/{self.symbol}.csv')
        self.add_data(data, self.symbol)

        # Create indicators
        close_prices = self.data[self.symbol].Close
        self.fast_ma = self.Indicator(self.sma(close_prices, self.fast_period))
        self.slow_ma = self.Indicator(self.sma(close_prices, self.slow_period))

        # Strategy state
        self.position_size = 0.1  # Use 10% of capital per trade

    def next(self):
        """Execute trading logic"""

        # Ensure we have enough data
        if len(self.fast_ma) < 2 or len(self.slow_ma) < 2:
            return

        # Get current values
        current_fast = self.fast_ma[-1]
        current_slow = self.slow_ma[-1]
        previous_fast = self.fast_ma[-2]
        previous_slow = self.slow_ma[-2]
        current_price = self.data[self.symbol].CClose

        # Bullish crossover: fast MA crosses above slow MA
        if previous_fast <= previous_slow and current_fast > current_slow:
            if self.positions[self.symbol].position <= 0:  # Not currently long
                self.positions[self.symbol].buy(self.position_size)

        # Bearish crossover: fast MA crosses below slow MA
        elif previous_fast >= previous_slow and current_fast < current_slow:
            if self.positions[self.symbol].position >= 0:  # Not currently short
                self.positions[self.symbol].sell(self.position_size)

    def sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        return pd.Series(prices).rolling(window=period).mean().values

    def calculate_position_size(self) -> float:
        """Calculate dynamic position size based on volatility"""
        # Implementation here
        return self.position_size

    def should_adjust_position(self) -> bool:
        """Check if position should be adjusted"""
        # Implementation here
        return False
```

## Strategy Development Patterns

### Parameter Validation

```python
def __init__(self, fast_period=10, slow_period=30):
    super().__init__()

    # Validate parameters
    if not isinstance(fast_period, int) or fast_period < 1:
        raise ValueError("fast_period must be a positive integer")

    if not isinstance(slow_period, int) or slow_period < 1:
        raise ValueError("slow_period must be a positive integer")

    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    self.fast_period = fast_period
    self.slow_period = slow_period
```

### Error Handling

```python
def next(self):
    try:
        # Your trading logic here
        if self.buy_signal():
            self.positions['EURUSD'].buy(0.5)

    except Exception as e:
        # Log error and decide whether to continue
        print(f"Strategy error: {e}")
        # Option 1: Re-raise to stop backtest
        # raise
        # Option 2: Continue with next step
        # pass
```

### State Management

```python
class StatefulStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.trade_count = 0
        self.daily_trade_limit = 3
        self.last_trade_date = None

    def init(self):
        # Initialize state
        self.reset_daily_state()

    def next(self):
        current_date = self.data['EURUSD'].Index[self.data['EURUSD'].current_index].date()

        # Reset daily state if new day
        if self.last_trade_date != current_date:
            self.reset_daily_state()
            self.last_trade_date = current_date

        # Check trade limits
        if self.trade_count >= self.daily_trade_limit:
            return

        # Trading logic
        if self.should_trade():
            self.positions['EURUSD'].buy(0.1)
            self.trade_count += 1

    def reset_daily_state(self):
        """Reset daily trading state"""
        self.trade_count = 0
```

## Best Practices

### 1. Indicator Management

```python
def init(self):
    # Create indicators efficiently
    close = self.data['EURUSD'].Close

    # Batch indicator calculations
    indicators = self.calculate_multiple_indicators(close)

    # Register indicators
    for name, values in indicators.items():
        setattr(self, name, self.Indicator(values))

def calculate_multiple_indicators(self, prices):
    """Calculate multiple indicators efficiently"""
    return {
        'sma_10': self.sma(prices, 10),
        'sma_20': self.sma(prices, 20),
        'ema_10': self.ema(prices, 10),
        'ema_20': self.ema(prices, 20)
    }
```

### 2. Memory Efficiency

```python
def init(self):
    # Limit indicator history for memory efficiency
    if len(self.data['EURUSD'].Close) > 100000:  # Large dataset
        # Use shorter lookback periods
        self.lookback = 50
    else:
        self.lookback = 200

    self.sma = self.Indicator(self.sma(self.data['EURUSD'].Close, self.lookback))
```

### 3. Debugging Support

```python
class DebuggableStrategy(Strategy):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.execution_log = []

    def next(self):
        if self.debug:
            self.log_execution_state()

        # Your trading logic here
        pass

    def log_execution_state(self):
        """Log current strategy state for debugging"""
        state = {
            'timestamp': self.data['EURUSD'].Index[self.data['EURUSD'].current_index],
            'price': self.data['EURUSD'].CClose,
            'position': self.positions['EURUSD'].position,
            'indicators': {
                'sma': self.sma[-1] if len(self.sma) > 0 else None
            }
        }
        self.execution_log.append(state)
```

## Integration with Backtester

### Strategy and Backtester Interaction

```python
# Strategy receives broker instances from backtester
def init(self):
    # self.positions is populated by backtester
    # Each position has a broker instance
    pass

def next(self):
    # Access broker functionality through positions
    eur_usd_position = self.positions['EURUSD']

    # Place orders
    eur_usd_position.buy(0.5)
    eur_usd_position.sell(0.3)

    # Access broker information
    cash = eur_usd_position.cash
    current_position = eur_usd_position.position
    orders = eur_usd_position.orders
```

## Common Strategy Patterns

### Mean Reversion Strategy

```python
class MeanReversionStrategy(Strategy):
    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Mean reversion indicators
        self.mean = self.Indicator(self.sma(self.data['EURUSD'].Close, 20))
        self.std = self.Indicator(self.std(self.data['EURUSD'].Close, 20))

    def next(self):
        if len(self.mean) < 1 or len(self.std) < 1:
            return

        current_price = self.data['EURUSD'].CClose
        z_score = (current_price - self.mean[-1]) / self.std[-1]

        # Buy when price is significantly below mean
        if z_score < -2.0:
            self.positions['EURUSD'].buy(0.3)

        # Sell when price is significantly above mean
        elif z_score > 2.0:
            self.positions['EURUSD'].sell(0.3)
```

### Trend Following Strategy

```python
class TrendFollowingStrategy(Strategy):
    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Trend indicators
        self.short_ma = self.Indicator(self.ema(self.data['EURUSD'].Close, 10))
        self.long_ma = self.Indicator(self.ema(self.data['EURUSD'].Close, 30))

    def next(self):
        if len(self.short_ma) < 2 or len(self.long_ma) < 2:
            return

        # Trend following logic
        if (self.short_ma[-1] > self.long_ma[-1] and
            self.short_ma[-2] <= self.long_ma[-2]):
            # Bullish trend
            self.positions['EURUSD'].buy(0.4)

        elif (self.short_ma[-1] < self.long_ma[-1] and
              self.short_ma[-2] >= self.long_ma[-2]):
            # Bearish trend
            self.positions['EURUSD'].sell(0.4)
```

## Error Handling

### Common Exceptions

```python
class StrategyError(Exception):
    """Base exception for strategy errors"""
    pass

class InvalidParameterError(StrategyError):
    """Raised when strategy parameters are invalid"""
    pass

class DataError(StrategyError):
    """Raised when data issues occur"""
    pass

class Strategy(Strategy):
    def __init__(self, fast_period=10, slow_period=30):
        super().__init__()

        try:
            # Validate parameters
            if fast_period >= slow_period:
                raise InvalidParameterError("Fast period must be less than slow period")

            self.fast_period = fast_period
            self.slow_period = slow_period

        except InvalidParameterError as e:
            print(f"Strategy configuration error: {e}")
            raise

    def init(self):
        try:
            data = CSVDataSource('data/EURUSD.csv')
            self.add_data(data, 'EURUSD')

        except FileNotFoundError:
            raise DataError("Data file not found")
        except Exception as e:
            raise DataError(f"Data loading error: {e}")
```

## Performance Considerations

### Efficient Indicator Calculation

```python
def init(self):
    # Pre-calculate expensive operations
    self.expensive_calculation = self.pre_calculate_expensive_operation()

def pre_calculate_expensive_operation(self):
    """Pre-calculate expensive operations"""
    # Implementation here
    return {}

def next(self):
    # Use pre-calculated values for efficiency
    if self.should_trade():
        self.positions['EURUSD'].buy(0.5)
```

### Memory Management

```python
def init(self):
    # Limit data retention for large datasets
    max_data_points = 100000

    if len(self.data['EURUSD'].Close) > max_data_points:
        # Strategy for handling large datasets
        self.use_rolling_window = True
        self.window_size = 50000
    else:
        self.use_rolling_window = False
```

## Related Documentation

- **[Usage Guide](../../../usage/strategy.md)**: Learn how to use the Strategy class
- **[Backtester API](../../quantex.backtest.md)**: Understand backtesting integration
- **[Data Sources API](../../quantex.sources.md)**: Work with market data
- **[Broker API](../../quantex.execution.md)**: Handle order execution