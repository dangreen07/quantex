# Strategy Development Guide

This guide covers how to create and implement trading strategies using QuantEx's strategy framework.

## Overview

QuantEx strategies are built by inheriting from the `Strategy` base class and implementing two key methods:

- `init()`: Initialize your strategy, load data, and set up indicators
- `next()`: Execute your trading logic for each time step

## Basic Strategy Structure

```python
from quantex import Strategy, SimpleBacktester, CSVDataSource
import pandas as pd
import numpy as np

class MyStrategy(Strategy):
    def __init__(self, param1=10, param2=20):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def init(self):
        # Load market data
        data = CSVDataSource('path/to/data.csv')
        self.add_data(data, 'SYMBOL')

        # Set up indicators
        close = self.data['SYMBOL'].Close
        self.my_indicator = self.Indicator(self.calculate_indicator(close))

    def next(self):
        # Your trading logic here
        if self.should_buy():
            self.positions['SYMBOL'].buy(0.5)  # Buy with 50% of cash
        elif self.should_sell():
            self.positions['SYMBOL'].sell(0.5)  # Sell 50% of position

    def calculate_indicator(self, prices):
        # Custom indicator calculation
        return pd.Series(prices).rolling(window=self.param1).mean().values

    def should_buy(self):
        # Buy signal logic
        return self.my_indicator[-1] > self.my_indicator[-2]

    def should_sell(self):
        # Sell signal logic
        return self.my_indicator[-1] < self.my_indicator[-2]
```

## Strategy Lifecycle

### 1. Initialization (`__init__`)

The constructor is called once when the strategy object is created:

```python
def __init__(self, fast_period=10, slow_period=30, symbol='EURUSD'):
    super().__init__()
    self.fast_period = fast_period
    self.slow_period = slow_period
    self.symbol = symbol
```

### 2. Setup (`init`)

Called once before backtesting begins. Use this method to:

- Load and register data sources
- Initialize indicators
- Set initial strategy state

```python
def init(self):
    # Load data
    data = CSVDataSource(f'data/{self.symbol}.csv')
    self.add_data(data, self.symbol)

    # Initialize indicators
    close_prices = self.data[self.symbol].Close
    self.fast_ma = self.Indicator(self.sma(close_prices, self.fast_period))
    self.slow_ma = self.Indicator(self.sma(close_prices, self.slow_period))

    # Strategy state
    self.position_size = 0.1  # Use 10% of cash per trade
```

### 3. Trading Logic (`next`)

Called for each time step during backtesting:

```python
def next(self):
    # Ensure we have enough data
    if len(self.fast_ma) < 2:
        return

    # Generate signals
    if self.generate_buy_signal():
        if self.positions[self.symbol].position <= 0:  # Not currently long
            self.positions[self.symbol].buy(self.position_size)

    elif self.generate_sell_signal():
        if self.positions[self.symbol].position >= 0:  # Not currently short
            self.positions[self.symbol].sell(self.position_size)

    # Risk management
    self.manage_risk()
```

## Working with Multiple Symbols

QuantEx supports strategies that trade multiple instruments simultaneously:

```python
def init(self):
    # Load multiple symbols
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

    for symbol in symbols:
        data = CSVDataSource(f'data/{symbol}.csv')
        self.add_data(data, symbol)

        # Set up symbol-specific indicators
        close = self.data[symbol].Close
        setattr(self, f'{symbol}_ma', self.Indicator(self.sma(close, 14)))

def next(self):
    for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
        if self.generate_signal(symbol):
            self.positions[symbol].buy(0.3)  # Allocate 30% to each symbol
```

## Using Indicators

### Built-in Indicator Helper

Use the `Indicator` method to create time-aware indicators:

```python
def init(self):
    close = self.data['SYMBOL'].Close

    # Simple Moving Average
    self.sma_20 = self.Indicator(self.sma(close, 20))

    # Exponential Moving Average
    self.ema_12 = self.Indicator(self.ema(close, 12))

    # RSI
    self.rsi_14 = self.Indicator(self.rsi(close, 14))

def sma(self, prices, period):
    return pd.Series(prices).rolling(window=period).mean().values

def ema(self, prices, period):
    return pd.Series(prices).ewm(span=period).mean().values

def rsi(self, prices, period):
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).values
```

### Accessing Indicator Values

Indicators are time-aware and only return values up to the current time step:

```python
def next(self):
    # Current value (most recent)
    current_sma = self.sma_20[-1]

    # Previous value
    previous_sma = self.sma_20[-2]

    # Historical values (as numpy array)
    sma_history = self.sma_20[:]

    # Check for crossovers
    if self.fast_ma[-1] > self.slow_ma[-1] and self.fast_ma[-2] <= self.slow_ma[-2]:
        # Bullish crossover
        self.positions['SYMBOL'].buy(0.5)
```

## Position Management

### Basic Position Operations

```python
def next(self):
    # Buy with different order types
    self.positions['SYMBOL'].buy(0.5)                    # Market buy with 50% cash
    self.positions['SYMBOL'].buy(0.5, limit=1.2500)     # Limit buy
    self.positions['SYMBOL'].buy(0.5, stop_loss=1.2400) # Buy with stop loss

    # Sell operations
    self.positions['SYMBOL'].sell(0.5)                   # Market sell
    self.positions['SYMBOL'].sell(0.5, limit=1.2600)    # Limit sell
    self.positions['SYMBOL'].sell(0.5, take_profit=1.2700) # Sell with take profit

    # Close entire position
    self.positions['SYMBOL'].close()
```

### Position Information

```python
def next(self):
    position = self.positions['SYMBOL']

    # Position details
    print(f"Current position: {position.position}")
    print(f"Position value: {position.position * self.data['SYMBOL'].CClose}")
    print(f"Available cash: {position.cash}")
    print(f"Unrealized PnL: {position.unrealized_pnl}")

    # Order management
    print(f"Active orders: {len(position.orders)}")
    print(f"Completed orders: {len(position.complete_orders)}")
```

## Advanced Strategy Patterns

### State Management

```python
class AdvancedStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.trade_count = 0
        self.last_trade_time = None
        self.max_trades_per_day = 3

    def init(self):
        # ... setup code ...

    def next(self):
        # Check trading frequency limits
        if self.should_limit_trading():
            return

        # Main trading logic
        if self.trade_count < self.max_trades_per_day:
            if self.buy_signal():
                self.positions['SYMBOL'].buy(0.1)
                self.trade_count += 1
                self.last_trade_time = self.data['SYMBOL'].Index[self.data['SYMBOL'].current_index]

    def should_limit_trading(self):
        # Implement your trading frequency logic
        return False
```

### Multi-Timeframe Strategies

```python
def init(self):
    # Load same symbol on different timeframes
    daily_data = CSVDataSource('data/EURUSD_D1.csv')
    hourly_data = CSVDataSource('data/EURUSD_H1.csv')

    self.add_data(daily_data, 'EURUSD_D1')
    self.add_data(hourly_data, 'EURUSD_H1')

def next(self):
    # Use daily trend for direction
    daily_trend = self.get_daily_trend()

    # Use hourly signals for timing
    if daily_trend == 'bullish' and self.hourly_buy_signal():
        self.positions['EURUSD_H1'].buy(0.3)
```

## Error Handling and Logging

```python
import logging

class RobustStrategy(Strategy):
    def __init__(self):
        super().__init__()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def next(self):
        try:
            # Your trading logic
            if self.buy_signal():
                self.positions['SYMBOL'].buy(0.5)
                self.logger.info(f"Buy signal at {self.data['SYMBOL'].CClose}")

        except Exception as e:
            self.logger.error(f"Error in strategy: {e}")
            # Decide whether to continue or stop
            raise  # Re-raise to stop backtest
```

## Best Practices

### 1. Parameter Validation

```python
def __init__(self, fast_period=10, slow_period=30):
    super().__init__()

    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    if fast_period < 1 or slow_period < 1:
        raise ValueError("Periods must be positive")

    self.fast_period = fast_period
    self.slow_period = slow_period
```

### 2. Data Validation

```python
def init(self):
    if 'SYMBOL' not in self.data:
        raise ValueError("Data not loaded properly")

    # Check for required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in self.data['SYMBOL'].data.columns:
            raise ValueError(f"Missing required column: {col}")
```

### 3. Performance Optimization

```python
def init(self):
    # Pre-calculate expensive operations
    self.lookback_periods = max(self.fast_period, self.slow_period) + 10

def next(self):
    # Check if we have enough data
    if len(self.data['SYMBOL'].Close) < self.lookback_periods:
        return

    # Your optimized trading logic here
```

## Running Your Strategy

```python
# Create strategy instance
strategy = MyStrategy(fast_period=10, slow_period=30)

# Set up backtester
backtester = SimpleBacktester(
    strategy,
    cash=10000,           # Starting capital
    commission=0.002,     # 0.2% commission
    commission_type=CommissionType.PERCENTAGE
)

# Run backtest
report = backtester.run()

# Display results
print(report)
```

## Next Steps

Now that you understand the basics of strategy development, explore these related topics:

- **[Data Sources Guide](../data-sources.md)**: Learn how to load and manage market data
- **[Backtesting Guide](../backtesting.md)**: Master the backtesting engine
- **[Technical Indicators Guide](../indicators.md)**: Work with built-in and custom indicators
- **[Optimization Guide](../optimizer.md)**: Optimize your strategy parameters

For complete API reference, see the [Strategy API documentation](../../reference/quantex.strategy.md).