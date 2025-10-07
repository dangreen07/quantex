# Models API Reference

This page provides comprehensive API documentation for QuantEx's core data models and enums.

## Enums

### CommissionType

Defines commission calculation methods.

```python
class CommissionType(Enum):
    PERCENTAGE = 0
    CASH = 1
```

**Values:**
- `CommissionType.PERCENTAGE`: Commission calculated as percentage of trade value
- `CommissionType.CASH`: Fixed cash commission per trade

**Usage:**
```python
from quantex import CommissionType

# Percentage-based commission (default)
commission_type = CommissionType.PERCENTAGE

# Fixed cash commission
commission_type = CommissionType.CASH
```

### OrderSide

Defines order direction.

```python
class OrderSide(Enum):
    BUY = 1
    SELL = -1
```

**Values:**
- `OrderSide.BUY`: Buy order (long position)
- `OrderSide.SELL`: Sell order (short position)

**Usage:**
```python
# Create buy order
if buy_signal:
    broker.buy(side=OrderSide.BUY, quantity=0.5)

# Create sell order
if sell_signal:
    broker.sell(side=OrderSide.SELL, quantity=0.3)
```

### OrderType

Defines order execution types.

```python
class OrderType(Enum):
    MARKET = 0
    LIMIT = 1
```

**Values:**
- `OrderType.MARKET`: Execute at current market price
- `OrderType.LIMIT`: Execute only at specified price or better

**Usage:**
```python
# Market order
broker.buy(quantity=0.5, order_type=OrderType.MARKET)

# Limit order
broker.buy(quantity=0.5, order_type=OrderType.LIMIT, limit_price=1.2500)
```

### OrderStatus

Defines order execution status.

```python
class OrderStatus(Enum):
    ACTIVE = 0    # Order executed with SL/TP conditions
    COMPLETE = 1  # Order fully executed
    PENDING = 2   # Order waiting for execution
```

**Values:**
- `OrderStatus.PENDING`: Order created but not yet executed
- `OrderStatus.ACTIVE`: Order executed but has stop loss/take profit conditions
- `OrderStatus.COMPLETE`: Order fully executed with no remaining conditions

**Usage:**
```python
# Check order status
for order in broker.orders:
    if order.status == OrderStatus.PENDING:
        print(f"Order waiting: {order.side.name}")
    elif order.status == OrderStatus.ACTIVE:
        print(f"Order active with conditions: {order.side.name}")
    elif order.status == OrderStatus.COMPLETE:
        print(f"Order completed: {order.side.name}")
```

## Data Classes

### Order

Represents a trading order with execution parameters.

```python
@dataclass
class Order:
    side: OrderSide
    quantity: np.float64
    type: OrderType
    price: np.float64 | None
    stop_loss: np.float64 | None
    take_profit: np.float64 | None
    status: OrderStatus
    timestamp: datetime
```

#### Fields

##### `side: OrderSide`
Order direction (BUY or SELL).

##### `quantity: np.float64`
Order quantity (positive number).

##### `type: OrderType`
Order type (MARKET or LIMIT).

##### `price: np.float64 | None`
Limit price for limit orders (None for market orders).

##### `stop_loss: np.float64 | None`
Stop loss price (None if not set).

##### `take_profit: np.float64 | None`
Take profit price (None if not set).

##### `status: OrderStatus`
Current order status.

##### `timestamp: datetime`
Order creation timestamp.

#### Usage

```python
from datetime import datetime
from quantex import Order, OrderSide, OrderType, OrderStatus

# Create buy market order
order = Order(
    side=OrderSide.BUY,
    quantity=1000.0,
    type=OrderType.MARKET,
    price=None,
    stop_loss=1.2400,
    take_profit=1.2700,
    status=OrderStatus.PENDING,
    timestamp=datetime.now()
)

# Create sell limit order
limit_order = Order(
    side=OrderSide.SELL,
    quantity=500.0,
    type=OrderType.LIMIT,
    price=1.2600,
    stop_loss=None,
    take_profit=None,
    status=OrderStatus.PENDING,
    timestamp=datetime.now()
)
```

### BacktestReport

Contains comprehensive backtest results and performance metrics.

```python
@dataclass
class BacktestReport:
    starting_cash: np.float64
    final_cash: np.float64
    PnlRecord: pd.Series
    orders: list[Order]
```

#### Fields

##### `starting_cash: np.float64`
Initial capital at start of backtest.

##### `final_cash: np.float64`
Final capital at end of backtest.

##### `PnlRecord: pd.Series`
Time series of portfolio value over backtest period.

##### `orders: list[Order]`
List of all executed orders during backtest.

#### Properties

##### `periods_per_year`

Calculate the number of periods per year based on data frequency.

```python
@property
def periods_per_year(self):
    """Infer periods per year from data frequency"""
    return _infer_periods_per_year(self.PnlRecord.index, 252 * 24 * 60)
```

**Usage:**
```python
# Get periods per year for annualization
periods = report.periods_per_year
annual_return = (report.final_cash / report.starting_cash) ** (periods / len(report.PnlRecord)) - 1
```

##### `total_return`

Calculate total return as percentage.

```python
@property
def total_return(self):
    """Calculate total return percentage"""
    if self.starting_cash == 0:
        return 0.0
    return (self.final_cash - self.starting_cash) / self.starting_cash
```

**Usage:**
```python
# Get total return
total_return_pct = report.total_return
print(f"Total Return: {total_return_pct:.2%}")
```

##### `sharpe`

Calculate Sharpe ratio.

```python
@property
def sharpe(self):
    """Calculate Sharpe ratio"""
    equity = self.PnlRecord.astype(float)
    returns = equity.pct_change().dropna()

    annual_rf = 0.04  # 4% annual risk-free rate
    rf_per_period = annual_rf / self.periods_per_year

    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return float("nan")

    excess = returns - rf_per_period
    mean = excess.mean()
    vol = excess.std(ddof=1)

    return float((mean / vol) * (self.periods_per_year ** 0.5))
```

**Usage:**
```python
# Get Sharpe ratio
sharpe_ratio = report.sharpe
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
```

##### `max_drawdown`

Calculate maximum drawdown.

```python
@property
def max_drawdown(self):
    """Calculate maximum drawdown"""
    equity = self.PnlRecord.astype(float)
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(abs(drawdown.min()))
```

**Usage:**
```python
# Get maximum drawdown
max_dd = report.max_drawdown
print(f"Maximum Drawdown: {max_dd:.2%}")
```

#### String Representation

```python
def __str__(self) -> str:
    """Return formatted string representation"""
    # Returns comprehensive performance summary
    pass
```

**Usage:**
```python
# Print comprehensive backtest results
print(report)
```

## Utility Functions

### `max_drawdown(equity: pd.Series) -> float`

Calculate maximum drawdown from equity curve.

```python
def max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown from equity series"""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(abs(drawdown.min()))
```

**Parameters:**
- `equity` (pd.Series): Equity curve time series

**Returns:**
- `float`: Maximum drawdown as positive percentage

**Usage:**
```python
import pandas as pd

# Calculate maximum drawdown
equity_curve = pd.Series([100, 102, 98, 105, 95, 110])
max_dd = max_drawdown(equity_curve)
print(f"Maximum Drawdown: {max_dd:.2%}")
```

### `_infer_periods_per_year(index: pd.Index, default: int = 252 * 24 * 60) -> int`

Infer the number of periods per year from data frequency.

```python
def _infer_periods_per_year(index: pd.Index, default: int = 252 * 24 * 60) -> int:
    """Infer periods per year from datetime index"""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return default

    dt = np.diff(index.values).astype("timedelta64[s]").astype(float)
    if not np.isfinite(dt).any():
        return default

    med_sec = np.median(dt[dt > 0])
    if not np.isfinite(med_sec) or med_sec <= 0:
        return default

    periods_per_day = 86400.0 / med_sec
    return int(round(252 * periods_per_day))
```

**Parameters:**
- `index` (pd.Index): Datetime index to analyze
- `default` (int): Default value for uncertain frequencies

**Returns:**
- `int`: Inferred periods per year

**Usage:**
```python
# Infer periods per year from data
periods = _infer_periods_per_year(data_source.Index)
print(f"Data frequency: {periods} periods per year")
```

### `same_sign(num1, num2)`

Check if two numbers have the same sign.

```python
def same_sign(num1, num2):
    """Check if two numbers have the same sign"""
    if (num1 > 0 and num2 > 0):
        return True
    elif (num1 < 0 and num2 < 0):
        return True
    return False
```

**Parameters:**
- `num1`: First number
- `num2`: Second number

**Returns:**
- `bool`: True if both numbers have the same sign

**Usage:**
```python
# Check position direction consistency
old_position = 100
new_position = 150
if same_sign(old_position, new_position):
    print("Position direction unchanged")
```

## Complete Models Example

```python
from quantex import *
from datetime import datetime
import pandas as pd
import numpy as np

# Demonstrate all data models and enums

# 1. Commission Types
print("=== Commission Types ===")
percentage_commission = CommissionType.PERCENTAGE
cash_commission = CommissionType.CASH

print(f"Percentage commission: {percentage_commission}")
print(f"Cash commission: {cash_commission}")

# 2. Order Sides and Types
print("\n=== Order Sides and Types ===")
buy_side = OrderSide.BUY
sell_side = OrderSide.SELL
market_type = OrderType.MARKET
limit_type = OrderType.LIMIT

print(f"Buy side: {buy_side}, Value: {buy_side.value}")
print(f"Market order type: {market_type}, Value: {market_type.value}")

# 3. Order Status
print("\n=== Order Status ===")
pending_status = OrderStatus.PENDING
active_status = OrderStatus.ACTIVE
complete_status = OrderStatus.COMPLETE

print(f"Pending status: {pending_status}")
print(f"Active status: {active_status}")

# 4. Create sample orders
print("\n=== Creating Orders ===")

# Market buy order
market_buy = Order(
    side=OrderSide.BUY,
    quantity=1000.0,
    type=OrderType.MARKET,
    price=None,
    stop_loss=1.2400,
    take_profit=1.2700,
    status=OrderStatus.PENDING,
    timestamp=datetime.now()
)

# Limit sell order
limit_sell = Order(
    side=OrderSide.SELL,
    quantity=500.0,
    type=OrderType.LIMIT,
    price=1.2600,
    stop_loss=None,
    take_profit=None,
    status=OrderStatus.PENDING,
    timestamp=datetime.now()
)

print(f"Market buy order: {market_buy.side.name} {market_buy.quantity} units")
print(f"Limit sell order: {limit_sell.side.name} {limit_sell.quantity} units at {limit_sell.price}")

# 5. Create sample backtest report
print("\n=== Backtest Report ===")

# Sample PnL data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
pnl_values = np.linspace(10000, 15000, 100)  # 50% return
pnl_series = pd.Series(pnl_values, index=dates)

# Sample orders
sample_orders = [market_buy, limit_sell]

# Create report
report = BacktestReport(
    starting_cash=10000.0,
    final_cash=15000.0,
    PnlRecord=pnl_series,
    orders=sample_orders
)

print(f"Starting cash: ${report.starting_cash:,.2f}")
print(f"Final cash: ${report.final_cash:,.2f}")
print(f"Total return: {report.total_return:.2%}")
print(f"Sharpe ratio: {report.sharpe:.2f}")
print(f"Max drawdown: {report.max_drawdown:.2%}")
print(f"Total trades: {len(report.orders)}")

# 6. Utility functions
print("\n=== Utility Functions ===")

# Test max_drawdown calculation
sample_equity = pd.Series([100, 110, 105, 115, 108, 120, 112, 125])
calculated_dd = max_drawdown(sample_equity)
print(f"Calculated max drawdown: {calculated_dd:.2%}")

# Test periods per year inference
sample_index = pd.date_range('2023-01-01', periods=100, freq='H')  # Hourly data
periods_per_year = _infer_periods_per_year(sample_index)
print(f"Inferred periods per year: {periods_per_year}")

# Test same_sign function
print(f"Same sign (10, 15): {same_sign(10, 15)}")
print(f"Same sign (10, -15): {same_sign(10, -15)}")
print(f"Same sign (-10, -15): {same_sign(-10, -15)}")
```

## Model Validation

### Order Validation

```python
def validate_order(order):
    """Validate order parameters"""

    # Validate side
    if not isinstance(order.side, OrderSide):
        raise ValueError("Invalid order side")

    # Validate quantity
    if not isinstance(order.quantity, (int, float)) or order.quantity <= 0:
        raise ValueError("Invalid order quantity")

    # Validate type
    if not isinstance(order.type, OrderType):
        raise ValueError("Invalid order type")

    # Validate price for limit orders
    if order.type == OrderType.LIMIT:
        if order.price is None or order.price <= 0:
            raise ValueError("Limit orders must have valid price")

    # Validate stop loss and take profit
    if order.stop_loss is not None and order.stop_loss <= 0:
        raise ValueError("Stop loss must be positive")

    if order.take_profit is not None and order.take_profit <= 0:
        raise ValueError("Take profit must be positive")

    # Validate status
    if not isinstance(order.status, OrderStatus):
        raise ValueError("Invalid order status")

    return True

# Usage
try:
    validate_order(market_buy)
    print("Market buy order is valid")
except ValueError as e:
    print(f"Order validation error: {e}")
```

### BacktestReport Validation

```python
def validate_backtest_report(report):
    """Validate backtest report consistency"""

    # Check cash values
    if report.starting_cash <= 0:
        raise ValueError("Starting cash must be positive")

    if report.final_cash < 0:
        print("Warning: Final cash is negative")

    # Check PnL record
    if len(report.PnlRecord) == 0:
        raise ValueError("PnL record is empty")

    if report.PnlRecord.isnull().any():
        raise ValueError("PnL record contains NaN values")

    # Check orders
    for order in report.orders:
        if not isinstance(order, Order):
            raise ValueError("Invalid order in report")

        validate_order(order)

    # Check consistency
    if not np.isclose(report.PnlRecord.iloc[-1], report.final_cash):
        print("Warning: Final PnL doesn't match final cash")

    return True
```

## Best Practices

### 1. Enum Usage

```python
# Always use enums for type safety
def place_order(side, quantity, order_type=OrderType.MARKET):
    """Place order with proper enum usage"""

    if side not in OrderSide:
        raise ValueError(f"Invalid side: {side}")

    if order_type not in OrderType:
        raise ValueError(f"Invalid order type: {order_type}")

    # Create order with validated enums
    order = Order(
        side=side,
        quantity=quantity,
        type=order_type,
        price=None,  # Market order
        stop_loss=None,
        take_profit=None,
        status=OrderStatus.PENDING,
        timestamp=datetime.now()
    )

    return order
```

### 2. Data Class Best Practices

```python
# Use dataclasses for immutable data structures
from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class ImmutableOrder:
    """Immutable order representation"""

    side: OrderSide
    quantity: float
    type: OrderType
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)

# Usage
immutable_order = ImmutableOrder(
    side=OrderSide.BUY,
    quantity=1000.0,
    type=OrderType.MARKET
)

# immutable_order.quantity = 2000  # This would raise an error
```

### 3. Error Handling

```python
def safe_enum_conversion(value, enum_class, default=None):
    """Safely convert value to enum"""

    try:
        return enum_class(value)
    except ValueError:
        if default is not None:
            return default
        raise ValueError(f"Invalid value for {enum_class.__name__}: {value}")

# Usage
side = safe_enum_conversion("BUY", OrderSide)
print(f"Converted side: {side}")

invalid_side = safe_enum_conversion("INVALID", OrderSide, OrderSide.BUY)
print(f"Default side: {invalid_side}")
```

## Integration with Other Modules

### Strategy Integration

```python
class ModelAwareStrategy(Strategy):
    """Strategy that properly uses data models"""

    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

    def next(self):
        # Use enums for type safety
        if self.should_buy():
            # Create properly typed order
            order = Order(
                side=OrderSide.BUY,
                quantity=self.calculate_quantity(),
                type=OrderType.MARKET,
                price=None,
                stop_loss=self.calculate_stop_loss(),
                take_profit=self.calculate_take_profit(),
                status=OrderStatus.PENDING,
                timestamp=self.data['EURUSD'].Index[self.data['EURUSD'].current_index]
            )

            # Execute through broker
            self.positions['EURUSD'].buy(
                quantity=order.quantity,
                limit=order.price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit
            )
```

### Backtester Integration

```python
def analyze_model_performance(report):
    """Analyze performance using data models"""

    # Use BacktestReport methods
    print(f"Performance Summary:")
    print(f"  Starting Cash: ${report.starting_cash:,.2f}")
    print(f"  Final Cash: ${report.final_cash:,.2f}")
    print(f"  Total Return: {report.total_return:.2%}")
    print(f"  Sharpe Ratio: {report.sharpe:.2f}")
    print(f"  Max Drawdown: {report.max_drawdown:.2%}")

    # Analyze orders using Order model
    buy_orders = [o for o in report.orders if o.side == OrderSide.BUY]
    sell_orders = [o for o in report.orders if o.side == OrderSide.SELL]

    print(f"\nOrder Analysis:")
    print(f"  Buy Orders: {len(buy_orders)}")
    print(f"  Sell Orders: {len(sell_orders)}")
    print(f"  Market Orders: {len([o for o in report.orders if o.type == OrderType.MARKET])}")
    print(f"  Limit Orders: {len([o for o in report.orders if o.type == OrderType.LIMIT])}")

    return {
        'total_return': report.total_return,
        'sharpe_ratio': report.sharpe,
        'max_drawdown': report.max_drawdown,
        'total_trades': len(report.orders),
        'buy_trades': len(buy_orders),
        'sell_trades': len(sell_orders)
    }
```

## Related Documentation

- **[Strategy API](../../quantex.strategy.md)**: Learn how models integrate with strategies
- **[Backtester API](../../quantex.backtest.md)**: Understand backtest result models
- **[Execution API](../../quantex.execution.md)**: See how orders are executed
- **[Usage Guides](../../../usage/)**: Learn how to use these models in practice