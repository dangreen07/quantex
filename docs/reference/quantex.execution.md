# Execution API Reference

This page provides comprehensive API documentation for QuantEx's order execution and broker simulation system.

## Broker Class

The `Broker` class handles order execution, position management, and portfolio tracking.

```python
class Broker:
    def __init__(self, source: DataSource):
        self.position: np.float64 = np.float64(0)
        self.position_avg_price: np.float64 = np.float64(0)
        self.cash: np.float64 = np.float64(10_000)
        self.commission: np.float64 = np.float64(0.002)
        self.commission_type: CommissionType = CommissionType.PERCENTAGE
        self.lot_size: int = 1
        self.share_decimals = 1
        self.orders: list[Order] = []
        self.complete_orders = []
        self._i = 0
        self.source = source
        self.PnLRecord = pd.Series([self.cash] * len(self.source.data['Close']),
                                  index=self.source.data['Close'].index,
                                  dtype=np.float64)
```

### Constructor Parameters

#### `source: DataSource`
The data source providing market data for execution.

### Properties

#### `position: np.float64`
Current position size (positive for long, negative for short).

#### `position_avg_price: np.float64`
Average price of current position.

#### `cash: np.float64`
Available cash for trading.

#### `commission: np.float64`
Commission rate for trades.

#### `commission_type: CommissionType`
Commission calculation method.

#### `PnLRecord: pd.Series`
Time series of portfolio value over time.

### Instance Methods

#### `buy(self, quantity: float = 1, limit: np.float64 | None = None, amount: np.float64 | None = None, stop_loss: np.float64 | None = None, take_profit: np.float64 | None = None)`

Place a buy order.

```python
def buy(self, quantity: float = 1, limit: np.float64 | None = None,
        amount: np.float64 | None = None, stop_loss: np.float64 | None = None,
        take_profit: np.float64 | None = None):
    """Place a buy order"""
    pass
```

**Parameters:**
- `quantity` (float): Fraction of cash to use (0-1) or specific quantity
- `limit` (float, optional): Limit price for order
- `amount` (float, optional): Specific amount to buy
- `stop_loss` (float, optional): Stop loss price
- `take_profit` (float, optional): Take profit price

**Usage:**
```python
# Buy with 50% of available cash at market
broker.buy(0.5)

# Buy with limit price
broker.buy(0.5, limit=1.2500)

# Buy specific amount
broker.buy(amount=10000)

# Buy with stop loss and take profit
broker.buy(0.3, stop_loss=1.2400, take_profit=1.2700)
```

#### `sell(self, quantity: float = 1, limit: np.float64 | None = None, amount: np.float64 | None = None, stop_loss: np.float64 | None = None, take_profit: np.float64 | None = None)`

Place a sell order.

```python
def sell(self, quantity: float = 1, limit: np.float64 | None = None,
         amount: np.float64 | None = None, stop_loss: np.float64 | None = None,
         take_profit: np.float64 | None = None):
    """Place a sell order"""
    pass
```

**Parameters:**
- `quantity` (float): Fraction of position to sell (0-1) or specific quantity
- `limit` (float, optional): Limit price for order
- `amount` (float, optional): Specific amount to sell
- `stop_loss` (float, optional): Stop loss price
- `take_profit` (float, optional): Take profit price

**Usage:**
```python
# Sell 30% of current position at market
broker.sell(0.3)

# Sell with limit price
broker.sell(0.3, limit=1.2600)

# Sell specific amount
broker.sell(amount=5000)
```

#### `close(self)`

Close the entire current position.

```python
def close(self):
    """Close entire position"""
    pass
```

**Usage:**
```python
# Close entire position at market
broker.close()
```

#### `_iterate(self, current_index: int)`

Process orders and update position for current time step.

```python
def _iterate(self, current_index: int):
    """Process orders for current time step"""
    pass
```

**Parameters:**
- `current_index` (int): Current time step index

## Order Class

Represents a trading order with all execution parameters.

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

### Fields

#### `side: OrderSide`
Order direction (BUY or SELL).

#### `quantity: np.float64`
Order quantity (positive number).

#### `type: OrderType`
Order type (MARKET or LIMIT).

#### `price: np.float64 | None`
Limit price for limit orders (None for market orders).

#### `stop_loss: np.float64 | None`
Stop loss price (None if not set).

#### `take_profit: np.float64 | None`
Take profit price (None if not set).

#### `status: OrderStatus`
Current order status.

#### `timestamp: datetime`
Order creation timestamp.

## Enums

### OrderSide

```python
class OrderSide(Enum):
    BUY = 1
    SELL = -1
```

### OrderType

```python
class OrderType(Enum):
    MARKET = 0
    LIMIT = 1
```

### OrderStatus

```python
class OrderStatus(Enum):
    ACTIVE = 0    # Order executed with SL/TP conditions
    COMPLETE = 1  # Order fully executed
    PENDING = 2   # Order waiting for execution
```

### CommissionType

```python
class CommissionType(Enum):
    PERCENTAGE = 0
    CASH = 1
```

## Complete Execution Example

```python
from quantex import Strategy, SimpleBacktester, CSVDataSource, CommissionType
import numpy as np

class ExecutionStrategy(Strategy):
    """Strategy demonstrating various execution techniques"""

    def __init__(self, execution_style='mixed'):
        super().__init__()
        self.execution_style = execution_style
        self.position_count = 0
        self.max_positions = 3

    def init(self):
        # Load data
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Execution parameters
        self.base_position_size = 0.2
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04

    def next(self):
        current_price = self.data['EURUSD'].CClose
        position = self.positions['EURUSD']

        # Demonstrate different execution styles
        if self.execution_style == 'market':
            self.market_execution_example()
        elif self.execution_style == 'limit':
            self.limit_execution_example()
        elif self.execution_style == 'risk_managed':
            self.risk_managed_execution_example()
        elif self.execution_style == 'mixed':
            self.mixed_execution_example()

    def market_execution_example(self):
        """Simple market order execution"""

        if self.should_enter_position():
            # Simple market buy
            self.positions['EURUSD'].buy(self.base_position_size)

        if self.should_exit_position():
            # Simple market sell
            self.positions['EURUSD'].sell(self.base_position_size)

    def limit_execution_example(self):
        """Limit order execution"""

        current_price = self.data['EURUSD'].CClose

        if self.should_buy_dip():
            # Buy limit order below current price
            limit_price = current_price * 0.998  # 0.2% below current
            self.positions['EURUSD'].buy(self.base_position_size, limit=limit_price)

        if self.should_sell_rally():
            # Sell limit order above current price
            limit_price = current_price * 1.002  # 0.2% above current
            self.positions['EURUSD'].sell(self.base_position_size, limit=limit_price)

    def risk_managed_execution_example(self):
        """Execution with risk management"""

        current_price = self.data['EURUSD'].CClose

        if self.should_enter_with_risk_management():
            # Calculate risk management levels
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)

            # Enter position with risk management
            self.positions['EURUSD'].buy(
                self.base_position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

    def mixed_execution_example(self):
        """Mixed execution strategies"""

        current_price = self.data['EURUSD'].CClose
        position = self.positions['EURUSD']

        # Use different strategies based on market conditions
        if self.is_trending():
            self.trend_execution()
        else:
            self.mean_reversion_execution()

    def trend_execution(self):
        """Trend-following execution"""

        if self.is_bullish_trend() and position.position <= 0:
            # Enter long position
            self.positions['EURUSD'].buy(self.base_position_size)

        elif self.is_bearish_trend() and position.position >= 0:
            # Enter short position
            self.positions['EURUSD'].sell(self.base_position_size)

    def mean_reversion_execution(self):
        """Mean-reversion execution"""

        current_price = self.data['EURUSD'].CClose

        if self.is_overbought():
            # Sell overbought conditions
            self.positions['EURUSD'].sell(self.base_position_size)

        elif self.is_oversold():
            # Buy oversold conditions
            self.positions['EURUSD'].buy(self.base_position_size)

    # Placeholder methods for demonstration
    def should_enter_position(self): return False
    def should_exit_position(self): return False
    def should_buy_dip(self): return False
    def should_sell_rally(self): return False
    def should_enter_with_risk_management(self): return False
    def is_trending(self): return False
    def is_bullish_trend(self): return False
    def is_bearish_trend(self): return False
    def is_overbought(self): return False
    def is_oversold(self): return False

# Execution examples
strategies = {
    'market': ExecutionStrategy('market'),
    'limit': ExecutionStrategy('limit'),
    'risk_managed': ExecutionStrategy('risk_managed'),
    'mixed': ExecutionStrategy('mixed')
}

for name, strategy in strategies.items():
    backtester = SimpleBacktester(strategy, cash=10000)
    report = backtester.run(progress_bar=False)
    print(f"{name} Strategy Results:")
    print(f"  Total Return: {report.total_return:.2%}")
    print(f"  Sharpe Ratio: {report.sharpe:.2f}")
    print(f"  Max Drawdown: {report.max_drawdown:.2%}")
    print(f"  Total Trades: {len(report.orders)}")
    print()
```

## Advanced Execution Techniques

### Dynamic Position Sizing

```python
class DynamicPositionStrategy(Strategy):
    """Strategy with dynamic position sizing"""

    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Position sizing parameters
        self.base_size = 0.1
        self.volatility_adjustment = True
        self.max_position = 0.5

    def calculate_dynamic_size(self):
        """Calculate position size based on market conditions"""

        if not self.volatility_adjustment:
            return self.base_size

        # Calculate recent volatility
        recent_prices = self.data['EURUSD'].Close[-20:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)

        # Adjust position size inversely to volatility
        # Lower volatility = larger position, higher volatility = smaller position
        adjusted_size = self.base_size / (1 + volatility * 10)

        return min(adjusted_size, self.max_position)

    def next(self):
        if self.should_trade():
            position_size = self.calculate_dynamic_size()

            # Execute with dynamic sizing
            self.positions['EURUSD'].buy(position_size)
```

### Order Management System

```python
class OrderManager:
    """Advanced order management system"""

    def __init__(self, broker):
        self.broker = broker
        self.pending_orders = []
        self.active_orders = []
        self.completed_orders = []

    def place_smart_order(self, side, quantity, order_type='market',
                         limit_price=None, stop_loss=None, take_profit=None):
        """Place order with intelligent defaults"""

        # Validate order parameters
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        if limit_price is not None and limit_price <= 0:
            raise ValueError("Limit price must be positive")

        # Place order based on type
        if order_type == 'market':
            if side == 'buy':
                self.broker.buy(quantity, stop_loss=stop_loss, take_profit=take_profit)
            else:
                self.broker.sell(quantity, stop_loss=stop_loss, take_profit=take_profit)

        elif order_type == 'limit':
            if side == 'buy':
                self.broker.buy(quantity, limit=limit_price, stop_loss=stop_loss, take_profit=take_profit)
            else:
                self.broker.sell(quantity, limit=limit_price, stop_loss=stop_loss, take_profit=take_profit)

        # Track order
        self.track_order()

    def track_order(self):
        """Track order status and update internal lists"""
        # Implementation would sync with broker.orders
        pass

    def get_order_status(self, order_id):
        """Get status of specific order"""
        # Implementation
        return None

    def cancel_order(self, order_id):
        """Cancel pending order"""
        # Note: Current QuantEx implementation doesn't support cancellation
        # This is for future enhancement
        pass
```

### Risk Management Integration

```python
class RiskManagedExecution(Strategy):
    """Strategy with integrated risk management"""

    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Risk management settings
        self.max_position_size = 0.2
        self.max_portfolio_risk = 0.05  # 5% max loss
        self.trailing_stop_pct = 0.02

    def next(self):
        current_price = self.data['EURUSD'].CClose
        position = self.positions['EURUSD']

        # Pre-execution risk checks
        if not self.passes_risk_checks():
            return

        # Execute trades with risk management
        if self.should_enter_position():
            # Calculate position size based on risk
            position_size = self.calculate_risk_adjusted_size(current_price)

            # Set stop loss based on risk tolerance
            stop_loss = current_price * (1 - self.max_portfolio_risk / position_size)

            self.positions['EURUSD'].buy(position_size, stop_loss=stop_loss)

        # Manage existing positions
        self.manage_open_positions()

    def passes_risk_checks(self):
        """Check if new position meets risk criteria"""

        position = self.positions['EURUSD']

        # Check maximum position size
        max_value = position.cash * self.max_position_size
        current_value = abs(position.position) * self.data['EURUSD'].CClose

        if current_value >= max_value:
            return False

        # Check portfolio heat (unrealized losses)
        unrealized_pnl = self.calculate_unrealized_pnl()
        if unrealized_pnl < -position.cash * self.max_portfolio_risk:
            return False

        return True

    def calculate_risk_adjusted_size(self, current_price):
        """Calculate position size based on risk tolerance"""

        # Kelly Criterion or similar risk-adjusted sizing
        # Simplified implementation
        return self.max_position_size

    def manage_open_positions(self):
        """Manage existing positions with trailing stops, etc."""

        position = self.positions['EURUSD']

        if position.position > 0:  # Long position
            current_price = self.data['EURUSD'].CClose
            entry_price = position.position_avg_price

            # Update trailing stop if profitable
            if current_price > entry_price * 1.02:  # 2% profit
                new_stop = current_price * (1 - self.trailing_stop_pct)
                # Implementation would update stop loss
```

## Commission and Cost Analysis

### Commission Calculation Methods

```python
class CommissionCalculator:
    """Handle different commission calculation methods"""

    @staticmethod
    def calculate_percentage_commission(quantity, price, commission_rate):
        """Calculate percentage-based commission"""
        trade_value = quantity * price
        return trade_value * commission_rate

    @staticmethod
    def calculate_cash_commission(quantity, price, commission_amount):
        """Calculate fixed cash commission"""
        return commission_amount

    @staticmethod
    def calculate_commission(broker, quantity, price):
        """Calculate commission based on broker settings"""

        if broker.commission_type == CommissionType.PERCENTAGE:
            return CommissionCalculator.calculate_percentage_commission(
                quantity, price, broker.commission
            )
        elif broker.commission_type == CommissionType.CASH:
            return CommissionCalculator.calculate_cash_commission(
                quantity, price, broker.commission
            )
        else:
            return 0.0

# Usage in analysis
def analyze_commission_impact(report, broker):
    """Analyze commission impact on performance"""

    total_commissions = 0
    total_trade_value = 0

    for order in report.orders:
        if order.status == OrderStatus.COMPLETE:
            quantity = order.quantity
            price = order.price or 0  # Market orders might not have price set

            commission = CommissionCalculator.calculate_commission(broker, quantity, price)
            total_commissions += commission
            total_trade_value += quantity * price

    print(f"Total Commissions: ${total_commissions:.2f}")
    print(f"Total Trade Value: ${total_trade_value:.2f}")
    print(f"Commission Rate: {total_commissions/total_trade_value:.4f}")
    print(f"Commission Impact: {total_commissions/report.final_cash:.2%}")

    return total_commissions
```

## Execution Quality Metrics

### Slippage Analysis

```python
def analyze_execution_quality(report, data_source):
    """Analyze execution quality metrics"""

    execution_quality = {
        'slippage': [],
        'fill_rate': 0,
        'average_fill_time': 0
    }

    # Analyze each order
    for order in report.orders:
        if order.status == OrderStatus.COMPLETE:
            # Calculate slippage (difference between order price and execution price)
            order_price = order.price
            execution_price = data_source.Close[data_source.current_index]

            if order_price is not None:
                slippage = abs(execution_price - order_price) / order_price
                execution_quality['slippage'].append(slippage)

    # Calculate metrics
    if execution_quality['slippage']:
        avg_slippage = np.mean(execution_quality['slippage'])
        max_slippage = np.max(execution_quality['slippage'])

        print(f"Average Slippage: {avg_slippage:.4f}")
        print(f"Maximum Slippage: {max_slippage:.4f}")
        print(f"Slippage Std: {np.std(execution_quality['slippage']):.4f}")

    return execution_quality
```

### Market Impact Analysis

```python
def analyze_market_impact(report, data_source, lookback=10):
    """Analyze market impact of trades"""

    impact_analysis = []

    for order in report.orders:
        if order.status == OrderStatus.COMPLETE:
            order_time = order.timestamp

            # Find price movement around order
            try:
                order_idx = data_source.Index.get_loc(order_time)
                pre_order_prices = data_source.Close[max(0, order_idx-lookback):order_idx]
                post_order_prices = data_source.Close[order_idx:min(len(data_source.Close), order_idx+lookback)]

                # Calculate price impact
                pre_avg = np.mean(pre_order_prices)
                post_avg = np.mean(post_order_prices)

                price_impact = (post_avg - pre_avg) / pre_avg

                impact_analysis.append({
                    'timestamp': order_time,
                    'side': order.side,
                    'quantity': order.quantity,
                    'price_impact': price_impact
                })

            except:
                continue

    # Summarize impact
    buy_impacts = [trade['price_impact'] for trade in impact_analysis if trade['side'] == OrderSide.BUY]
    sell_impacts = [trade['price_impact'] for trade in impact_analysis if trade['side'] == OrderSide.SELL]

    print(f"Buy Impact - Avg: {np.mean(buy_impacts):.6f}, Count: {len(buy_impacts)}")
    print(f"Sell Impact - Avg: {np.mean(sell_impacts):.6f}, Count: {len(sell_impacts)}")

    return impact_analysis
```

## Best Practices

### 1. Order Validation

```python
def validate_order_parameters(broker, quantity, price=None, stop_loss=None, take_profit=None):
    """Validate order parameters before submission"""

    # Check quantity
    if quantity <= 0:
        raise ValueError("Order quantity must be positive")

    if quantity > 1.0 and price is None:
        # If quantity > 1, it might be intended as amount rather than fraction
        print("Warning: Large quantity without price - check if amount is intended")

    # Check prices
    current_price = broker.source.CClose

    if price is not None:
        if price <= 0:
            raise ValueError("Order price must be positive")

        # Check if limit price makes sense
        price_distance = abs(price - current_price) / current_price
        if price_distance > 0.1:  # 10% away from current price
            print(f"Warning: Limit price {price_distance:.1%} away from current price")

    # Validate stop loss and take profit
    if stop_loss is not None and stop_loss <= 0:
        raise ValueError("Stop loss must be positive")

    if take_profit is not None and take_profit <= 0:
        raise ValueError("Take profit must be positive")

    # Check stop loss / take profit logic
    if stop_loss is not None and take_profit is not None:
        if stop_loss >= take_profit:
            raise ValueError("Stop loss must be below take profit for buy orders")

    return True
```

### 2. Position Size Management

```python
def calculate_optimal_position_size(broker, risk_tolerance=0.02, volatility=None):
    """Calculate optimal position size based on risk tolerance"""

    available_cash = broker.cash
    current_price = broker.source.CClose

    if volatility is None:
        # Estimate volatility from recent data
        recent_prices = broker.source.Close[-20:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)

    # Position size based on risk tolerance
    # Risk = position_size * price * volatility
    # position_size = risk_tolerance / (price * volatility)

    max_risk_amount = available_cash * risk_tolerance
    position_size = max_risk_amount / (current_price * volatility)

    # Cap position size
    max_position_size = 0.5  # Maximum 50% of capital
    position_size = min(position_size, max_position_size)

    return position_size
```

### 3. Execution Timing

```python
def optimize_execution_timing(strategy):
    """Optimize when orders are executed during the bar"""

    # Option 1: Execute at open
    def execute_at_open(self):
        # Use opening price for execution
        pass

    # Option 2: Execute at close
    def execute_at_close(self):
        # Use closing price for execution
        pass

    # Option 3: Execute on signal
    def execute_on_signal(self):
        # Execute immediately when signal is generated
        pass
```

## Error Handling

### Execution Errors

```python
class ExecutionError(Exception):
    """Base exception for execution errors"""
    pass

class InsufficientFundsError(ExecutionError):
    """Raised when insufficient funds for order"""
    pass

class InvalidOrderError(ExecutionError):
    """Raised when order parameters are invalid"""
    pass

class MarketHoursError(ExecutionError):
    """Raised when trading outside market hours"""
    pass

# Error handling in broker
def safe_order_execution(broker, order_func, *args, **kwargs):
    """Safely execute orders with error handling"""

    try:
        # Validate order before execution
        validate_order_parameters(broker, *args, **kwargs)

        # Execute order
        return order_func(*args, **kwargs)

    except InsufficientFundsError:
        print("Insufficient funds for order")
        return None

    except InvalidOrderError as e:
        print(f"Invalid order: {e}")
        return None

    except Exception as e:
        print(f"Unexpected execution error: {e}")
        return None
```

## Related Documentation

- **[Usage Guide](../../../usage/execution.md)**: Learn how to use the execution system
- **[Strategy API](../../quantex.strategy.md)**: Understand strategy integration
- **[Backtester API](../../quantex.backtest.md)**: See how execution integrates with backtesting
- **[Data Sources API](../../quantex.sources.md)**: Work with market data for execution