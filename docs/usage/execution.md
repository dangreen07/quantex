# Execution Simulator Guide

This guide covers QuantEx's order execution system and broker simulation, which realistically models how trades are executed in live markets.

## Overview

QuantEx's execution simulator provides realistic order processing including:

- **Order Types**: Market orders, limit orders, stop losses, take profits
- **Order States**: Pending, active, and completed orders
- **Position Tracking**: Real-time position and PnL calculation
- **Commission Handling**: Realistic transaction cost modeling
- **Risk Management**: Automatic stop loss and take profit execution

## Basic Order Execution

### Market Orders

```python
def next(self):
    # Market buy order
    self.positions['EURUSD'].buy(0.5)  # Buy with 50% of available cash

    # Market sell order
    self.positions['EURUSD'].sell(0.3)  # Sell 30% of current position

    # Close entire position
    self.positions['EURUSD'].close()
```

### Order Parameters

```python
# Buy with specific dollar amount
self.positions['EURUSD'].buy(amount=1000)  # Buy $1000 worth

# Sell specific quantity
self.positions['EURUSD'].sell(amount=10000)  # Sell 10,000 units

# Percentage-based sizing
self.positions['EURUSD'].buy(0.25)  # Use 25% of available cash
```

## Advanced Order Types

### Limit Orders

```python
def next(self):
    current_price = self.data['EURUSD'].CClose

    # Buy limit order - only execute if price drops to target
    target_buy_price = current_price * 0.995  # 0.5% below current price
    self.positions['EURUSD'].buy(0.5, limit=target_buy_price)

    # Sell limit order - only execute if price rises to target
    target_sell_price = current_price * 1.01  # 1% above current price
    self.positions['EURUSD'].sell(0.3, limit=target_sell_price)
```

### Stop Loss Orders

```python
def next(self):
    # Buy with stop loss
    entry_price = self.data['EURUSD'].CClose
    stop_loss_price = entry_price * 0.98  # 2% stop loss

    self.positions['EURUSD'].buy(0.5, stop_loss=stop_loss_price)

    # Sell with stop loss (for short positions)
    self.positions['EURUSD'].sell(0.3, stop_loss=entry_price * 1.02)
```

### Take Profit Orders

```python
def next(self):
    # Buy with take profit
    entry_price = self.data['EURUSD'].CClose
    take_profit_price = entry_price * 1.03  # 3% profit target

    self.positions['EURUSD'].buy(0.5, take_profit=take_profit_price)

    # Multiple targets
    self.positions['EURUSD'].sell(0.3, take_profit=take_profit_price)
```

## Order States and Lifecycle

### Understanding Order States

Orders in QuantEx can have three states:

- **PENDING**: Order created but not yet executed
- **ACTIVE**: Order executed but has stop loss/take profit conditions
- **COMPLETE**: Order fully executed with no remaining conditions

```python
def next(self):
    # Check order states
    orders = self.positions['EURUSD'].orders

    for order in orders:
        print(f"Order {order.side.name}: {order.status.name}")

        if order.status == OrderStatus.PENDING:
            print(f"  - Waiting for execution at {order.price}")
        elif order.status == OrderStatus.ACTIVE:
            print(f"  - Position open with SL/TP conditions")
        elif order.status == OrderStatus.COMPLETE:
            print(f"  - Order completed")
```

### Order Management

```python
class OrderManager:
    def __init__(self, strategy):
        self.strategy = strategy

    def get_pending_orders(self, symbol):
        """Get all pending orders for a symbol"""
        return [order for order in self.strategy.positions[symbol].orders
                if order.status == OrderStatus.PENDING]

    def get_active_orders(self, symbol):
        """Get all active orders for a symbol"""
        return [order for order in self.strategy.positions[symbol].orders
                if order.status == OrderStatus.ACTIVE]

    def cancel_order(self, symbol, order_id):
        """Cancel a specific order"""
        # Note: QuantEx doesn't currently support order cancellation
        # This is for illustration of the concept
        pass
```

## Position Management

### Position Information

```python
def next(self):
    position = self.positions['EURUSD']

    # Basic position info
    print(f"Current Position: {position.position}")
    print(f"Position Value: {position.position * self.data['EURUSD'].CClose}")
    print(f"Available Cash: ${position.cash:,.2f}")

    # Position status
    if position.position > 0:
        print("Currently LONG")
    elif position.position < 0:
        print("Currently SHORT")
    else:
        print("Currently FLAT")

    # Average entry price
    if position.position != 0:
        print(f"Average Entry Price: {position.position_avg_price}")
```

### Unrealized PnL

```python
def calculate_unrealized_pnl(self):
    """Calculate unrealized profit/loss"""

    position = self.positions['EURUSD']
    current_price = self.data['EURUSD'].CClose

    if position.position == 0:
        return 0.0

    # Calculate PnL based on position and current price
    entry_value = position.position * position.position_avg_price
    current_value = position.position * current_price

    return current_value - entry_value

def next(self):
    unrealized_pnl = self.calculate_unrealized_pnl()
    print(f"Unrealized PnL: ${unrealized_pnl:.2f}")
```

## Risk Management

### Automatic Stop Losses

```python
class RiskManagedStrategy(Strategy):
    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Risk management parameters
        self.max_position_size = 0.1  # Max 10% of capital per trade
        self.stop_loss_pct = 0.02     # 2% stop loss
        self.take_profit_pct = 0.04   # 4% take profit

    def next(self):
        if self.should_enter_trade():
            current_price = self.data['EURUSD'].CClose

            # Calculate stop loss and take profit levels
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)

            # Enter position with risk management
            self.positions['EURUSD'].buy(
                self.max_position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
```

### Dynamic Risk Management

```python
def next(self):
    position = self.positions['EURUSD']

    # Adjust stop loss based on profit
    if position.position > 0:
        entry_price = position.position_avg_price
        current_price = self.data['EURUSD'].CClose
        profit_pct = (current_price - entry_price) / entry_price

        # Trail stop loss if profitable
        if profit_pct > 0.02:  # If 2% profit
            new_stop = current_price * 0.99  # Trail at 1% below current

            # Update stop loss for existing position
            # Note: This would require modifying the broker implementation
            # For now, this is conceptual
```

## Commission and Fees

### Commission Calculation

```python
# Percentage-based commission (default)
position = self.positions['EURUSD']
commission_rate = 0.002  # 0.2%

# Commission is calculated as: quantity * price * commission_rate
trade_value = quantity * price
commission = trade_value * commission_rate

# Fixed cash commission
fixed_commission = 1.00  # $1 per trade
```

### Commission Impact Analysis

```python
def analyze_commission_impact(self, report):
    """Analyze how commissions affect performance"""

    total_trades = len(report.orders)
    total_commission = 0

    for order in report.orders:
        if order.status == OrderStatus.COMPLETE:
            # Calculate commission for this order
            commission = self.calculate_commission(order)
            total_commission += commission

    print(f"Total Trades: {total_trades}")
    print(f"Total Commissions: ${total_commission:.2f}")
    print(f"Average Commission per Trade: ${total_commission/total_trades:.2f}")
    print(f"Commission as % of Final Capital: {total_commission/report.final_cash:.2%}")

    return total_commission
```

## Order Execution Examples

### Simple Buy and Hold

```python
class BuyAndHoldStrategy(Strategy):
    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')
        self.entered = False

    def next(self):
        # Buy on first day and hold
        if not self.entered and len(self.data['EURUSD'].Close) > 1:
            self.positions['EURUSD'].buy(1.0)  # Use all available cash
            self.entered = True
```

### Mean Reversion Strategy

```python
class MeanReversionStrategy(Strategy):
    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Calculate moving average and standard deviation
        close_prices = self.data['EURUSD'].Close
        self.mean = self.Indicator(self.calculate_sma(close_prices, 20))
        self.std = self.Indicator(self.calculate_std(close_prices, 20))

    def next(self):
        if len(self.mean) < 2:
            return

        current_price = self.data['EURUSD'].CClose
        z_score = (current_price - self.mean[-1]) / self.std[-1]

        # Buy when price is significantly below mean
        if z_score < -2.0 and self.positions['EURUSD'].position <= 0:
            self.positions['EURUSD'].buy(0.5)

        # Sell when price is significantly above mean
        elif z_score > 2.0 and self.positions['EURUSD'].position >= 0:
            self.positions['EURUSD'].sell(0.5)

    def calculate_sma(self, prices, period):
        return pd.Series(prices).rolling(window=period).mean().values

    def calculate_std(self, prices, period):
        return pd.Series(prices).rolling(window=period).std().values
```

### Breakout Strategy

```python
class BreakoutStrategy(Strategy):
    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Track recent high/low
        self.lookback = 20
        self.highest_high = self.Indicator(self.calculate_highest_high(self.data['EURUSD'].High, self.lookback))
        self.lowest_low = self.Indicator(self.calculate_lowest_low(self.data['EURUSD'].Low, self.lookback))

    def next(self):
        if len(self.highest_high) < 1:
            return

        current_price = self.data['EURUSD'].CClose

        # Bullish breakout
        if (current_price > self.highest_high[-1] and
            self.positions['EURUSD'].position <= 0):
            self.positions['EURUSD'].buy(0.5)

        # Bearish breakout
        elif (current_price < self.lowest_low[-1] and
              self.positions['EURUSD'].position >= 0):
            self.positions['EURUSD'].sell(0.5)

    def calculate_highest_high(self, highs, period):
        return pd.Series(highs).rolling(window=period).max().values

    def calculate_lowest_low(self, lows, period):
        return pd.Series(lows).rolling(window=period).min().values
```

## Execution Quality Analysis

### Slippage Modeling

```python
# QuantEx doesn't currently model slippage, but here's how you could implement it
class SlippageAwareBroker(Broker):
    def __init__(self, *args, slippage_pct=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.slippage_pct = slippage_pct

    def execute_market_order(self, order):
        """Execute market order with slippage"""

        base_price = self.source.CClose

        # Add random slippage (simplified model)
        slippage = base_price * self.slippage_pct * (2 * random.random() - 1)
        execution_price = base_price + slippage

        # Execute order at slippage-adjusted price
        return self._execute_order_at_price(order, execution_price)
```

### Fill Rate Analysis

```python
def analyze_fill_rates(self, report):
    """Analyze order fill rates and execution quality"""

    total_orders = len(report.orders)
    filled_orders = len([o for o in report.orders if o.status == OrderStatus.COMPLETE])

    print(f"Total Orders: {total_orders}")
    print(f"Filled Orders: {filled_orders}")
    print(f"Fill Rate: {filled_orders/total_orders:.2%}")

    # Analyze fill rates by order type
    market_orders = [o for o in report.orders if o.type == OrderType.MARKET]
    limit_orders = [o for o in report.orders if o.type == OrderType.LIMIT]

    market_fills = len([o for o in market_orders if o.status == OrderStatus.COMPLETE])
    limit_fills = len([o for o in limit_orders if o.status == OrderStatus.COMPLETE])

    print(f"Market Order Fill Rate: {market_fills/len(market_orders):.2%}")
    print(f"Limit Order Fill Rate: {limit_fills/len(limit_orders):.2%}")
```

## Portfolio-Level Execution

### Multi-Symbol Execution

```python
class MultiSymbolStrategy(Strategy):
    def init(self):
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

        for symbol in symbols:
            data = CSVDataSource(f'data/{symbol}.csv')
            self.add_data(data, symbol)

    def next(self):
        # Allocate capital across symbols
        total_allocation = 0.9  # Use 90% of capital
        allocation_per_symbol = total_allocation / 3  # Equal allocation

        for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
            if self.should_trade(symbol):
                # Check if we have enough cash for this allocation
                required_cash = self.positions[symbol].cash * allocation_per_symbol

                if required_cash > 100:  # Minimum trade size
                    self.positions[symbol].buy(allocation_per_symbol)
```

### Position Sizing

```python
class PositionSizingStrategy(Strategy):
    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')

        # Position sizing parameters
        self.base_position_size = 0.1  # Base 10% position
        self.volatility_adjustment = True
        self.max_position = 0.5  # Maximum 50% position

    def calculate_position_size(self):
        """Calculate dynamic position size based on volatility"""

        if not self.volatility_adjustment:
            return self.base_position_size

        # Calculate recent volatility
        recent_returns = pd.Series(self.data['EURUSD'].Close).pct_change()
        volatility = recent_returns.tail(20).std()

        # Adjust position size inversely to volatility
        # Lower volatility = larger position, higher volatility = smaller position
        volatility_adjusted_size = self.base_position_size / (1 + volatility * 10)

        return min(volatility_adjusted_size, self.max_position)

    def next(self):
        if self.should_trade():
            position_size = self.calculate_position_size()
            self.positions['EURUSD'].buy(position_size)
```

## Execution Debugging

### Order Tracing

```python
class TraceableStrategy(Strategy):
    def init(self):
        data = CSVDataSource('data/EURUSD.csv')
        self.add_data(data, 'EURUSD')
        self.execution_log = []

    def next(self):
        # Log before execution
        current_price = self.data['EURUSD'].CClose
        current_position = self.positions['EURUSD'].position

        self.execution_log.append({
            'timestamp': self.data['EURUSD'].Index[self.data['EURUSD'].current_index],
            'price': current_price,
            'position_before': current_position,
            'cash_before': self.positions['EURUSD'].cash
        })

        # Execute trading logic
        if self.buy_signal():
            self.positions['EURUSD'].buy(0.5)

        # Log after execution
        self.execution_log[-1].update({
            'position_after': self.positions['EURUSD'].position,
            'cash_after': self.positions['EURUSD'].cash,
            'orders': len(self.positions['EURUSD'].orders)
        })

    def print_execution_log(self):
        """Print detailed execution trace"""
        for log_entry in self.execution_log[-10:]:  # Last 10 entries
            print(f"{log_entry['timestamp']}: "
                  f"Price={log_entry['price']:.5f}, "
                  f"Pos={log_entry['position_before']:>.2f}->{log_entry['position_after']:.2f}")
```

### Performance Attribution

```python
def analyze_execution_impact(self, report):
    """Analyze how execution affects strategy performance"""

    # Compare entry prices vs. ideal prices
    entry_analysis = []

    for order in report.orders:
        if order.side == OrderSide.BUY and order.status == OrderStatus.COMPLETE:
            # Compare execution price vs. price at order placement
            execution_price = order.price or self.get_price_at_time(order.timestamp)

            entry_analysis.append({
                'timestamp': order.timestamp,
                'execution_price': execution_price,
                'slippage': execution_price - self.get_price_at_time(order.timestamp)
            })

    # Calculate average slippage
    avg_slippage = sum(entry['slippage'] for entry in entry_analysis) / len(entry_analysis)

    print(f"Average Slippage: {avg_slippage:.6f}")
    print(f"Slippage Impact on Returns: {avg_slippage * len(entry_analysis) / report.final_cash:.2%}")

    return entry_analysis
```

## Best Practices

### 1. Order Management

```python
# Proper order lifecycle management
def manage_orders(self):
    """Best practices for order management"""

    # 1. Check for execution before placing new orders
    if len(self.positions['EURUSD'].orders) > 5:
        return  # Too many pending orders

    # 2. Validate order parameters
    current_price = self.data['EURUSD'].CClose
    if self.buy_limit_price <= current_price * 0.9:  # Price too far
        return

    # 3. Monitor order status
    for order in self.positions['EURUSD'].orders:
        if order.status == OrderStatus.PENDING:
            # Check if order should be cancelled/modified
            pass
```

### 2. Risk Controls

```python
def implement_risk_controls(self):
    """Implement proper risk management"""

    position = self.positions['EURUSD']

    # Maximum position size check
    max_position_value = position.cash * 2.0  # Max 2x leverage
    current_position_value = abs(position.position) * self.data['EURUSD'].CClose

    if current_position_value > max_position_value:
        # Reduce position
        self.positions['EURUSD'].close()

    # Maximum drawdown check
    if self.calculate_drawdown() > 0.15:  # 15% drawdown limit
        # Close all positions
        for symbol in self.positions:
            self.positions[symbol].close()
```

### 3. Execution Quality

```python
def ensure_execution_quality(self):
    """Ensure high-quality execution"""

    # 1. Use appropriate order sizes
    min_order_size = 100  # Minimum meaningful order
    if self.calculate_order_size() < min_order_size:
        return

    # 2. Consider market conditions
    if self.is_market_volatile():
        # Use smaller orders or wait for calmer conditions
        self.positions['EURUSD'].buy(0.2)  # Smaller size
    else:
        self.positions['EURUSD'].buy(0.5)  # Normal size

    # 3. Monitor execution performance
    self.track_execution_metrics()
```

## Next Steps

Now that you understand execution simulation in QuantEx, explore these related topics:

- **[Strategy Guide](./strategy.md)**: Learn how to integrate execution into your strategies
- **[Backtesting Guide](./backtesting.md)**: Understand how execution affects backtest results
- **[Broker API Reference](../../reference/quantex.execution.md)**: Complete broker and execution API

For complete API reference, see the [Execution API documentation](../../reference/quantex.execution.md).