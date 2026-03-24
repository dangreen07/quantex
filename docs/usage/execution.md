# Execution Guide

This guide explains how orders work in Quantex and what the broker actually simulates.

If you are new to the library, the most important thing to understand is that you do not place orders directly on the backtester. You place them on a symbol-specific [`Broker`](../../src/quantex/broker.py:113), usually from inside [`Strategy.next()`](../../src/quantex/strategy.py:71).

## Where brokers come from

When you call [`Strategy.add_data()`](../../src/quantex/strategy.py:98), Quantex does two things:

1. stores the data source in [`Strategy.data`](../../src/quantex/strategy.py:48)
2. creates a matching [`Broker`](../../src/quantex/broker.py:113) in [`Strategy.positions`](../../src/quantex/strategy.py:47)

That means this call:

```python
self.add_data(CSVDataSource("eurusd.csv"), "EURUSD")
```

gives you a broker at:

```python
self.positions["EURUSD"]
```

## The public execution methods

The broker methods intended for normal strategy code are:

- [`Broker.buy()`](../../src/quantex/broker.py:159)
- [`Broker.sell()`](../../src/quantex/broker.py:235)
- [`Broker.close()`](../../src/quantex/broker.py:307)
- [`Broker.is_long()`](../../src/quantex/broker.py:359)
- [`Broker.is_short()`](../../src/quantex/broker.py:373)
- [`Broker.is_closed()`](../../src/quantex/broker.py:387)

## Basic market orders

### Buying

```python
def next(self):
    broker = self.positions["EURUSD"]
    broker.buy(quantity=0.5)
```

In the current implementation of [`Broker.buy()`](../../src/quantex/broker.py:159), `quantity=0.5` means “use roughly 50% of the broker cash for this symbol”, unless you pass `amount`.

### Selling

```python
def next(self):
    broker = self.positions["EURUSD"]
    broker.sell(quantity=0.25)
```

[`Broker.sell()`](../../src/quantex/broker.py:235) can reduce an existing long position or create/increase a short position.

### Closing

```python
def next(self):
    broker = self.positions["EURUSD"]

    if broker.is_long():
        broker.close()
```

[`Broker.close()`](../../src/quantex/broker.py:307) submits an offsetting market order for the current position.

## Order sizing rules

This is one of the most important practical details in Quantex.

### Fractional sizing through `quantity`

In both [`Broker.buy()`](../../src/quantex/broker.py:159) and [`Broker.sell()`](../../src/quantex/broker.py:235), the `quantity` argument must be greater than 0 and less than or equal to 1.

That validation is enforced in both methods.

For buys, share count is calculated from available broker cash and current visible price.

Example:

```python
broker.buy(quantity=0.25)
```

means “allocate about 25% of broker cash to a buy order”.

### Explicit sizing through `amount`

If you pass `amount`, it overrides the automatic quantity calculation.

```python
broker.buy(amount=100)
broker.sell(amount=50)
```

Important correction: in the current code, `amount` is treated as the number of units or shares, not a cash amount. Some older documentation described `amount` as currency-based sizing, which is inaccurate for the current implementation.

## Order types

The order type enum is [`OrderType`](../../src/quantex/broker.py:24).

Current supported types:

- [`OrderType.MARKET`](../../src/quantex/broker.py:34)
- [`OrderType.LIMIT`](../../src/quantex/broker.py:35)

### Market orders

If you do not pass a `limit`, the broker creates a market order.

```python
broker.buy(quantity=0.5)
broker.sell(quantity=0.5)
```

Market orders are processed in [`Broker._iterate()`](../../src/quantex/broker.py:483) and execute using [`DataSource.COpen`](../../src/quantex/datasource.py:145).

That means the fill logic is based on the current bar's open price, not the close.

### Limit orders

If you pass `limit`, the broker creates a limit order.

```python
broker.buy(quantity=0.5, limit=99.5)
broker.sell(quantity=0.5, limit=105.0)
```

In the current implementation:

- a buy limit executes when [`DataSource.COpen`](../../src/quantex/datasource.py:145) is less than or equal to the limit price
- a sell limit executes when [`DataSource.COpen`](../../src/quantex/datasource.py:145) is greater than or equal to the limit price

## Stop loss and take profit behavior

Both [`Broker.buy()`](../../src/quantex/broker.py:159) and [`Broker.sell()`](../../src/quantex/broker.py:235) accept:

- `stop_loss`
- `take_profit`

Example:

```python
broker.buy(
    quantity=0.25,
    stop_loss=95.0,
    take_profit=110.0,
)
```

When an order with stop-loss or take-profit levels becomes active, the broker keeps monitoring it in [`Broker._iterate()`](../../src/quantex/broker.py:589).

Important implementation detail:

- stop-loss and take-profit checks are also based on [`DataSource.COpen`](../../src/quantex/datasource.py:145)
- when the condition is met, the broker creates a new market order in the opposite direction

## Order lifecycle

Order state is represented by [`OrderStatus`](../../src/quantex/broker.py:37).

The three states are:

- [`OrderStatus.PENDING`](../../src/quantex/broker.py:50)
- [`OrderStatus.ACTIVE`](../../src/quantex/broker.py:48)
- [`OrderStatus.COMPLETE`](../../src/quantex/broker.py:49)

### What they mean in practice

- `PENDING`: order exists but has not been fully processed yet
- `ACTIVE`: order has executed and still has stop-loss or take-profit monitoring attached
- `COMPLETE`: order no longer has further actions to manage

You can inspect pending orders through [`Broker.orders`](../../src/quantex/broker.py:151) and completed ones through [`Broker.complete_orders`](../../src/quantex/broker.py:152).

Example:

```python
def next(self):
    broker = self.positions["EURUSD"]

    print("pending", len(broker.orders))
    print("completed", len(broker.complete_orders))

    for order in broker.orders:
        print(order.side, order.type, order.status)
```

## Position state

Useful broker fields include:

- [`Broker.position`](../../src/quantex/broker.py:143)
- [`Broker.position_avg_price`](../../src/quantex/broker.py:144)
- [`Broker.cash`](../../src/quantex/broker.py:145)

Example:

```python
def next(self):
    broker = self.positions["EURUSD"]

    print("position", broker.position)
    print("avg entry", broker.position_avg_price)
    print("cash", broker.cash)
```

If you want a simple unrealized value estimate, use the current close:

```python
def next(self):
    broker = self.positions["EURUSD"]
    current_price = self.data["EURUSD"].CClose
    unrealized_value = broker.position * current_price
    print(unrealized_value)
```

Important correction: the current [`Broker`](../../src/quantex/broker.py:113) class does **not** expose a public `unrealized_pnl` property. Older docs referenced one, but it is not in the current implementation.

## Commission handling

Commission logic is controlled by [`CommissionType`](../../src/quantex/enums.py:4) and implemented in [`Broker._calc_commission()`](../../src/quantex/broker.py:440).

### Percentage commission

With [`CommissionType.PERCENTAGE`](../../src/quantex/enums.py:17), commission is:

```text
quantity * price * commission
```

### Cash commission

With [`CommissionType.CASH`](../../src/quantex/enums.py:18), commission is:

```text
quantity * commission / lot_size
```

## Margin-call behavior

The broker stores a margin threshold in [`Broker.margin_call`](../../src/quantex/broker.py:149). During [`Broker._iterate()`](../../src/quantex/broker.py:633), if equity falls below the calculated margin threshold while the position is short, the broker calls [`Broker.close()`](../../src/quantex/broker.py:307).

This is a limited form of margin handling, not a full brokerage margin model.

## Leverage for amplified position sizing

The broker supports leverage to amplify position sizing. With leverage enabled via the [`SimpleBacktester`](../../src/quantex/backtester.py:356):

- `leverage` multiplier is set on the backtester and passed to each broker
- When buying, shares are calculated as `base_shares * leverage`
- Margin (cash used) is calculated as `position_value / leverage`
- This allows controlling larger positions with the same cash

Example with leverage:

```python
from quantex import SimpleBacktester

# 2x leverage means you control 2x the position while only using 1x cash as margin
backtester = SimpleBacktester(
    strategy,
    cash=10_000,
    leverage=2.0,
)

# In a strategy, a buy with quantity=0.5:
# - Without leverage: controls 50% of cash worth of shares
# - With 2x leverage: controls 100% of cash worth of shares
# - Margin used: 100% / 2 = 50% of cash
```

Note: Leverage must be at least 0.1 (validated in the backtester constructor).

## A realistic example

```python
from quantex import Strategy, CSVDataSource


class BreakoutStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.entered = False

    def init(self):
        self.add_data(CSVDataSource("data.csv"), "TEST")

    def next(self):
        broker = self.positions["TEST"]
        current_open = self.data["TEST"].COpen

        if not self.entered:
            broker.buy(
                quantity=0.25,
                stop_loss=current_open * 0.98,
                take_profit=current_open * 1.04,
            )
            self.entered = True

        if broker.is_long() and self.data["TEST"].CClose < current_open:
            broker.close()
```

## Things the current execution engine does not do

The current codebase does **not** include:

- slippage modeling
- partial fills
- public order cancellation
- separate bid/ask prices
- advanced order types such as stop-limit orders

Some earlier documentation discussed these ideas conceptually, but they are not current built-in features.

## Debugging order behavior

One good way to debug is to print current market state, broker state, and queued orders from inside [`Strategy.next()`](../../src/quantex/strategy.py:71).

```python
def next(self):
    broker = self.positions["EURUSD"]
    source = self.data["EURUSD"]

    print(source.Index[source.current_index])
    print("open", source.COpen, "close", source.CClose)
    print("position", broker.position, "cash", broker.cash)

    for order in broker.orders:
        print(order.side, order.type, order.status, order.price)
```

See [`tests/test_broker.py`](../../tests/test_broker.py) for the behavior currently asserted by the test suite.

## Summary

Use the broker in [`Strategy.positions`](../../src/quantex/strategy.py:47) to place orders.

Remember these three rules:

1. market orders execute from current open logic, not close logic
2. `quantity` is fractional sizing, while `amount` is unit count
3. the broker API is intentionally small and does not model every market microstructure detail

For portfolio results and reports, see [Backtesting guide](./backtesting.md).

