# Execution Simulator

# Default Simulator

`NextBarSimulator` is now the **default** execution model used by
`BacktestRunner`. It **queues** orders generated during bar *t* and fills them
at the **next bar's open** price (*t&nbsp;+&nbsp;1*). This closely mirrors how
most retail broker back-tests behave and avoids look-ahead bias.

```python
from quantex.execution import NextBarSimulator

# Instance already created for you by BacktestRunner, but you can customise
# parameters (e.g. commission) and pass it explicitly.
sim = NextBarSimulator(portfolio, commission=1.00)
```

If you require immediate, zero-latency fills you can still import
`ImmediateFillSimulator`:

```python
from quantex.execution import ImmediateFillSimulator

sim = ImmediateFillSimulator(portfolio)
fill = sim.execute(order, price=100.5, timestamp=ts)
```

## Commission

Both `ImmediateFillSimulator` and `NextBarSimulator` support configurable commission through the `commission` parameter:

```python
from quantex.execution import NextBarSimulator

# Fixed commission per trade
sim = NextBarSimulator(portfolio, commission=1.00)  # $1.00 per trade

# Use with BacktestRunner
bt = BacktestRunner(strategy, data_sources, simulator=sim)
```

### Commission Models

**Fixed Commission** - Simple flat fee per trade:
```python
simulator = NextBarSimulator(portfolio, commission=1.00)
```

**Percentage-based Commission** - Commission as percentage of trade value:
```python
class PercentageCommissionSimulator(NextBarSimulator):
    def __init__(self, portfolio, commission_rate=0.001):  # 0.1%
        super().__init__(portfolio, commission=0.0)
        self.commission_rate = commission_rate
    
    # Override flush_pending to calculate dynamic commission
```

**Tiered Commission** - Different rates based on trade size:
```python
class TieredCommissionSimulator(NextBarSimulator):
    def __init__(self, portfolio):
        super().__init__(portfolio, commission=0.0)
        self.tiers = [(0, 5.00), (1000, 3.00), (5000, 1.00)]
    
    # Override flush_pending to apply tiered rates
```

Commission is automatically:
- **Deducted from portfolio cash** when trades execute
- **Included in Fill objects** for tracking and analysis  
- **Accounted for in cash sufficiency checks** before trade execution

### Custom Commission Logic
- Extend `ImmediateFillSimulator` and override `execute()` to model fees, spreads or partial fills.

```python
class CustomCommission(ImmediateFillSimulator):
    def execute(self, order, price, timestamp):
        # Custom logic here
        fill = super().execute(order, price, timestamp)
        return fill
``` 

### Next-bar fills

For workflows where you want trades to be executed on the **next bar's price**
(similar to *Backtrader* or *backtesting.py* defaults), import the
`NextBarSimulator`:

```python
from quantex.execution import NextBarSimulator

sim = NextBarSimulator(portfolio)
# The EventBus will automatically call ``flush_pending`` so orders raised on
# bar *t* are filled on bar *t+1* **before** your strategy logic runs.
```

By default, the simulator fills at the next bar's **open** price. If you need a
different rule (e.g. fill at the close), pass `fill_at="close"` when creating
the `NextBarSimulator` instance:

```python
sim = NextBarSimulator(portfolio, fill_at="close")
``` 
