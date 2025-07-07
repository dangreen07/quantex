# Execution Simulator

The default simulator, `ImmediateFillSimulator`, **fills market orders at the current bar's close price** and limit orders when the bar's *high≥limit≥low*.

```python
from quantex.execution import ImmediateFillSimulator

sim = ImmediateFillSimulator(portfolio)
fill = sim.execute(order, price=100.5, timestamp=ts)
```

### Commission & Slippage
- Extend `ImmediateFillSimulator` and override `execute()` to model fees, spreads or partial fills.

```python
class FixedCommission(ImmediateFillSimulator):
    def execute(self, order, price, timestamp):
        fill = super().execute(order, price, timestamp)
        fill.commission = 1.0  # flat USD per trade
        return fill
``` 