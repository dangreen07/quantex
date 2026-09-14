# QuantEx Documentation

QuantEx is a Python backtesting library for OHLCV data. It is designed to be easy to use and extend, and to provide a simple interface for backtesting trading strategies.

## Getting Started

To get started with QuantEx, you can install it using pip:
```bash
pip install quantex
```

QuantEx is designed to be used by writing your strategy as event-driven code. An example buy and hold strategy is shown below:

```python
from quantex.strategy import Strategy

class BuyAndHold(Strategy):
    def init(self):
        pass

    def next(self):
        if self.broker.is_closed():
            self.broker.buy(amount=50)
```

This strategy will buy 50 shares of the stock when there are no open positions. To run this strategy, we must first get some data. Currently, QuantEx supports Yahoo Finance and pandas DataFrames as data sources.  
To run the strategy with the data, we can use the Backtester class:
```python
from quantex.backtester import Backtester
from quantex.datasource import YahooDataSource

bt = Backtester(BuyAndHold)
source = YahooDataSource("NVDA", start="2020-01-01", end="2025-12-31")
bt.add_data(source, "NVDA")
result = bt.run()
```

The result object contains information about the strategy, such as the total return, annualized return, and max drawdown.

If you would like to include commissions in your strategy, you can use the Commission class:

```python
from quantex.commission import Commission, ApplicationType
from quantex.backtester import Backtester
from quantex.datasource import YahooDataSource

bt = Backtester(BuyAndHold, commission=Commission(ApplicationType.PER_ORDER, fixed=1))
source = YahooDataSource("NVDA", start="2020-01-01", end="2025-12-31")
bt.add_data(source, "NVDA")
result = bt.run()
```

This will charge a fixed commission of 1 dollar (or whatever the currency price is quoted in) per order.