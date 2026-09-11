from quantex.datasource import YahooDataSource
from quantex.strategy import Strategy
import talib


class SMACrossover(Strategy):
    fast_period = 10
    slow_period = 30

    def init(self):
        self.fast_ma = self.Indicator(
            talib.SMA(self.data.Close, timeperiod=self.fast_period)
        )
        self.slow_ma = self.Indicator(
            talib.SMA(self.data.Close, timeperiod=self.slow_period)
        )

    def next(self):
        if self.fast_ma[-1] > self.slow_ma[-1] and self.fast_ma[-2] < self.slow_ma[-2]:
            print("BUY")


def test_strategy():
    strat = SMACrossover()
    source = YahooDataSource("NVDA", start="2022-01-01", end="2023-12-31")
    strat.add_data(source)
    strat.init()
    assert len(strat.fast_ma) == 1
    strat.fast_ma.__current__ = 15
    assert len(strat.fast_ma) == 15
