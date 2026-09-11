from quantex.backtester import Backtester
from quantex.datasource import YahooDataSource
from tests.test_strategy import SMACrossover


def test_backtester():
    bt = Backtester(SMACrossover)
    source = YahooDataSource("NVDA", start="2022-01-01", end="2023-12-31")
    bt.add_data(source, "NVDA")
    bt.run()
