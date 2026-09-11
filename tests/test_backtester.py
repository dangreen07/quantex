from quantex.backtester import Backtester
from quantex.datasource import YahooDataSource
from tests.test_strategy import SMACrossover


def test_backtester():
    bt = Backtester(SMACrossover)
    source = YahooDataSource("NVDA", start="2022-01-01", end="2023-12-31")
    bt.add_data(source, "NVDA")
    bt.run()
    print(f"Total Return: {bt.result.total_return():.2%}")
    print(f"Total Trades: {bt.result.total_trades:,}")
