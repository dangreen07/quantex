from quantex.backtester import Backtester
from quantex.datasource import YahooDataSource
from quantex.strategy import Strategy
import talib


class SMACrossover(Strategy):
    fast_period = 5
    slow_period = 20
    trade_size = 0.25

    def init(self):
        self.fast_ma = self.Indicator(
            talib.SMA(self.data.Close, timeperiod=self.fast_period)
        )
        self.slow_ma = self.Indicator(
            talib.SMA(self.data.Close, timeperiod=self.slow_period)
        )

    def next(self):
        if (
            self.fast_ma[-1] > self.slow_ma[-1]
            and self.fast_ma[-2] < self.slow_ma[-2]
            and (self.broker.is_closed() or self.broker.is_short())
        ):
            self.broker.close()
            amount = self.broker.equity() * self.trade_size / self.data.Close[-1]
            self.broker.buy(
                amount=amount,
                stop_loss=0.975 * self.data.Close[-1],
                take_profit=1.1 * self.data.Close[-1],
            )
        elif (
            self.fast_ma[-1] < self.slow_ma[-1]
            and self.fast_ma[-2] > self.slow_ma[-2]
            and (self.broker.is_closed() or self.broker.is_long())
        ):
            self.broker.close()
            amount = self.broker.equity() * self.trade_size / self.data.Close[-1]
            self.broker.sell(
                amount=amount,
                stop_loss=1.025 * self.data.Close[-1],
                take_profit=0.9 * self.data.Close[-1],
            )


def test_backtester():
    bt = Backtester(SMACrossover)
    source = YahooDataSource("NVDA", start="2020-01-01", end="2025-12-31")
    bt.add_data(source, "NVDA")
    result = bt.run()
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Total Trades: {result.total_trades:,}")
    print(f"Sharpe Ratio: {result.sharpe_ratio():.2f}")
    print(f"Annualized Return: {result.annualized_return:.2%}")
    drawdown_dollars, drawdown_percent = result.max_drawdown
    print(f"Max Drawdown: $ {drawdown_dollars:,.2f} ({drawdown_percent:.2%})")
    # plt.plot(bt.result.run_strategy.data.Timestamp, bt.result.equity)
    # plt.gcf().autofmt_xdate()
    # plt.tight_layout()
    # plt.show()
