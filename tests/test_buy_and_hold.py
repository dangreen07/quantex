import pytest
from quantex import Strategy, SimpleBacktester, DataSource
import numpy as np
import pandas as pd

class BuyAndHold(Strategy):
    def init(self):
        self.data_length = len(self.data['A'])

    def next(self):
        if self.positions['A'].is_closed():
            self.positions['A'].buy(0.8)
        elif len(self.data['A'].Close) >= self.data_length:
            self.positions['A'].close()

class TestBuyAndHold:
    def test_buy_and_hold(self):
        y = 2 * np.linspace(2, 100, 500)
        print(f"Final Value: {y[-1]}, Initial Value: {y[0]}")
        expected_return = ((y[-1] - y[1]) / y[0]) * 0.8
        index = np.arange('2024-01-01', '2024-12-01', dtype='datetime64[h]')[:len(y)]
        data = np.array([y] * 5).transpose()
        df = pd.DataFrame(data, index, columns=["Open", "High", "Low", "Close", "Volume"])
        source = DataSource(df)
        strategy = BuyAndHold()
        strategy.add_data(source, 'A')
        bt = SimpleBacktester(strategy, cash=1_000, commission=0)
        report = bt.run()
        actual_return = (report.PnlRecord.iloc[-1] - report.PnlRecord.iloc[0]) / report.PnlRecord.iloc[0]
        assert actual_return == expected_return