import pytest
from quantex.strategy import Strategy
from quantex.backtester import SimpleBacktester
from quantex.datasource import DataSource
import numpy as np
from pytest import approx
import pandas as pd

class BollingerBandsStrategy(Strategy):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def init(self):
        log_A = np.log(self.data['A'].Close)
        log_B = np.log(self.data['B'].Close)
        resid_log = log_A - (self.alpha + self.beta * log_B)
        self.resid_log = self.Indicator(resid_log)
    
    def next(self):
        zscore = (self.resid_log - self.resid_log.mean()) / self.resid_log.std(ddof=0)
        if zscore[-1] >= 1.5 and (self.positions['A'].is_closed() or self.positions['A'].is_short()):
            self.positions['B'].sell(0.5)
            self.positions['A'].buy(0.5)
        elif zscore[-1] <= -1.5 and (self.positions['A'].is_closed() or self.positions['A'].is_long()):
            self.positions['B'].buy(0.5)
            self.positions['A'].sell(0.5)
        elif abs(zscore[-1]) < 0.5 and not self.positions['A'].is_closed():
            self.positions['B'].close()
            self.positions['A'].close()

class TestPairsTrading:
    def test_two_data_sources(self):
        B0 = 100
        B1 = 2
        np.random.seed(3)
        normal_dist = np.random.normal(loc=0, scale=1, size=200)
        seriesA = B0 + B1 * np.cumsum(normal_dist)
        x = np.linspace(0, 4 * 2 * np.pi, 200)
        offset = np.sin(x) + np.random.normal(loc=0, scale=0.1, size=200)
        seriesB = seriesA + offset * B1
        alpha, beta = np.log(B1), 0
        strategy = BollingerBandsStrategy(alpha, beta)
        index = np.arange('2024-01-01', '2024-12-01', dtype='datetime64[h]')[:len(seriesB)]
        dataB = np.array([seriesB] * 5).transpose()
        dataA = np.array([seriesA] * 5).transpose()
        dfA = pd.DataFrame(dataA, index=index, columns=["Open", "High", "Low", "Close", "Volume"])
        dfB = pd.DataFrame(dataB, index=index, columns=["Open", "High", "Low", "Close", "Volume"])
        sourceA = DataSource(dfA)
        sourceB = DataSource(dfB)
        strategy.add_data(sourceA, 'A')
        strategy.add_data(sourceB, 'B')
        bt = SimpleBacktester(strategy, commission=0)
        report = bt.run()
        # Manually compute total return from the equity curve and ensure it
        # matches the report's total_return.
        total_return = (report.PnlRecord.iloc[-1] - 10_000) / 10_000
        assert total_return == approx(report.total_return)