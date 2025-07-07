import pandas as pd

from quantex.sources import BacktestingDataSource
from quantex.models import Bar, Order
from quantex.strategy import Strategy
from quantex.backtest import BacktestRunner


class DummyDataSource(BacktestingDataSource):
    """Very small in-memory OHLCV data source for testing."""

    def __init__(self, prices, symbol: str = "TEST"):
        ts_index = pd.date_range("2024-01-01", periods=len(prices), freq="min")
        self._df = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": [0] * len(prices),
            },
            index=ts_index,
        )
        self.symbol = symbol
        self.index = 0

    # ------------------------------------------------------------------
    # BacktestingDataSource interface
    # ------------------------------------------------------------------
    def __len__(self):  # noqa: D401 – simple method
        return len(self._df)

    def get_current_bar(self):  # noqa: D401
        row = self._df.iloc[self.index]
        ts = self._df.index[self.index]
        return Bar(
            timestamp=ts,
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            symbol=self.symbol,
        )

    def get_lookback_data(self, lookback_period):  # noqa: D401
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1].copy()


class BuyHoldStrategy(Strategy):
    """Buys on the first bar, sells on the last bar."""

    def run(self):
        ds = self.data_sources["source"]
        current_bar = ds.get_current_bar()
        if self.index == 0:
            self.submit_order(
                Order(
                    id="buy",
                    symbol=str(current_bar.symbol),
                    side="buy",
                    quantity=10,
                )
            )
        elif self.index == len(ds) - 1:  # type: ignore[arg-type]
            self.submit_order(
                Order(
                    id="sell",
                    symbol=str(current_bar.symbol),
                    side="sell",
                    quantity=10,
                )
            )


def test_backtest_runner_basic():
    prices = [10, 11, 12]
    ds = DummyDataSource(prices)
    strat = BuyHoldStrategy({"source": ds}, initial_cash=1000)

    runner = BacktestRunner(strat, {"source": ds})
    result = runner.run()

    # NAV recorded for each bar
    assert len(result.nav) == len(prices)
    # Total return should not be zero because price moved
    assert result.metrics["total_return"] != 0
