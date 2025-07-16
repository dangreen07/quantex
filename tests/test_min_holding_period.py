import pandas as pd
import pytest

from quantex.backtest import BacktestRunner
from quantex.strategy import Strategy
from tests.test_backtest import DummyDataSource  # Reuse helper


class BuySellNextBarStrategy(Strategy):
    """Buys on first bar, attempts to sell on second bar."""

    def run(self):
        source = self.data_sources.get("source")
        assert source is not None  # noqa: S101 – test guard
        if self.index == 0:
            self.buy(str(source.symbol), 1)
        elif self.index == 1:
            self.close_position(str(source.symbol))


def test_min_holding_period_blocks_early_exit():
    """Ensure that selling before min holding period raises ValueError."""
    prices = [100, 101, 102]
    ds = DummyDataSource(prices, "TEST")
    strat = BuySellNextBarStrategy({"source": ds}, initial_cash=1000)

    min_period = pd.Timedelta(minutes=2)  # Each bar is 1 minute in DummyDataSource

    runner = BacktestRunner(strat, {"source": ds}, min_holding_period=min_period)

    with pytest.raises(ValueError, match="minimum holding period"):
        runner.run()


class BuySellAfterPeriodStrategy(Strategy):
    """Buys on first bar, sells after the minimum holding period."""

    def run(self):
        source = self.data_sources.get("source")
        assert source is not None  # noqa: S101 – test guard
        if self.index == 0:
            self.buy(str(source.symbol), 1)
        elif self.index == 2:
            # Sell after 2-minute hold
            self.close_position(str(source.symbol))


def test_min_holding_period_allows_after_wait():
    """Verify that closing after min holding period succeeds."""
    prices = [100, 101, 102, 103]
    ds = DummyDataSource(prices, "TEST")
    strat = BuySellAfterPeriodStrategy({"source": ds}, initial_cash=1000)

    min_period = pd.Timedelta(minutes=2)

    runner = BacktestRunner(strat, {"source": ds}, min_holding_period=min_period)
    result = runner.run()

    # Strategy should have closed position successfully, NAV recorded for each bar
    assert len(result.nav) == len(prices)
    # Ensure no open position at the end
    assert strat.portfolio.positions["TEST"].is_closed
