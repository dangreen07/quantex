import pandas as pd
from datetime import datetime
import pytest

from quantex.sources import BacktestingDataSource
from quantex.models import Bar
from quantex.strategy import Strategy
from quantex.backtest import BacktestRunner, _max_drawdown


class DummyDataSource(BacktestingDataSource):
    """Very small in-memory OHLCV data source for testing."""

    def __init__(self, prices, symbol: str = "TEST"):
        ts_index = pd.to_datetime(
            pd.date_range("2024-01-01", periods=len(prices), freq="min")
        )
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

    def peek_timestamp(self) -> datetime | None:
        """Peeks at the timestamp of the next available bar."""
        if self.index < len(self):
            return self._df.index[self.index]
        return None

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

    def get_raw_data(self) -> pd.DataFrame:
        return self._df


class BuyHoldStrategy(Strategy):
    """Buys on the first bar, sells on the last bar."""

    def run(self):
        source = self.data_sources.get("source")
        if not isinstance(source, DummyDataSource):
            return  # Or raise an error, depending on desired strictness

        if self.index == 0:
            self.buy(str(source.symbol), 10)
        elif self.index == len(source) - 1:
            self.close_position(str(source.symbol))


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

    # Sharpe ratio should be computed and finite
    assert "sharpe_ratio" in result.metrics
    assert result.metrics["sharpe_ratio"] is not None

    # Maximum drawdown should be computed
    assert "max_drawdown" in result.metrics
    assert isinstance(result.metrics["max_drawdown"], float)


# Test cases for the _max_drawdown function
def test_max_drawdown_empty_series():
    """Test max drawdown with empty series."""
    empty_series = pd.Series([], dtype=float)
    result = _max_drawdown(empty_series)
    assert result == 0.0


def test_max_drawdown_single_value():
    """Test max drawdown with single value."""
    single_series = pd.Series([100.0])
    result = _max_drawdown(single_series)
    assert result == 0.0


def test_max_drawdown_increasing_series():
    """Test max drawdown with strictly increasing series (no drawdown)."""
    increasing_series = pd.Series([100, 110, 120, 130])
    result = _max_drawdown(increasing_series)
    assert result == 0.0


def test_max_drawdown_decreasing_series():
    """Test max drawdown with strictly decreasing series."""
    decreasing_series = pd.Series([100, 90, 80, 70])
    result = _max_drawdown(decreasing_series)
    # Maximum drawdown should be -30% (from 100 to 70)
    assert result == pytest.approx(-0.3, rel=1e-6)


def test_max_drawdown_with_recovery():
    """Test max drawdown with a drawdown followed by recovery."""
    # Peak at 100, drops to 80 (20% drawdown), then recovers to 95
    series_with_recovery = pd.Series([100, 95, 80, 85, 95])
    result = _max_drawdown(series_with_recovery)
    # Maximum drawdown should be -20% (from 100 to 80)
    assert result == pytest.approx(-0.2, rel=1e-6)


def test_max_drawdown_multiple_peaks():
    """Test max drawdown with multiple peaks and valleys."""
    # First peak at 100, drops to 90, recovers to 110, drops to 88
    multiple_peaks_series = pd.Series([100, 90, 110, 88, 95])
    result = _max_drawdown(multiple_peaks_series)
    # Maximum drawdown should be -20% (from 110 to 88)
    assert result == pytest.approx(-0.2, rel=1e-6)


def test_max_drawdown_with_timestamps():
    """Test max drawdown with datetime index."""
    timestamps = pd.date_range("2024-01-01", periods=5, freq="D")
    nav_series = pd.Series([1000, 950, 800, 850, 900], index=timestamps)
    result = _max_drawdown(nav_series)
    # Maximum drawdown should be -20% (from 1000 to 800)
    assert result == pytest.approx(-0.2, rel=1e-6)


def test_max_drawdown_precision():
    """Test max drawdown with high precision values."""
    precise_series = pd.Series([1000.0, 999.5, 998.0, 997.5, 999.0])
    result = _max_drawdown(precise_series)
    # Maximum drawdown should be -0.25% (from 1000 to 997.5)
    assert result == pytest.approx(-0.0025, rel=1e-6)


def test_backtest_runner_with_drawdown_scenario():
    """Test BacktestRunner with a strategy that creates a drawdown scenario."""

    class DrawdownStrategy(Strategy):
        """Strategy that creates a specific drawdown pattern."""

        def run(self):
            source = self.data_sources.get("source")
            if not isinstance(source, DummyDataSource):
                return

            # Buy on first bar, sell on second bar (creating a loss), buy smaller amount on third
            if self.index == 0:
                self.buy(str(source.symbol), 10)  # Buy at 100
            elif self.index == 1:
                self.close_position(str(source.symbol))  # Sell at 80 (loss)
            elif self.index == 2:
                # Buy fewer shares since we have less cash after the loss
                self.buy(str(source.symbol), 8)  # Buy at 90

    # Create a price series that goes 100 -> 80 -> 90 -> 95
    prices = [100, 80, 90, 95]
    ds = DummyDataSource(prices, "TEST")
    strat = DrawdownStrategy({"source": ds}, initial_cash=1000)

    runner = BacktestRunner(strat, {"source": ds})
    result = runner.run()

    # Should have a meaningful drawdown
    assert "max_drawdown" in result.metrics
    assert result.metrics["max_drawdown"] < 0  # Should be negative

    # Verify the NAV series makes sense
    assert len(result.nav) == len(prices)
    assert result.nav.iloc[0] == 1000  # Starting cash
