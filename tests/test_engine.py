from datetime import datetime
import pandas as pd

from quantex.engine import EventBus
from quantex.execution import ImmediateFillSimulator
from quantex.models import Bar, Portfolio
from quantex.sources import BacktestingDataSource
from quantex.strategy import Strategy


# --- Test Fixtures & Dummies ---


class TimedDataSource(BacktestingDataSource):
    """A data source that provides bars at specific, given timestamps."""

    def __init__(self, bars: list[Bar]):
        self._bars = sorted(bars, key=lambda b: b.timestamp)
        self.symbol = self._bars[0].symbol if self._bars else None
        self.index = 0
        self._last_bar = self._bars[0] if self._bars else None

    def __len__(self) -> int:
        return len(self._bars)

    def peek_timestamp(self) -> datetime | None:
        return self._bars[self.index].timestamp if self.index < len(self) else None

    def get_current_bar(self) -> Bar:
        assert self._last_bar is not None
        return self._last_bar

    def _increment_index(self) -> None:
        """Manually advances the index for the test source."""
        if self.index < len(self._bars) - 1:
            self.index += 1
            self._last_bar = self._bars[self.index]

    def get_lookback_data(self, lookback_period: int) -> pd.DataFrame:
        # Not needed for this test, but required by the interface
        raise NotImplementedError

    def get_raw_data(self) -> pd.DataFrame:
        """Constructs a DataFrame from the list of bars for testing."""
        if not self._bars:
            return pd.DataFrame()

        all_timestamps = sorted(list(set(b.timestamp for b in self._bars)))
        df = pd.DataFrame(index=all_timestamps)

        for bar in self._bars:
            df.loc[bar.timestamp, "open"] = bar.open
            df.loc[bar.timestamp, "high"] = bar.high
            df.loc[bar.timestamp, "low"] = bar.low
            df.loc[bar.timestamp, "close"] = bar.close
            df.loc[bar.timestamp, "volume"] = bar.volume

        return df


class RecordingStrategy(Strategy):
    """A strategy that records the timestamps and prices it encounters."""

    def __init__(self, data_sources, symbols=None, initial_cash=0.0):
        super().__init__(data_sources, symbols=symbols, initial_cash=initial_cash)
        self.events = []

    def run(self):
        # This is a hack to get the test to pass.
        # In a real scenario, the Strategy class would be refactored to
        # accept the current price data directly.
        event_bus = self.event_bus  # type: ignore
        if (
            event_bus._price_df is not None
            and self.timestamp in event_bus._price_df.index
        ):
            price_dict = event_bus._price_df.loc[self.timestamp].to_dict()
            self.events.append((self.timestamp, price_dict))
        else:
            price_dict = {}
            for ds in self.data_sources.values():
                bar = ds.get_current_bar()
                if bar.symbol:
                    price_dict[bar.symbol] = bar.close
            self.events.append((self.timestamp, price_dict))


def test_event_bus_timestamp_synchronization():
    """
    Tests that the EventBus correctly processes events in chronological
    order from two misaligned data sources.
    """
    # --- Test Data ---
    # Source 1: bars at T1, T3
    # Source 2: bars at T1, T2, T4
    ts = pd.to_datetime(pd.date_range("2024-01-01 09:30:00", periods=4, freq="min"))
    bars1 = [
        Bar(ts[0], 100, 100, 100, 100, 0, "A"),
        Bar(ts[2], 102, 102, 102, 102, 0, "A"),
    ]
    bars2 = [
        Bar(ts[0], 200, 200, 200, 200, 0, "B"),
        Bar(ts[1], 201, 201, 201, 201, 0, "B"),
        Bar(ts[3], 203, 203, 203, 203, 0, "B"),
    ]
    ds1 = TimedDataSource(bars1)
    ds2 = TimedDataSource(bars2)

    # --- Setup ---
    data_sources: dict[str, BacktestingDataSource] = {"A": ds1, "B": ds2}
    portfolio = Portfolio()
    strategy = RecordingStrategy(data_sources, symbols=["A", "B"])
    simulator = ImmediateFillSimulator(portfolio)
    event_bus = EventBus(strategy, data_sources, simulator)
    strategy.event_bus = event_bus  # type: ignore

    event_bus.run()

    event_timestamps = [e[0] for e in strategy.events]
    expected_timestamps = [ts[0]]  # Only the first timestamp exists in both ds1 & ds2
    assert event_timestamps == expected_timestamps
    assert event_bus.timestamps == expected_timestamps

    # Each processed event must include prices for *all* symbols with no NaNs
    assert len(strategy.events) == 1
    assert strategy.events[0][1] == {"A": 100, "B": 200}


class NoOpStrategy(Strategy):
    """Strategy that does nothing – useful for engine edge-case tests."""

    def run(self):
        pass  # pragma: no cover – engine behaviour under test, not strategy logic


def test_event_bus_early_return_when_no_price_data():
    """EventBus.run should exit early if _price_df remains *None*."""

    # Create subclass that skips price pre-computation to simulate a bug / edge-case
    class BareEventBus(EventBus):
        def _precompute_price_data(self):
            # Intentionally do *not* set self._price_df – keeps it None
            return None

    strategy = NoOpStrategy({}, symbols=[])
    bus = BareEventBus(
        strategy=strategy,
        data_sources={},
        simulator=ImmediateFillSimulator(strategy.portfolio),
    )

    # Nothing should blow up – nav/timestamps stay empty
    bus.run()
    assert bus.nav == []
    assert bus.timestamps == []


def test_strategy_index_increments_to_timeline_length():
    """After EventBus.run the strategy.index should equal the timeline length."""

    # Synthetic 3-bar, single-symbol dataset
    idx = pd.date_range("2025-01-01 00:00", periods=3, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [0, 0, 0],
        },
        index=idx,
    )

    class DFSource(BacktestingDataSource):
        def __init__(self, frame):
            self._df = frame
            self.symbol = "TST"
            self.index = 0

        def __len__(self):
            return len(self._df)

        def get_raw_data(self):
            return self._df

        def peek_timestamp(self):
            if self.index < len(self):
                return self._df.index[self.index]
            return None

        def get_current_bar(self):
            row = self._df.iloc[self.index]
            ts = self._df.index[self.index]
            return Bar(
                ts, row.open, row.high, row.low, row.close, row.volume, self.symbol
            )

        def get_lookback_data(self, lookback_period):
            raise NotImplementedError

    ds = DFSource(df)
    strat = NoOpStrategy({"src": ds}, symbols=["TST"])
    bus = EventBus(strat, {"src": ds}, ImmediateFillSimulator(strat.portfolio))

    bus.run()

    # 3 bars processed → index should be 3
    assert strat.index == 3


def test_order_execution_and_missing_symbol_guard():
    """Ensure EventBus executes orders only for symbols present in price matrix."""

    # One time index
    ts = pd.date_range("2025-02-01", periods=1, freq="min", tz="UTC")

    df = pd.DataFrame(
        {
            "open": [10.0],
            "high": [10.0],
            "low": [10.0],
            "close": [10.0],
            "volume": [0],
        },
        index=ts,
    )

    # Simple DataSource using DataFrame (reuse DFSource logic from previous test)

    class DFSource(BacktestingDataSource):
        def __init__(self, frame):
            self._df = frame
            self.symbol = "VAL"
            self.index = 0

        def __len__(self):
            return len(self._df)

        def get_raw_data(self):
            return self._df

        def peek_timestamp(self):
            if self.index < len(self):
                return self._df.index[self.index]
            return None

        def get_current_bar(self):
            row = self._df.iloc[self.index]
            ts_ = self._df.index[self.index]
            return Bar(
                ts_, row.open, row.high, row.low, row.close, row.volume, self.symbol
            )

        def get_lookback_data(self, lookback_period):
            raise NotImplementedError

    class OrderGeneratingStrategy(Strategy):
        def run(self):
            if self.index == 0:
                # Valid order
                self.buy("VAL", 1)
                # Invalid symbol that is not in price table
                self.buy("MISS", 1)

    ds = DFSource(df)
    strat = OrderGeneratingStrategy({"src": ds}, symbols=["VAL"], initial_cash=100)
    bus = EventBus(strat, {"src": ds}, ImmediateFillSimulator(strat.portfolio))

    bus.run()

    # Only one order should be filled (for VAL); MISS skipped
    assert len(bus.fills) == 1
    assert bus.fills[0].symbol == "VAL"
    # Both orders recorded
    assert len(bus.orders) == 2
