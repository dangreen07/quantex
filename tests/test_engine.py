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

    # --- Run ---
    event_bus.run()

    # --- Assertions ---
    # 1. The bus should have processed 4 unique timestamps in the correct order
    event_timestamps = [e[0] for e in strategy.events]
    expected_timestamps = [ts[0], ts[1], ts[2], ts[3]]
    assert event_timestamps == expected_timestamps
    assert event_bus.timestamps == expected_timestamps

    # 2. Check forward-filling of prices at each step
    event_prices = [e[1] for e in strategy.events]
    # T1 (ts[0]): Both A and B have data
    assert event_prices[0] == {"A": 100, "B": 200}
    # T2 (ts[1]): Only B has new data, A's price is from T1
    assert event_prices[1] == {"A": 100, "B": 201}
    # T3 (ts[2]): Only A has new data, B's price is from T2
    assert event_prices[2] == {"A": 102, "B": 201}
    # T4 (ts[3]): Only B has new data, A's price is from T3
    assert event_prices[3] == {"A": 102, "B": 203}
