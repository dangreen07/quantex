import pytest
from quantex.engine import EventBus
from quantex.strategy import Strategy
from quantex.models import Order
from datetime import datetime


class DummyStrategy(Strategy):
    """A concrete strategy for testing that does nothing on its own."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus: EventBus | None = None

    def run(self):
        pass  # pragma: no cover


class TestStrategyHelpers:
    """Tests for the buy/sell/close helper methods on the Strategy class."""

    @pytest.fixture
    def strategy(self):
        """Returns a basic strategy instance with no data sources."""
        return DummyStrategy({}, symbols=["TEST"])

    def test_buy_helper(self, strategy: DummyStrategy):
        """Test that the buy() helper creates and queues a valid buy order."""
        strategy.timestamp = datetime(2024, 1, 1)
        strategy.buy("TEST", 100)

        pending = strategy._pop_pending_orders()
        assert len(pending) == 1
        order = pending[0]
        assert isinstance(order, Order)
        assert order.symbol == "TEST"
        assert order.side == "buy"
        assert order.quantity == 100
        assert order.order_type == "market"

    def test_sell_helper(self, strategy: DummyStrategy):
        """Test that the sell() helper creates and queues a valid sell order."""
        strategy.timestamp = datetime(2024, 1, 1)
        strategy.sell("TEST", 50, limit_price=150.0)

        pending = strategy._pop_pending_orders()
        assert len(pending) == 1
        order = pending[0]
        assert order.side == "sell"
        assert order.quantity == 50
        assert order.order_type == "limit"
        assert order.limit_price == 150.0

    def test_close_position_long(self, strategy: DummyStrategy):
        """Test closing a long position."""
        strategy.timestamp = datetime(2024, 1, 1)
        # Manually create a long position
        strategy.positions["TEST"].position = 100

        strategy.close_position("TEST")
        pending = strategy._pop_pending_orders()
        assert len(pending) == 1
        order = pending[0]
        assert order.side == "sell"
        assert order.quantity == 100

    def test_close_position_short(self, strategy: DummyStrategy):
        """Test closing a short position."""
        strategy.timestamp = datetime(2024, 1, 1)
        # Manually create a short position
        strategy.positions["TEST"].position = -75

        strategy.close_position("TEST")
        pending = strategy._pop_pending_orders()
        assert len(pending) == 1
        order = pending[0]
        assert order.side == "buy"
        assert order.quantity == 75

    def test_close_position_flat(self, strategy: DummyStrategy):
        """Test that closing a flat position does nothing."""
        strategy.timestamp = datetime(2024, 1, 1)
        strategy.close_position("TEST")
        pending = strategy._pop_pending_orders()
        assert len(pending) == 0

    def test_helpers_raise_if_timestamp_is_none(self, strategy: DummyStrategy):
        """Test that helpers raise a RuntimeError if timestamp isn't set."""
        with pytest.raises(RuntimeError, match="timestamp is not set"):
            strategy.buy("TEST", 1)
        with pytest.raises(RuntimeError, match="timestamp is not set"):
            strategy.sell("TEST", 1)
        with pytest.raises(RuntimeError, match="timestamp is not set"):
            strategy.close_position("TEST")

    # ------------------------------------------------------------------
    # New tests covering get_price / prices helpers and price history fetcher
    # ------------------------------------------------------------------

    def test_get_price_and_prices_helpers(self):
        strat = DummyStrategy({}, symbols=["AAA", "BBB"])

        # Prepare fake market data injected by EventBus
        price_row = [1.0, 2.0]
        symbols = ["AAA", "BBB"]
        symbol_idx = {"AAA": 0, "BBB": 1}

        # Inject internals manually (simulate EventBus._update_market_data)
        strat._update_market_data(price_row, symbols, symbol_idx)  # type: ignore[attr-defined]

        # get_price returns correct value
        assert strat.get_price("AAA") == 1.0
        assert strat.get_price("BBB") == 2.0

        # prices dict reflects both
        assert strat.prices == {"AAA": 1.0, "BBB": 2.0}

        # Unknown symbol raises KeyError
        with pytest.raises(KeyError):
            strat.get_price("MISSING")

    def test_get_price_runtime_error_when_uninitialised(self):
        strat = DummyStrategy({}, symbols=["X"])
        with pytest.raises(RuntimeError, match="Market data not yet initialised"):
            _ = strat.get_price("X")

        with pytest.raises(RuntimeError, match="Market data not yet initialised"):
            _ = strat.prices

    def test_get_price_history_df_errors_and_success(self):
        strat = DummyStrategy({}, symbols=["X"])

        # No event_bus attached
        with pytest.raises(RuntimeError, match="Strategy is not attached"):
            _ = strat._get_price_history_df()

        # Attach event_bus but with _price_df None – should raise second error
        class FakeBus(EventBus):
            def __init__(self):
                self._price_close_df: pd.DataFrame | None = None
                self._price_open_df: pd.DataFrame | None = None

        fake_bus = FakeBus()
        strat.event_bus = fake_bus
        with pytest.raises(RuntimeError, match="Price history not yet initialised"):
            _ = strat._get_price_history_df()

        # Provide actual DataFrame
        import pandas as pd

        df = pd.DataFrame({"X": [1, 2, 3]})
        fake_bus._price_close_df = df

        # Should now return the DataFrame
        returned = strat._get_price_history_df()
        assert returned.equals(df)

    def test_buy_with_cash_and_default(self, strategy: DummyStrategy):
        """Buy helper sizes position correctly when quantity omitted."""
        strategy.timestamp = datetime(2024, 1, 1)
        # Inject fake price so helper can compute quantity
        strategy._update_market_data([50.0], [50.0], ["TEST"], {"TEST": 0})

        # Case 1: No args -> invest all cash
        strategy.portfolio.cash = 1_000
        strategy.buy("TEST")
        order = strategy._pop_pending_orders()[0]
        assert order.quantity == 20  # floor(1000 / 50)

        # Case 2: cash-limited sizing
        strategy.portfolio.cash = 1_000
        strategy.buy("TEST", cash=250)
        order2 = strategy._pop_pending_orders()[0]
        assert order2.quantity == 5  # floor(250/50)
