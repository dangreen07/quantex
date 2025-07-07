import pytest
from quantex.strategy import Strategy
from quantex.models import Order
from datetime import datetime


class DummyStrategy(Strategy):
    """A concrete strategy for testing that does nothing on its own."""

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
