import pytest
import pandas as pd
import numpy as np
from quantex.backtester.constants import CommissionType
from quantex.datasource import DataSource
from quantex.broker import Broker, Order, OrderSide, OrderType, OrderStatus


class TestBroker:
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'Low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Volume': [1000] * 10
        }, index=dates)
        return data

    @pytest.fixture
    def datasource(self, sample_data):
        """Create a DataSource instance."""
        return DataSource(sample_data)

    @pytest.fixture
    def broker(self, datasource):
        """Create a Broker instance."""
        return Broker(datasource)

    def test_init(self, broker):
        """Test Broker initialization."""
        assert broker.position == 0.0
        assert broker.cash == 10000.0
        assert broker.commision == 0.002
        assert broker.commision_type == CommissionType.PERCENTAGE
        assert broker.lot_size == 1
        assert len(broker.orders) == 0
        assert len(broker.complete_orders) == 0

    def test_buy_market_order(self, broker):
        """Test buying with market order."""
        broker._i = 0  # Set current index
        broker.buy(quantity=0.5)  # Buy 50% of cash

        assert len(broker.orders) == 1
        order = broker.orders[0]
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING

    def test_sell_market_order(self, broker):
        """Test selling with market order."""
        broker._i = 0
        broker.sell(quantity=0.5)

        assert len(broker.orders) == 1
        order = broker.orders[0]
        assert order.side == OrderSide.SELL
        assert order.type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING

    def test_buy_limit_order(self, broker):
        """Test buying with limit order."""
        broker._i = 0
        broker.buy(quantity=0.5, limit=99.0)

        assert len(broker.orders) == 1
        order = broker.orders[0]
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.LIMIT
        assert order.price == 99.0
        assert order.status == OrderStatus.PENDING

    def test_invalid_quantity_buy(self, broker):
        """Test invalid quantity for buy order."""
        with pytest.raises(ValueError, match="Quantity must be between 0 and 1"):
            broker.buy(quantity=2.0)

    def test_invalid_quantity_sell(self, broker):
        """Test invalid quantity for sell order."""
        with pytest.raises(ValueError, match="Quantity must be between 0 and 1"):
            broker.sell(quantity=2.0)

    def test_negative_limit_buy(self, broker):
        """Test negative limit price for buy."""
        with pytest.raises(ValueError, match="Cannot have a negative limit price"):
            broker.buy(quantity=0.5, limit=-10.0)

    def test_negative_amount_buy(self, broker):
        """Test negative amount for buy."""
        with pytest.raises(ValueError, match="Cannot have a negative amount"):
            broker.buy(quantity=0.5, amount=-100.0)

    def test_close_position_long(self, broker):
        """Test closing a long position."""
        broker.position = 10.0
        broker._i = 0
        broker.close()

        assert len(broker.orders) == 1
        order = broker.orders[0]
        assert order.side == OrderSide.SELL
        assert order.quantity == 10.0
        assert order.type == OrderType.MARKET

    def test_close_position_short(self, broker):
        """Test closing a short position."""
        broker.position = -10.0
        broker._i = 0
        broker.close()

        assert len(broker.orders) == 1
        order = broker.orders[0]
        assert order.side == OrderSide.BUY
        assert order.quantity == 10.0
        assert order.type == OrderType.MARKET

    def test_calc_commission_percentage(self, broker):
        """Test commission calculation for percentage type."""
        commission = broker._calc_commission(100.0, 10.0)
        expected = 100.0 * 10.0 * 0.002  # quantity * price * commission_rate
        assert commission == expected

    def test_calc_commission_cash(self, broker):
        """Test commission calculation for cash type."""
        broker.commision_type = CommissionType.CASH
        commission = broker._calc_commission(100.0, 10.0)
        expected = 100.0 * 0.002 / 1  # quantity * commission / lot_size
        assert commission == expected

    def test_debit_insufficient_funds(self, broker):
        """Test debiting more than available equity."""
        broker.cash = 50.0
        with pytest.raises(ValueError, match="Insufficient equity for this operation"):
            broker._debit(100.0)

    def test_credit(self, broker):
        """Test crediting cash."""
        initial_cash = broker.cash
        broker._credit(100.0)
        assert broker.cash == initial_cash + 100.0

    def test_margin_call_records_event_and_sets_flag(self, broker):
        """Test that a margin call is visible to the user through broker state."""
        broker.position = -10.0
        broker.cash = 0.0
        broker._i = 0
        broker.source.current_index = 0

        broker._iterate(0)

        assert broker.margin_call_triggered is True
        assert len(broker.margin_call_events) == 1
        event = broker.margin_call_events[0]
        assert event["position"] == -10.0
        assert event["margin_call_threshold"] > event["equity"]

    def test_backtest_report_includes_margin_calls(self, datasource):
        """Test that the backtest report surfaces margin call events in its output."""
        from quantex.backtester.reports import BacktestReport

        report = BacktestReport(
            starting_cash=np.float64(10000),
            final_cash=np.float64(9000),
            PnlRecord=pd.Series([10000.0, 9000.0]),
            orders=[],
            tradeRecord=[],
            margin_call_events=[{"timestamp": datasource.Index[0], "equity": 9000.0, "margin_call_threshold": 9500.0, "position": -10.0}],
        )

        assert "Margin Calls: 1" in str(report)

    def test_leverage_default_value(self, broker):
        """Test that leverage defaults to 1.0 (no leverage)."""
        assert broker.leverage == 1.0

    def test_leverage_position_amplification(self, datasource):
        """Test that leverage amplifies position size when using quantity parameter."""
        broker = Broker(datasource)
        broker.leverage = 2.0  # 2x leverage
        broker._i = 0
        broker.source.current_index = 0
        
        # With 2x leverage, buying quantity=0.5 should result in 2x the base shares
        initial_cash = broker.cash  # 10000
        current_price = datasource.Close[0]  # 102
        
        broker.buy(quantity=0.5)  # 50% of cash
        
        order = broker.orders[0]
        
        # Base shares: (10000 * 0.5) / 102 = 49.02
        # After round with share_decimals=1: 49.0
        # With 2x leverage: 49.0 * 2 = 98.0
        expected_leveraged_shares = 98.0
        
        assert abs(order.quantity - expected_leveraged_shares) < 1.0

    def test_leverage_margin_calculation(self, datasource):
        """Test that leverage affects margin requirement but not cash deduction."""
        broker = Broker(datasource)
        broker.leverage = 2.0  # 2x leverage
        broker._i = 0
        broker.source.current_index = 0
        
        # Buy using quantity (not amount) to test leverage effect
        initial_cash = broker.cash
        
        # Buy quantity=0.5: base shares = 49.0, with 2x leverage = 98.0 shares
        broker.buy(quantity=0.5)
        
        # Process the order
        broker._iterate(0)
        
        # Position: 98 shares at ~$102 = ~$10,000
        # With 2x leverage, margin = $10,000 / 2 = $5,000
        # Commission on full position: 98 * 102 * 0.002 = ~$20
        # NOTE: In the new model, only commission is deducted from cash.
        # Margin is tracked separately via used_margin, not deducted from cash.
        expected_position = 98.0
        expected_commission = expected_position * datasource.COpen * broker.commision
        expected_cash = initial_cash - expected_commission  # Only commission deducted
        
        # Verify cash was reduced only by commission
        assert abs(broker.cash - expected_cash) < 1.0
        
        # Verify position was opened correctly
        assert abs(broker.position - expected_position) < 1.0
        
        # Verify used_margin is tracked correctly
        expected_used_margin = abs(broker.position) * datasource.CClose / broker.leverage
        assert abs(broker._get_used_margin() - expected_used_margin) < 1.0

    def test_leverage_amount_bypasses_leverage(self, datasource):
        """Test that using amount parameter bypasses leverage (exact shares bought)."""
        broker = Broker(datasource)
        broker.leverage = 2.0  # 2x leverage
        broker._i = 0
        
        # Buy exact 100 shares
        broker.buy(amount=np.float64(100))
        
        order = broker.orders[0]
        
        # When amount is specified, leverage is NOT applied
        assert order.quantity == 100

    def test_leverage_validation_minimum(self, broker):
        """Test that leverage below 0.1 is rejected in backtester."""
        # This is tested at the backtester level
        from quantex.backtester import SimpleBacktester
        from quantex.strategy import Strategy
        
        class TestStrategy(Strategy):
            def init(self):
                pass
            def next(self):
                pass
        
        with pytest.raises(ValueError, match="leverage must be at least 0.1"):
            SimpleBacktester(TestStrategy(), leverage=0.05)

    def test_leverage_validation_maximum(self, broker):
        """Test that leverage above 100 is rejected in backtester."""
        from quantex.backtester import SimpleBacktester
        from quantex.strategy import Strategy
        
        class TestStrategy(Strategy):
            def init(self):
                pass
            def next(self):
                pass
        
        with pytest.raises(ValueError, match="leverage cannot exceed 100"):
            SimpleBacktester(TestStrategy(), leverage=150)

    def test_leverage_commission_on_full_position(self, datasource):
        """Test that commission is calculated on full position value, only commission deducted from cash."""
        broker = Broker(datasource)
        broker.leverage = 2.0
        broker._i = 0
        broker.source.current_index = 0
        
        initial_cash = broker.cash
        
        # Buy using quantity (not amount) to test leverage effect
        broker.buy(quantity=0.5)  # Will be amplified by leverage to ~98 shares
        
        broker._iterate(0)
        
        # Commission should be on full leveraged position value
        expected_position = 98.0  # (10000 * 0.5) / 102 * 2
        expected_commission = expected_position * datasource.COpen * broker.commision
        
        # Only commission is deducted from cash (not margin)
        expected_cash = initial_cash - expected_commission
        
        assert abs(broker.cash - expected_cash) < 1.0
