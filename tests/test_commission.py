"""Tests for commission functionality in execution simulators.

This module tests various commission models with dummy data to ensure
commission is properly calculated, deducted from portfolio cash, and
tracked in Fill objects.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from typing import Sequence

from quantex.models import Order, Portfolio, Bar, Fill
from quantex.execution import ImmediateFillSimulator, NextBarSimulator
from quantex.sources import BacktestingDataSource
from quantex.strategy import Strategy
from quantex.backtest import BacktestRunner


class DummyCommissionDataSource(BacktestingDataSource):
    """Simple data source for commission testing with predefined prices."""

    def __init__(self, prices: list[float], symbol: str = "TEST"):
        """Initialize with a list of prices and generate timestamps.

        Args:
            prices: List of close prices for each bar
            symbol: Symbol name for the instrument
        """
        self.symbol = symbol
        self.index = 0

        # Create timestamps and build DataFrame
        start_time = datetime(2024, 1, 1, 9, 30)  # Market open time
        timestamps = [start_time + timedelta(days=i) for i in range(len(prices))]

        self._df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],  # Slightly higher highs
                "low": [p * 0.99 for p in prices],  # Slightly lower lows
                "close": prices,
                "volume": [1000] * len(prices),
            },
            index=pd.DatetimeIndex(timestamps),
        )

    def __len__(self) -> int:
        return len(self._df)

    def peek_timestamp(self) -> datetime | None:
        if self.index < len(self._df):
            return self._df.index[self.index]
        return None

    def get_current_bar(self) -> Bar:
        row = self._df.iloc[self.index]
        timestamp = self._df.index[self.index]
        return Bar(
            timestamp=timestamp,
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            symbol=self.symbol,
        )

    def get_lookback_data(self, lookback_period: int) -> pd.DataFrame:
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1].copy()

    def get_raw_data(self) -> pd.DataFrame:
        return self._df


class SimpleCommissionStrategy(Strategy):
    """Strategy that makes predictable trades for commission testing."""

    def run(self):
        # Buy on first bar, sell on second bar
        if self.index == 0:
            self.buy("TEST", 10)  # Buy 10 shares
        elif self.index == 1:
            self.close_position("TEST")  # Sell all shares


class MultiTradeStrategy(Strategy):
    """Strategy that makes multiple trades for commission testing."""

    def run(self):
        # Make a trade every bar
        if self.index % 2 == 0:  # Even bars: buy
            self.buy("TEST", 5)
        else:  # Odd bars: sell
            if not self.positions["TEST"].is_closed:
                self.sell("TEST", 5)


# ========== CUSTOM COMMISSION SIMULATORS ==========


class PercentageCommissionSimulator(NextBarSimulator):
    """Commission calculated as percentage of trade value."""

    def __init__(self, portfolio: Portfolio, commission_rate: float = 0.001, **kwargs):
        super().__init__(portfolio, commission=0.0, **kwargs)
        self.commission_rate = commission_rate

    def flush_pending(
        self,
        timestamp: datetime,
        price_row: Sequence[float],
        symbol_idx: dict[str, int],
    ) -> list[Fill]:
        """Override to apply percentage-based commission."""
        fills = []
        pending, self._pending_orders = self._pending_orders, []

        for order in pending:
            # Get execution price using parent logic
            idx = symbol_idx.get(order.symbol)
            if idx is None:
                continue
            execution_price = float(price_row[idx])

            # Calculate percentage-based commission
            trade_value = order.quantity * execution_price
            dynamic_commission = trade_value * self.commission_rate

            # Apply commission and execute
            original_commission = self.commission
            self.commission = dynamic_commission

            try:
                fill = super(NextBarSimulator, self).execute(
                    order, execution_price, timestamp
                )
                fills.append(fill)
            except ValueError:
                # Skip if insufficient cash
                pass
            finally:
                self.commission = original_commission

        return fills


class TieredCommissionSimulator(NextBarSimulator):
    """Commission based on trade value tiers."""

    def __init__(self, portfolio: Portfolio, **kwargs):
        super().__init__(portfolio, commission=0.0, **kwargs)
        # Define commission tiers: (min_value, commission)
        self.commission_tiers = [
            (0, 5.00),  # $5 for trades under $500
            (500, 3.00),  # $3 for trades $500-$1000
            (1000, 1.00),  # $1 for trades over $1000
        ]

    def flush_pending(
        self,
        timestamp: datetime,
        price_row: Sequence[float],
        symbol_idx: dict[str, int],
    ) -> list[Fill]:
        """Override to apply tiered commission."""
        fills = []
        pending, self._pending_orders = self._pending_orders, []

        for order in pending:
            # Get execution price
            idx = symbol_idx.get(order.symbol)
            if idx is None:
                continue
            execution_price = float(price_row[idx])

            # Calculate tiered commission
            trade_value = order.quantity * execution_price
            commission = self.commission_tiers[0][1]  # default
            for min_value, tier_commission in reversed(self.commission_tiers):
                if trade_value >= min_value:
                    commission = tier_commission
                    break

            # Apply commission and execute
            original_commission = self.commission
            self.commission = commission

            try:
                fill = super(NextBarSimulator, self).execute(
                    order, execution_price, timestamp
                )
                fills.append(fill)
            except ValueError:
                # Skip if insufficient cash
                pass
            finally:
                self.commission = original_commission

        return fills


# ========== TESTS ==========


class TestBasicCommission:
    """Test basic commission functionality with both simulators."""

    def test_immediate_fill_fixed_commission(self):
        """Test ImmediateFillSimulator with fixed commission."""
        portfolio = Portfolio(cash=1000)
        simulator = ImmediateFillSimulator(portfolio, commission=2.50)

        order = Order(
            id="test_order",
            symbol="TEST",
            side="buy",
            quantity=10,
            timestamp=datetime.now(),
        )

        fill = simulator.execute(order, execution_price=50.0, timestamp=datetime.now())

        # Check commission was applied
        assert fill.commission == 2.50
        # Check cash was reduced by stock cost + commission
        expected_cash = 1000 - (10 * 50.0) - 2.50
        assert portfolio.cash == expected_cash
        # Check position was created
        assert portfolio.positions["TEST"].position == 10

    def test_next_bar_fixed_commission(self):
        """Test NextBarSimulator with fixed commission."""
        portfolio = Portfolio(cash=1000)
        simulator = NextBarSimulator(portfolio, commission=1.50)

        # Submit order (should be queued)
        order = Order(
            id="test_order",
            symbol="TEST",
            side="buy",
            quantity=8,
            timestamp=datetime(2024, 1, 1),
        )

        result = simulator.execute(
            order, execution_price=60.0, timestamp=datetime(2024, 1, 1)
        )
        assert result is None  # Order queued, not executed yet
        assert portfolio.cash == 1000  # Cash unchanged

        # Flush pending orders
        price_row = [65.0]  # Next bar price
        symbol_idx = {"TEST": 0}
        fills = simulator.flush_pending(datetime(2024, 1, 2), price_row, symbol_idx)

        assert len(fills) == 1
        fill = fills[0]
        assert fill.commission == 1.50
        assert fill.price == 65.0  # Executed at next bar price

        # Check cash and position
        expected_cash = 1000 - (8 * 65.0) - 1.50
        assert portfolio.cash == expected_cash
        assert portfolio.positions["TEST"].position == 8

    def test_commission_insufficient_cash(self):
        """Test that commission is considered in cash sufficiency checks."""
        portfolio = Portfolio(cash=100)  # Low cash
        simulator = ImmediateFillSimulator(portfolio, commission=5.00)

        # Try to buy shares that would exceed available cash when commission is included
        order = Order(
            id="test_order",
            symbol="TEST",
            side="buy",
            quantity=2,
            timestamp=datetime.now(),
        )

        # This should fail: 2 * 50 + 5 = 105 > 100 available cash
        with pytest.raises(ValueError, match="Insufficient cash"):
            simulator.execute(order, execution_price=50.0, timestamp=datetime.now())


class TestCustomCommissionModels:
    """Test custom commission implementations."""

    def test_percentage_commission_simulator(self):
        """Test percentage-based commission calculation."""
        portfolio = Portfolio(cash=10000)
        simulator = PercentageCommissionSimulator(
            portfolio, commission_rate=0.002
        )  # 0.2%

        # Create dummy order
        order = Order(
            id="test_order",
            symbol="TEST",
            side="buy",
            quantity=100,
            timestamp=datetime(2024, 1, 1),
        )

        # Queue the order
        simulator.execute(order, execution_price=50.0, timestamp=datetime(2024, 1, 1))

        # Flush at higher price
        price_row = [60.0]
        symbol_idx = {"TEST": 0}
        fills = simulator.flush_pending(datetime(2024, 1, 2), price_row, symbol_idx)

        fill = fills[0]
        expected_commission = 100 * 60.0 * 0.002  # Trade value * rate
        assert fill.commission == expected_commission

        # Check cash deduction
        expected_cash = 10000 - (100 * 60.0) - expected_commission
        assert portfolio.cash == expected_cash

    def test_tiered_commission_simulator(self):
        """Test tiered commission structure."""
        portfolio = Portfolio(cash=10000)
        simulator = TieredCommissionSimulator(portfolio)

        # Test small trade (should get $5 commission)
        small_order = Order(
            id="small",
            symbol="TEST",
            side="buy",
            quantity=5,
            timestamp=datetime(2024, 1, 1),
        )
        simulator.execute(
            small_order, execution_price=50.0, timestamp=datetime(2024, 1, 1)
        )

        # Test medium trade (should get $3 commission)
        medium_order = Order(
            id="medium",
            symbol="TEST",
            side="buy",
            quantity=15,
            timestamp=datetime(2024, 1, 1),
        )
        simulator.execute(
            medium_order, execution_price=50.0, timestamp=datetime(2024, 1, 1)
        )

        # Test large trade (should get $1 commission)
        large_order = Order(
            id="large",
            symbol="TEST",
            side="buy",
            quantity=25,
            timestamp=datetime(2024, 1, 1),
        )
        simulator.execute(
            large_order, execution_price=50.0, timestamp=datetime(2024, 1, 1)
        )

        # Flush all orders
        price_row = [50.0]
        symbol_idx = {"TEST": 0}
        fills = simulator.flush_pending(datetime(2024, 1, 2), price_row, symbol_idx)

        assert len(fills) == 3

        # Check commission tiers
        commissions = [fill.commission for fill in fills]
        # Small trade: 5 * 50 = $250 -> $5 commission
        # Medium trade: 15 * 50 = $750 -> $3 commission
        # Large trade: 25 * 50 = $1250 -> $1 commission
        expected_commissions = [5.00, 3.00, 1.00]
        assert commissions == expected_commissions


class TestCommissionIntegration:
    """Test commission integration with BacktestRunner and strategies."""

    def test_backtest_runner_with_commission(self):
        """Test end-to-end backtest with commission."""
        # Create test data: prices go 100 -> 110 -> 105
        prices = [100.0, 110.0, 105.0]
        data_source = DummyCommissionDataSource(prices, "TEST")

        # Create strategy and simulator with commission
        strategy = SimpleCommissionStrategy(
            {"source": data_source}, symbols=["TEST"], initial_cash=2000
        )
        simulator = NextBarSimulator(strategy.portfolio, commission=2.00)

        # Run backtest
        runner = BacktestRunner(strategy, {"source": data_source}, simulator=simulator)
        result = runner.run()

        # Should have 2 trades (buy and sell)
        assert len(result.fills) == 2

        # Check commission was applied to both trades
        total_commission = sum(fill.commission for fill in result.fills)
        assert total_commission == 4.00  # 2 trades * $2.00 each

        # Verify commission is reflected in final portfolio value
        buy_fill, sell_fill = result.fills
        assert buy_fill.commission == 2.00
        assert sell_fill.commission == 2.00

    def test_commission_impact_on_performance(self):
        """Test that commission impacts strategy performance metrics."""
        prices = [100.0, 105.0, 110.0, 115.0, 120.0]
        data_source = DummyCommissionDataSource(prices, "TEST")

        # Test with no commission
        strategy_no_comm = MultiTradeStrategy(
            {"source": data_source}, symbols=["TEST"], initial_cash=5000
        )
        simulator_no_comm = NextBarSimulator(strategy_no_comm.portfolio, commission=0.0)
        runner_no_comm = BacktestRunner(
            strategy_no_comm, {"source": data_source}, simulator=simulator_no_comm
        )
        result_no_comm = runner_no_comm.run()

        # Test with high commission
        strategy_with_comm = MultiTradeStrategy(
            {"source": data_source}, symbols=["TEST"], initial_cash=5000
        )
        simulator_with_comm = NextBarSimulator(
            strategy_with_comm.portfolio, commission=10.0
        )
        runner_with_comm = BacktestRunner(
            strategy_with_comm, {"source": data_source}, simulator=simulator_with_comm
        )
        result_with_comm = runner_with_comm.run()

        # Commission should reduce total return
        return_no_comm = result_no_comm.metrics["total_return"]
        return_with_comm = result_with_comm.metrics["total_return"]

        assert return_with_comm < return_no_comm

        # Check total commission paid
        total_commission = sum(fill.commission for fill in result_with_comm.fills)
        assert total_commission > 0

    def test_commission_tracking_in_fills(self):
        """Test that Fill objects properly track commission amounts."""
        portfolio = Portfolio(cash=1000)
        simulator = ImmediateFillSimulator(portfolio, commission=1.25)

        # Create multiple orders with different sizes
        orders = [
            Order(
                id="1", symbol="TEST", side="buy", quantity=10, timestamp=datetime.now()
            ),
            Order(
                id="2", symbol="TEST", side="sell", quantity=5, timestamp=datetime.now()
            ),
            Order(
                id="3", symbol="TEST", side="buy", quantity=3, timestamp=datetime.now()
            ),
        ]

        fills = []
        for order in orders:
            fill = simulator.execute(
                order, execution_price=50.0, timestamp=datetime.now()
            )
            fills.append(fill)

        # All fills should have the same commission (fixed model)
        assert all(fill.commission == 1.25 for fill in fills)

        # Check Fill value() method includes commission in cash impact
        for fill in fills:
            expected_cash_impact = -fill.quantity * fill.price  # Base trade value
            assert (
                fill.value() == expected_cash_impact
            )  # Commission handled separately in portfolio


if __name__ == "__main__":
    pytest.main([__file__])
