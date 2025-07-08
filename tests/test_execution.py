from datetime import datetime

from quantex.models import Order, Portfolio
from quantex.execution import ImmediateFillSimulator
import pytest


def test_immediate_fill_execution():
    portfolio = Portfolio(cash=1000)
    simulator = ImmediateFillSimulator(portfolio)

    order = Order(
        id="1",
        symbol="TEST",
        side="buy",
        quantity=10,
        order_type="market",
        timestamp=datetime.now(),
    )

    fill = simulator.execute(order, execution_price=10.0, timestamp=datetime.now())

    # Check that cash decreased by cost of purchase (no commission/slippage)
    assert portfolio.cash == 1000 - 100
    # Position size should match order quantity
    assert portfolio.positions["TEST"].position == 10
    # Fill object should reflect signed quantity (+10 for buy)
    assert fill.quantity == 10


def test_immediate_fill_insufficient_cash():
    portfolio = Portfolio(cash=50)
    simulator = ImmediateFillSimulator(portfolio)

    order = Order(
        id="2",
        symbol="TEST",
        side="buy",
        quantity=10,  # Would cost 100 > 50 cash
        order_type="market",
        timestamp=datetime.now(),
    )

    with pytest.raises(ValueError, match="Insufficient cash"):
        simulator.execute(order, execution_price=10.0, timestamp=datetime.now())
