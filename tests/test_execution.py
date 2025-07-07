from datetime import datetime

from quantex.models import Order, Portfolio
from quantex.execution import ImmediateFillSimulator


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
