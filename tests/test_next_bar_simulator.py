import numpy as np
from datetime import datetime, timedelta

from quantex.execution import NextBarSimulator
from quantex.models import Order, Portfolio


def test_next_bar_simulator_delays_execution():
    """Orders submitted on bar *t* should fill at bar *t+1*."""

    portfolio = Portfolio(cash=1_000)
    sim = NextBarSimulator(portfolio)

    # Bar timestamps & prices
    ts1 = datetime(2020, 1, 1)
    ts2 = ts1 + timedelta(days=1)

    # Submit a buy order during *ts1* – it should NOT be filled immediately
    order = Order(
        id="o1",
        symbol="TEST",
        side="buy",
        quantity=10,
        order_type="market",
        timestamp=ts1,
    )

    # Execute queues the order, returns None
    assert sim.execute(order, execution_price=10.0, timestamp=ts1) is None
    assert portfolio.cash == 1_000  # cash unchanged until fill occurs

    # Price row for *ts2* (close price only)
    price_row = np.array([12.0])
    symbol_idx = {"TEST": 0}

    fills = sim.flush_pending(ts2, price_row.tolist(), symbol_idx)

    # Exactly one fill should be generated
    assert len(fills) == 1
    fill = fills[0]
    assert fill.timestamp == ts2
    assert fill.price == 12.0

    # Portfolio should now reflect the filled position
    assert portfolio.positions["TEST"].position == 10
    assert portfolio.cash == 1_000 - 10 * 12.0


def test_next_bar_simulator_fill_at_close():
    """When fill_at='close', simulator should use given close price row."""
    portfolio = Portfolio(cash=1000)
    sim = NextBarSimulator(portfolio, fill_at="close")

    ts1 = datetime(2020, 1, 1)
    ts2 = ts1 + timedelta(days=1)

    order = Order(
        id="o",
        symbol="TEST",
        side="buy",
        quantity=10,
        order_type="market",
        timestamp=ts1,
    )
    sim.execute(order, 10.0, ts1)

    # price_row close 15
    price_row = [15.0]
    fills = sim.flush_pending(ts2, price_row, {"TEST": 0})
    assert fills[0].price == 15.0
