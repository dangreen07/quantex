import pytest
from quantex.strategy import Trade, Positions
from datetime import datetime

def test_trade_str():
    trade = Trade("AAPL", 55.4, 25, datetime.now())
    assert str(trade) == "Trade(symbol=AAPL, price=55.40, quantity=25.0)"

def test_long_position():
    pos = Positions("AAPL")
    pos.buy(50, 55.45, datetime.now())
    pos.sell(25, 55.6, datetime.now())
    pos.sell(25, 55.3, datetime.now())
    assert pos.position == 0
    assert pos.calculate_total_pnl(55.5) == pytest.approx(0.0, abs=1e-8)

def test_short_position():
    pos = Positions("AAPL")
    pos.sell(50, 55.45, datetime.now())
    pos.buy(25, 55.3, datetime.now())
    pos.buy(25, 55, datetime.now())
    assert pos.position == 0
    assert pos.calculate_total_pnl(55.5) == pytest.approx(15.0, abs=1e-8)

def test_correct_pnl():
    pos = Positions("AAPL")
    pos.buy(50, 55.45, datetime.now())
    pos.sell(25, 55.6, datetime.now())
    assert pos.calculate_total_pnl(55.5) == pytest.approx(5.0, abs=1e-8)

def test_flip_long_to_short():
    """Selling more than current long flips to short and calculates correct PnL."""
    pos = Positions("AAPL")
    now = datetime.now()

    pos.buy(50, 10, now)      # Long 50 @ 10
    pos.sell(100, 11, now)    # Close long 50 (+ open short 50) @ 11

    assert pos.position == -50  # Short 50
    # Realized: (11-10)*50 = 50
    assert pos.realized_pnl == pytest.approx(50.0, abs=1e-8)
    # New average price for short side should be 11
    assert pos.average_price == pytest.approx(11.0, abs=1e-8)

    # At price 9: unrealized = (9-11)*(-50) = 100, total = 150
    assert pos.calculate_total_pnl(9) == pytest.approx(150.0, abs=1e-8)

def test_flip_short_to_long():
    """Buying more than current short flips to long and calculates correct PnL."""
    pos = Positions("AAPL")
    now = datetime.now()

    pos.sell(40, 20, now)    # Short 40 @ 20
    pos.buy(80, 18, now)     # Close short 40 (+ open long 40) @ 18

    assert pos.position == 40
    # Realized: covered 40 @ (20 -> 18) = +2 * 40 = 80
    assert pos.realized_pnl == pytest.approx(80.0, abs=1e-8)
    # New cost basis should be 18
    assert pos.average_price == pytest.approx(18.0, abs=1e-8)

    # Price 19: unrealized = (19-18)*40 = 40, total = 120
    assert pos.calculate_total_pnl(19) == pytest.approx(120.0, abs=1e-8)

def test_average_price_updates_same_direction():
    """Average cost basis updates correctly when increasing a position in same direction."""
    now = datetime.now()
    pos_long = Positions("AAPL")
    pos_long.buy(10, 100, now)
    pos_long.buy(10, 120, now)
    assert pos_long.average_price == pytest.approx(110.0, abs=1e-8)

    pos_short = Positions("AAPL")
    pos_short.sell(10, 50, now)
    pos_short.sell(30, 40, now)
    # avg price for short calculated the same way
    assert pos_short.average_price == pytest.approx(42.5, abs=1e-8)
    assert pos_short.position == -40