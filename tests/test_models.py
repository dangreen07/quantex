from datetime import datetime
import pytest

from quantex.models import Bar, Tick, Order, Fill, Position, Portfolio, Trade


def test_bar_and_tick_dataclasses():
    ts = datetime.now()
    bar = Bar(
        timestamp=ts, open=100, high=110, low=95, close=105, volume=1000, symbol="AAPL"
    )
    tick = Tick(timestamp=ts, price=105, volume=50, symbol="AAPL")

    assert bar.close == 105
    assert bar.symbol == "AAPL"
    assert tick.price == 105


def test_order_and_fill_dataclasses():
    ts = datetime.now()
    order = Order(
        id="1",
        symbol="AAPL",
        side="buy",
        quantity=10,
        order_type="market",
        timestamp=ts,
    )
    fill = Fill(
        order_id="1",
        symbol="AAPL",
        quantity=10,
        price=100.5,
        timestamp=ts,
        commission=0.1,
    )

    assert order.side == "buy"
    assert fill.price == 100.5


def test_position_buy_sell_and_pnl():
    ts = datetime.now()
    pos = Position("AAPL")

    # Buy 100 @ 10, Sell 50 @ 11, Sell 50 @ 9
    pos.buy(100, 10, ts)
    pos.sell(50, 11, ts)
    pos.sell(50, 9, ts)

    assert pos.position == 0  # Flat
    # Realized: (11-10)*50 + (9-10)*50 = 50 - 50 = 0
    assert pos.realized_pnl == pytest.approx(0.0)
    assert pos.calculate_total_pnl(10) == pytest.approx(0.0)


def test_portfolio_cash_and_nav():
    ts = datetime.now()
    portfolio = Portfolio(cash=10000)

    # Buy 100 shares @ 50
    fill1 = Fill(order_id="1", symbol="AAPL", quantity=100, price=50, timestamp=ts)
    portfolio.process_fill(fill1)

    # Current price 55: unrealized +500
    nav = portfolio.net_asset_value({"AAPL": 55})
    # Starting cash 10000, cash after purchase 5000, shares worth 5500 -> NAV 10500
    assert nav == pytest.approx(10500)

    # Sell 100 @ 55 (realize +500)
    fill2 = Fill(order_id="2", symbol="AAPL", quantity=-100, price=55, timestamp=ts)
    portfolio.process_fill(fill2)

    assert portfolio.cash == pytest.approx(10000 + 500)  # starting cash + realized PnL
    assert portfolio.realized_pnl == pytest.approx(500)


# --- Additional Position & Trade behaviour tests migrated from test_strategy.py ---


def test_trade_str_representation():
    trade = Trade("AAPL", 55.4, 25, datetime.now())
    assert str(trade) == "Trade(symbol=AAPL, price=55.40, quantity=25.0)"


def test_position_long_round_trip():
    pos = Position("AAPL")
    pos.buy(50, 55.45, datetime.now())
    pos.sell(25, 55.6, datetime.now())
    pos.sell(25, 55.3, datetime.now())
    assert pos.position == 0
    assert pos.calculate_total_pnl(55.5) == pytest.approx(0.0, abs=1e-8)


def test_position_short_round_trip():
    pos = Position("AAPL")
    pos.sell(50, 55.45, datetime.now())
    pos.buy(25, 55.3, datetime.now())
    pos.buy(25, 55.0, datetime.now())
    assert pos.position == 0
    assert pos.calculate_total_pnl(55.5) == pytest.approx(15.0, abs=1e-8)


def test_position_flip_long_to_short():
    now = datetime.now()
    pos = Position("AAPL")

    pos.buy(50, 10, now)
    pos.sell(100, 11, now)

    assert pos.position == -50
    assert pos.realized_pnl == pytest.approx(50.0, abs=1e-8)
    assert pos.average_price == pytest.approx(11.0, abs=1e-8)
    assert pos.calculate_total_pnl(9) == pytest.approx(150.0, abs=1e-8)


def test_position_flip_short_to_long():
    now = datetime.now()
    pos = Position("AAPL")

    pos.sell(40, 20, now)
    pos.buy(80, 18, now)

    assert pos.position == 40
    assert pos.realized_pnl == pytest.approx(80.0, abs=1e-8)
    assert pos.average_price == pytest.approx(18.0, abs=1e-8)
    assert pos.calculate_total_pnl(19) == pytest.approx(120.0, abs=1e-8)


def test_average_price_updates_same_direction():
    now = datetime.now()
    pos_long = Position("AAPL")
    pos_long.buy(10, 100, now)
    pos_long.buy(10, 120, now)
    assert pos_long.average_price == pytest.approx(110.0, abs=1e-8)

    pos_short = Position("AAPL")
    pos_short.sell(10, 50, now)
    pos_short.sell(30, 40, now)
    assert pos_short.average_price == pytest.approx(42.5, abs=1e-8)
    assert pos_short.position == -40
