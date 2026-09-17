from quantex.broker import OrderDirection
from quantex.margin import Margin, MarginType
import pytest


def test_margin_percentage():
    margin = Margin(
        (0.09, 0.1), MarginType.PERCENTAGE, (0.07, 0.08), MarginType.PERCENTAGE
    )
    assert margin.get_initial_margin(
        direction=OrderDirection.BUY, amount=100, price=100, multiplier=50
    ) == pytest.approx(100 * 100 * 50 * 0.09)
    assert margin.get_initial_margin(
        direction=OrderDirection.SELL, amount=100, price=100, multiplier=50
    ) == pytest.approx(100 * 100 * 50 * 0.1)
    assert margin.get_maintenance_margin(
        direction=OrderDirection.BUY, amount=100, price=100, multiplier=50
    ) == pytest.approx(100 * 100 * 50 * 0.07)
    assert margin.get_maintenance_margin(
        direction=OrderDirection.SELL, amount=100, price=100, multiplier=50
    ) == pytest.approx(100 * 100 * 50 * 0.08)


def test_margin_fixed():
    margin = Margin(
        (90, 100),
        MarginType.FIXED_PER_CONTRACT,
        (70, 80),
        MarginType.FIXED_PER_CONTRACT,
    )
    assert margin.get_initial_margin(
        direction=OrderDirection.BUY, amount=100, price=100, multiplier=50
    ) == pytest.approx(100 * 90)
    assert margin.get_initial_margin(
        direction=OrderDirection.SELL, amount=100, price=100, multiplier=50
    ) == pytest.approx(100 * 100)
    assert margin.get_maintenance_margin(
        direction=OrderDirection.BUY, amount=100, price=100, multiplier=50
    ) == pytest.approx(100 * 70)
    assert margin.get_maintenance_margin(
        direction=OrderDirection.SELL, amount=100, price=100, multiplier=50
    ) == pytest.approx(100 * 80)
