from quantex.commission import Commission, ApplicationType
from quantex.broker import Order, OrderType, OrderDirection
import pandas as pd
import pytest


def test_no_commission():
    order = Order(
        1,
        pd.Timestamp("2022-01-01"),
        OrderType.LIMIT,
        OrderDirection.BUY,
        50,
        125,
        None,
    )
    commission = Commission()
    assert commission.calculate(order, 124.5) == 0


def test_fixed_per_order_commission():
    order = Order(
        1,
        pd.Timestamp("2022-01-01"),
        OrderType.LIMIT,
        OrderDirection.BUY,
        50,
        125,
        None,
    )
    commission = Commission(ApplicationType.PER_ORDER, fixed=0.5)
    assert commission.calculate(order, 124.5) == pytest.approx(0.5)


def test_fixed_per_contract_commission():
    order = Order(
        1,
        pd.Timestamp("2022-01-01"),
        OrderType.LIMIT,
        OrderDirection.BUY,
        50,
        125,
        None,
    )
    commission = Commission(ApplicationType.PER_CONTRACT, fixed=0.5)
    assert commission.calculate(order, 124.5) == pytest.approx(25)


def test_percentage_per_order_commission():
    order = Order(
        1,
        pd.Timestamp("2022-01-01"),
        OrderType.LIMIT,
        OrderDirection.BUY,
        50,
        125,
        None,
    )
    commission = Commission(ApplicationType.PER_ORDER, percentage=0.005)
    assert commission.calculate(order, 124.5) == pytest.approx(31.125)


def test_percentage_per_contract_commission():
    order = Order(
        1,
        pd.Timestamp("2022-01-01"),
        OrderType.LIMIT,
        OrderDirection.BUY,
        50,
        125,
        None,
    )
    commission = Commission(ApplicationType.PER_ORDER, percentage=0.005)
    assert commission.calculate(order, 124.5) == pytest.approx(31.125)
