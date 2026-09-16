from quantex.broker import Order, OrderType, OrderDirection
from quantex.commission import Commission, ApplicationType
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
        pd.Timestamp("2022-01-02"),
        0,
        124.5,
    )
    commission = Commission()
    assert commission.calculate(order) == 0


def test_fixed_per_order_commission():
    order = Order(
        1,
        pd.Timestamp("2022-01-01"),
        OrderType.LIMIT,
        OrderDirection.BUY,
        50,
        125,
        None,
        pd.Timestamp("2022-01-02"),
        0,
        124.5,
    )
    commission = Commission(ApplicationType.PER_ORDER, fixed=0.5)
    assert commission.calculate(order) == pytest.approx(0.5)


def test_fixed_per_contract_commission():
    order = Order(
        1,
        pd.Timestamp("2022-01-01"),
        OrderType.LIMIT,
        OrderDirection.BUY,
        50,
        125,
        None,
        pd.Timestamp("2022-01-02"),
        0,
        124.5,
    )
    commission = Commission(ApplicationType.PER_CONTRACT, fixed=0.5)
    assert commission.calculate(order) == pytest.approx(25)


def test_percentage_per_order_commission():
    order = Order(
        1,
        pd.Timestamp("2022-01-01"),
        OrderType.LIMIT,
        OrderDirection.BUY,
        50,
        125,
        None,
        pd.Timestamp("2022-01-02"),
        0,
        124.5,
    )
    commission = Commission(ApplicationType.PER_ORDER, percentage=0.005)
    assert commission.calculate(order) == pytest.approx(31.125)


def test_percentage_per_contract_commission():
    order = Order(
        1,
        pd.Timestamp("2022-01-01"),
        OrderType.LIMIT,
        OrderDirection.BUY,
        50,
        125,
        None,
        pd.Timestamp("2022-01-02"),
        0,
        124.5,
    )
    commission = Commission(ApplicationType.PER_ORDER, percentage=0.005)
    assert commission.calculate(order) == pytest.approx(31.125)
