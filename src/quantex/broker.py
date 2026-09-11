from dataclasses import dataclass
import pandas as pd
from enum import Enum
from quantex.datasource import PricingData


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"


class Direction(Enum):
    BUY = 1
    SELL = -1


@dataclass
class Order:
    timestamp: pd.Timestamp
    type: OrderType
    direction: Direction
    amount: float
    price: float | None


def execute_condition(order: Order, price: float):
    """
    Figures out if an order should execute and executes it based on it.

    Parameters:
        - order: The Order object
        - price: Current market price
    """
    if order.type == OrderType.MARKET:
        return True
    elif order.type == OrderType.LIMIT and order.price:
        if order.direction == Direction.BUY and price <= order.price:
            return True
        elif order.direction == Direction.SELL and price >= order.price:
            return True
    elif order.type == OrderType.STOP and order.price:
        if order.direction == Direction.BUY and price >= order.price:
            return True
        elif order.direction == Direction.SELL and price <= order.price:
            return True
    return False


class Broker:
    orderQueue: dict[str, list[Order]]  ## Orders to be processed
    processedOrders: dict[str, list[Order]]  ## Orders that have been processed
    openPositions: dict[str, list[Order]]  ## Orders that are currently open

    def __init__(self, context: PricingData, cash: float = 10_000):
        self.cash = cash
        self.total_trades = 0
        self.orderQueue = {}
        self.processedOrders = {}
        self.openPositions = {}
        self.__context__ = context
        self.orderQueue = {}
        for name in self.__context__.datas.keys():
            self.orderQueue[name] = []
            self.processedOrders[name] = []
            self.openPositions[name] = []

    def __process_order__(self, order: Order, name: str):
        """
        Processes an order and updates the broker accordingly.
        Currently does not support margin or futures contracts.
        """
        openPositionsDirection = None
        if self.openPositions[name] and len(self.openPositions[name]) > 0:
            openPositionsDirection = self.openPositions[name][0].direction
        if openPositionsDirection and openPositionsDirection != order.direction:
            price: float = self.__context__.datas[name].Close[-1]
            if execute_condition(order, price):
                total = order.amount * price
                amount = order.amount
                for i in range(len(self.openPositions[name])):
                    open_amount = self.openPositions[name][i].amount
                    closed_amount = min(open_amount, amount)
                    self.openPositions[name][i].amount -= closed_amount
                    amount -= closed_amount
                    if amount == 0:
                        break
                positions = []
                for openOrder in self.openPositions[name]:
                    if openOrder.amount > 0:
                        positions.append(
                            openOrder
                        )  ## This order has not been fully executed
                    else:
                        self.processedOrders[name].append(
                            openOrder
                        )  ## This order has been fully executed
                if len(positions) == 0:
                    self.total_trades += 1
                if len(positions) == 0 and amount > 0:
                    positions.append(
                        Order(
                            order.timestamp,
                            order.type,
                            order.direction,
                            amount,
                            order.price,
                        )
                    )
                else:
                    self.processedOrders[name].append(
                        order
                    )  ## This order has been fully executed
                self.openPositions[name] = positions
                if order.direction == Direction.BUY:
                    self.cash -= total
                else:
                    self.cash += total
        else:
            price: float = self.__context__.datas[name].Close[-1]
            if execute_condition(order, price):
                total = order.amount * price
                self.openPositions[name].append(order)
                if order.direction == Direction.BUY:
                    self.cash -= total
                else:
                    self.cash += total

    def __process_orders__(self):
        for name in self.orderQueue.keys():
            for order in self.orderQueue[name]:
                self.__process_order__(order, name)
            self.orderQueue[name] = []

    def equity(self) -> float:
        """
        Get the current equity at the current time.
        Returns:
            float: The current equity.
        """
        equity = self.cash
        for name in self.orderQueue.keys():
            for order in self.openPositions[name]:
                prices = self.__context__.datas[name].Close
                if order.direction == Direction.BUY:
                    equity += order.amount * prices[-1]
                else:
                    equity -= order.amount * prices[-1]
        return equity

    def buy(
        self,
        name: str | None = None,
        amount: float = 1,
        limit: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        data = self.__context__.datas[name]
        if limit is not None:
            self.orderQueue[name].append(
                Order(data.Timestamp[-1], OrderType.LIMIT, Direction.BUY, amount, limit)
            )
        else:
            self.orderQueue[name].append(
                Order(data.Timestamp[-1], OrderType.MARKET, Direction.BUY, amount, None)
            )
        if stop_loss is not None:
            price = limit or data.Close[-1]
            if stop_loss < price:
                self.orderQueue[name].append(
                    Order(
                        data.Timestamp[-1],
                        OrderType.STOP,
                        Direction.SELL,
                        amount,
                        stop_loss,
                    )
                )
            else:
                raise ValueError(
                    "Stop loss price must be less than current price for BUY order"
                )
        if take_profit is not None:
            price = limit or data.Close[-1]
            if take_profit > price:
                self.orderQueue[name].append(
                    Order(
                        data.Timestamp[-1],
                        OrderType.STOP,
                        Direction.SELL,
                        amount,
                        take_profit,
                    )
                )
            else:
                raise ValueError(
                    "Take profit price must be greater than current price for BUY order"
                )

    def sell(
        self,
        name: str | None = None,
        amount: float = 1,
        limit: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        data = self.__context__.datas[name]
        if limit is not None:
            self.orderQueue[name].append(
                Order(
                    data.Timestamp[-1], OrderType.LIMIT, Direction.SELL, amount, limit
                )
            )
        else:
            self.orderQueue[name].append(
                Order(
                    data.Timestamp[-1], OrderType.MARKET, Direction.SELL, amount, None
                )
            )
        if stop_loss is not None:
            price = limit or data.Close[-1]
            if stop_loss > price:
                self.orderQueue[name].append(
                    Order(
                        data.Timestamp[-1],
                        OrderType.STOP,
                        Direction.BUY,
                        amount,
                        stop_loss,
                    )
                )
            else:
                raise ValueError(
                    "Stop loss price must be greater than current price for SELL order"
                )
        if take_profit is not None:
            price = limit or data.Close[-1]
            if take_profit < price:
                self.orderQueue[name].append(
                    Order(
                        data.Timestamp[-1],
                        OrderType.STOP,
                        Direction.BUY,
                        amount,
                        take_profit,
                    )
                )
            else:
                raise ValueError(
                    "Take profit price must be less than current price for SELL order"
                )

    def is_long(self, name: str | None = None) -> bool:
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        if len(self.openPositions[name]) == 0:
            return False
        return self.openPositions[name][0].direction == Direction.BUY

    def is_short(self, name: str | None = None) -> bool:
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        if len(self.openPositions[name]) == 0:
            return False
        return self.openPositions[name][0].direction == Direction.SELL

    def is_closed(self, name: str | None = None) -> bool:
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        if len(self.openPositions[name]) == 0:
            return True
        return False
