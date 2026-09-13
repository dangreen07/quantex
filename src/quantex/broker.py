from dataclasses import dataclass
import pandas as pd
from enum import Enum
from quantex.datasource import PricingData


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"


class OrderDirection(Enum):
    BUY = 1
    SELL = -1


@dataclass
class Order:
    id: int
    timestamp: pd.Timestamp
    type: OrderType
    direction: OrderDirection
    amount: float
    price: float | None
    parentId: int | None


def execute_condition(order: Order, price: float):
    """
    Figures out if an order should execute and executes it based on it.
    """
    if order.type == OrderType.MARKET:
        return True
    elif order.type == OrderType.LIMIT and order.price:
        if order.direction == OrderDirection.BUY and price <= order.price:
            return True
        elif order.direction == OrderDirection.SELL and price >= order.price:
            return True
    elif order.type == OrderType.STOP and order.price:
        if order.direction == OrderDirection.BUY and price >= order.price:
            return True
        elif order.direction == OrderDirection.SELL and price <= order.price:
            return True
    return False


class Broker:
    orderQueue: dict[str, list[Order]]  ## Orders to be processed
    processedOrders: dict[str, list[Order]]  ## Orders that have been processed
    openPositions: dict[str, list[Order]]  ## Orders that are currently open
    __orderId__: int

    def __init__(self, context: PricingData, cash: float = 10_000):
        self.cash = cash
        self.total_trades = 0
        self.orderQueue = {}
        self.processedOrders = {}
        self.openPositions = {}
        self.__context__ = context
        self.__orderId__ = 1
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
                        if order.parentId is not None:
                            idx = 0
                            for i in range(len(self.orderQueue[name])):
                                if self.orderQueue[name][i].id == order.id:
                                    continue
                                elif (
                                    self.orderQueue[name][i].parentId == order.parentId
                                ):
                                    idx = i
                                    break
                            self.orderQueue[name].pop(
                                idx
                            )  ## OCO - One Cancels the Other
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
                            order.id,
                            order.timestamp,
                            order.type,
                            order.direction,
                            amount,
                            order.price,
                            order.parentId,
                        )
                    )
                else:
                    self.processedOrders[name].append(
                        order
                    )  ## This order has been fully executed
                self.openPositions[name] = positions
                if order.direction == OrderDirection.BUY:
                    self.cash -= total
                else:
                    self.cash += total
                return True
        else:
            price: float = self.__context__.datas[name].Close[-1]
            if execute_condition(order, price):
                total = order.amount * price
                self.openPositions[name].append(order)
                if order.direction == OrderDirection.BUY:
                    self.cash -= total
                else:
                    self.cash += total
                return True
        return False

    def __process_orders__(self):
        for name in self.orderQueue.keys():
            queue = []
            for order in self.orderQueue[name]:
                if not self.__process_order__(order, name):
                    queue.append(order)
            self.orderQueue[name] = queue

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
                if order.direction == OrderDirection.BUY:
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
        parentId = self.__orderId__
        self.__orderId__ += 1
        if limit is not None:
            self.orderQueue[name].append(
                Order(
                    parentId,
                    data.Timestamp[-1],
                    OrderType.LIMIT,
                    OrderDirection.BUY,
                    amount,
                    limit,
                    None,
                )
            )
        else:
            self.orderQueue[name].append(
                Order(
                    parentId,
                    data.Timestamp[-1],
                    OrderType.MARKET,
                    OrderDirection.BUY,
                    amount,
                    None,
                    None,
                )
            )
        if stop_loss is not None:
            price = limit or data.Close[-1]
            if stop_loss < price:
                self.orderQueue[name].append(
                    Order(
                        self.__orderId__,
                        data.Timestamp[-1],
                        OrderType.STOP,
                        OrderDirection.SELL,
                        amount,
                        stop_loss,
                        parentId,
                    )
                )
                self.__orderId__ += 1
            else:
                raise ValueError(
                    "Stop loss price must be less than current price for BUY order"
                )
        if take_profit is not None:
            price = limit or data.Close[-1]
            if take_profit > price:
                self.orderQueue[name].append(
                    Order(
                        self.__orderId__,
                        data.Timestamp[-1],
                        OrderType.STOP,
                        OrderDirection.SELL,
                        amount,
                        take_profit,
                        parentId,
                    )
                )
                self.__orderId__ += 1
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
        parentId = self.__orderId__
        self.__orderId__ += 1
        if limit is not None:
            self.orderQueue[name].append(
                Order(
                    parentId,
                    data.Timestamp[-1],
                    OrderType.LIMIT,
                    OrderDirection.SELL,
                    amount,
                    limit,
                    None,
                )
            )
        else:
            self.orderQueue[name].append(
                Order(
                    parentId,
                    data.Timestamp[-1],
                    OrderType.MARKET,
                    OrderDirection.SELL,
                    amount,
                    None,
                    None,
                )
            )
        if stop_loss is not None:
            price = limit or data.Close[-1]
            if stop_loss > price:
                self.orderQueue[name].append(
                    Order(
                        self.__orderId__,
                        data.Timestamp[-1],
                        OrderType.STOP,
                        OrderDirection.BUY,
                        amount,
                        stop_loss,
                        parentId,
                    )
                )
                self.__orderId__ += 1
            else:
                raise ValueError(
                    "Stop loss price must be greater than current price for SELL order"
                )
        if take_profit is not None:
            price = limit or data.Close[-1]
            if take_profit < price:
                self.orderQueue[name].append(
                    Order(
                        self.__orderId__,
                        data.Timestamp[-1],
                        OrderType.STOP,
                        OrderDirection.BUY,
                        amount,
                        take_profit,
                        parentId,
                    )
                )
                self.__orderId__ += 1
            else:
                raise ValueError(
                    "Take profit price must be less than current price for SELL order"
                )

    def close(self, name: str | None = None) -> None:
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        totalPositionAmount = 0
        if len(self.openPositions[name]) == 0:
            return
        direction = self.openPositions[name][0].direction
        for order in self.openPositions[name]:
            totalPositionAmount += order.amount
        if direction == OrderDirection.BUY:
            self.orderQueue[name].append(
                Order(
                    self.__orderId__,
                    self.__context__.datas[name].Timestamp[-1],
                    OrderType.MARKET,
                    OrderDirection.SELL,
                    totalPositionAmount,
                    None,
                    None,
                )
            )
            self.__orderId__ += 1
        else:
            self.orderQueue[name].append(
                Order(
                    self.__orderId__,
                    self.__context__.datas[name].Timestamp[-1],
                    OrderType.MARKET,
                    OrderDirection.BUY,
                    totalPositionAmount,
                    None,
                    None,
                )
            )
            self.__orderId__ += 1

    def is_long(self, name: str | None = None) -> bool:
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        if len(self.openPositions[name]) == 0:
            return False
        return self.openPositions[name][0].direction == OrderDirection.BUY

    def is_short(self, name: str | None = None) -> bool:
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        if len(self.openPositions[name]) == 0:
            return False
        return self.openPositions[name][0].direction == OrderDirection.SELL

    def is_closed(self, name: str | None = None) -> bool:
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        if len(self.openPositions[name]) == 0:
            return True
        return False
