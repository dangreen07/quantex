from quantex.datasource import PricingData
from quantex.commission import Commission
from dataclasses import dataclass
import pandas as pd
from enum import Enum


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"


class OrderDirection(Enum):
    BUY = 1
    SELL = -1


@dataclass
class NewOrder:
    id: int
    transmit_timestamp: pd.Timestamp
    type: OrderType
    direction: OrderDirection
    amount: float
    price: float | None
    parentId: int | None


@dataclass
class Order(NewOrder):
    fill_timestamp: pd.Timestamp
    amount_filled: float
    fill_price: float


class Broker:
    orderQueue: dict[str, list[NewOrder]]  ## Orders to be processed
    cancelQueue: dict[str, list[NewOrder]]  ## Orders that need to be cancelled
    processedOrders: dict[str, list[Order]]  ## Orders that have been processed
    openPositions: dict[str, list[Order]]  ## Orders that are currently open
    __orderId__: int

    def __init__(
        self,
        context: PricingData,
        commission: Commission,
        cash: float = 10_000,
        multiplier: float = 1,
    ):
        self.cash = cash
        self.total_trades = 0
        self.orderQueue = {}
        self.processedOrders = {}
        self.openPositions = {}
        self.cancelQueue = {}
        self.__context__ = context
        self.__orderId__ = 1
        self.multiplier = multiplier
        self.commission = commission
        for name in self.__context__.datas.keys():
            self.orderQueue[name] = []
            self.processedOrders[name] = []
            self.openPositions[name] = []
            self.cancelQueue[name] = []

    def execute_condition(self, order: NewOrder, price: float) -> bool:
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

    def __process_order__(self, order: NewOrder, name: str):
        openPositionsDirection = None
        if self.openPositions[name] and len(self.openPositions[name]) > 0:
            openPositionsDirection = self.openPositions[name][0].direction
        curr_timestamp = self.__context__.datas[name].Timestamp[-1]
        if openPositionsDirection and openPositionsDirection != order.direction:
            price: float = self.__context__.datas[name].Open[-1]
            if self.execute_condition(order, price):
                amount = order.amount
                for i in range(len(self.openPositions[name])):
                    open_amount = self.openPositions[name][i].amount_filled
                    closed_amount = min(open_amount, amount)
                    self.openPositions[name][i].amount_filled -= closed_amount
                    if self.openPositions[name][i].amount_filled == 0:
                        ## Check if the order has any take profit or stop loss and if so, cancel it
                        queueParents = [i.parentId for i in self.orderQueue[name]]
                        if self.openPositions[name][i].id in queueParents:
                            item = self.orderQueue[name][
                                queueParents.index(self.openPositions[name][i].id)
                            ]
                            self.cancelQueue[name].append(item)
                    amount -= closed_amount
                    if amount == 0:
                        if order.parentId is not None:
                            for queued_order in self.orderQueue[name]:
                                if (
                                    queued_order.id != order.id
                                    and queued_order.parentId == order.parentId
                                ):
                                    self.cancelQueue[name].append(
                                        queued_order
                                    )  ## OCO - One Cancels the Other
                        break
                positions = []
                for openOrder in self.openPositions[name]:
                    if openOrder.amount_filled > 0:
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
                    order = Order(
                        id=order.id,
                        transmit_timestamp=order.transmit_timestamp,
                        type=order.type,
                        direction=order.direction,
                        amount=order.amount,
                        price=order.price,
                        parentId=order.parentId,
                        fill_timestamp=curr_timestamp,
                        amount_filled=order.amount - amount,  ## Partially filled
                        fill_price=price,
                    )
                    positions.append(order)
                    self.cash -= self.commission.calculate(order)
                else:
                    order = Order(
                        id=order.id,
                        transmit_timestamp=order.transmit_timestamp,
                        type=order.type,
                        direction=order.direction,
                        amount=order.amount,
                        price=order.price,
                        parentId=order.parentId,
                        fill_timestamp=curr_timestamp,
                        amount_filled=order.amount,  ## Fully filled
                        fill_price=price,
                    )
                    self.processedOrders[name].append(
                        order
                    )  ## This order has been fully executed
                    self.cash -= self.commission.calculate(order)
                self.openPositions[name] = positions
                total = (order.amount - amount) * price * self.multiplier
                if order.direction == OrderDirection.BUY:
                    self.cash -= total
                else:
                    self.cash += total
                return True
        else:
            price: float = self.__context__.datas[name].Open[-1]
            if self.execute_condition(order, price):
                total = order.amount * price * self.multiplier
                order = Order(
                    id=order.id,
                    transmit_timestamp=order.transmit_timestamp,
                    type=order.type,
                    direction=order.direction,
                    amount=order.amount,
                    price=order.price,
                    parentId=order.parentId,
                    fill_timestamp=curr_timestamp,
                    amount_filled=order.amount,  ## Fully filled
                    fill_price=price,
                )
                self.openPositions[name].append(order)
                self.cash -= self.commission.calculate(order)
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
                if order in self.cancelQueue[name]:
                    self.cancelQueue[name].remove(order)
                    continue
                if not self.__process_order__(order, name):
                    queue.append(order)
            self.orderQueue[name] = queue

    def equity(self) -> float:
        """
        Get the current equity at the current time.

        Returns:
            The current equity.
        """
        equity = self.cash
        for name in self.orderQueue.keys():
            for order in self.openPositions[name]:
                prices = self.__context__.datas[name].Close
                initial_value = order.fill_price * order.amount_filled * self.multiplier
                price_change = (prices[-1] - order.fill_price) * self.multiplier
                if order.direction == OrderDirection.BUY:
                    equity += initial_value + price_change * order.amount_filled
                else:
                    equity -= initial_value + price_change * order.amount_filled
        return equity

    def buy(
        self,
        name: str | None = None,
        amount: float = 1,
        limit: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        """
        Places a buy order.

        Parameters:
            name: The name of the data source to buy from. If None, the first data source in the context will be used.
            amount: The amount of shares to buy.
            limit: The limit price to buy at. If None, a market order will be placed.
            stop_loss: The stop loss price to use. If None, no stop loss will be used.
            take_profit: The take profit price to use. If None, no take profit will be used.
        """
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        data = self.__context__.datas[name]
        parentId = self.__orderId__
        self.__orderId__ += 1
        if limit is not None:
            self.orderQueue[name].append(
                NewOrder(
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
                NewOrder(
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
                    NewOrder(
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
                    NewOrder(
                        self.__orderId__,
                        data.Timestamp[-1],
                        OrderType.LIMIT,
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
        """
        Places a sell order.

        Parameters:
            name: The name of the data source to sell from. If None, the first data source in the context will be used.
            amount: The amount of shares to sell.
            limit: The limit price to sell at. If None, a market order will be placed.
            stop_loss: The stop loss price to use. If None, no stop loss will be used.
            take_profit: The take profit price to use. If None, no take profit will be used.
        """
        if amount <= 0:
            raise ValueError("Amount must be greater than 0")
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        data = self.__context__.datas[name]
        parentId = self.__orderId__
        self.__orderId__ += 1
        if limit is not None:
            self.orderQueue[name].append(
                NewOrder(
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
                NewOrder(
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
                    NewOrder(
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
                    NewOrder(
                        self.__orderId__,
                        data.Timestamp[-1],
                        OrderType.LIMIT,
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

    def close(self, data_name: str | None = None) -> None:
        """
        Closes all open positions.

        Parameters:
            name: The name of the data source to close positions from. If None, the first data source in the context will be used.
        """
        names = (
            [data_name]
            if data_name is not None
            else list(self.__context__.datas.keys())
        )
        for name in names:
            if len(self.openPositions[name]) == 0:
                return
            for order in self.openPositions[name]:
                if order.direction == OrderDirection.BUY:
                    self.orderQueue[name].append(
                        NewOrder(
                            self.__orderId__,
                            self.__context__.datas[name].Timestamp[-1],
                            OrderType.MARKET,
                            OrderDirection.SELL,
                            order.amount_filled,
                            None,
                            order.parentId or order.id,
                        )
                    )
                    self.__orderId__ += 1
                elif order.direction == OrderDirection.SELL:
                    self.orderQueue[name].append(
                        NewOrder(
                            self.__orderId__,
                            self.__context__.datas[name].Timestamp[-1],
                            OrderType.MARKET,
                            OrderDirection.BUY,
                            order.amount_filled,
                            None,
                            order.parentId or order.id,
                        )
                    )
                    self.__orderId__ += 1

    def is_long(self, name: str | None = None) -> bool:
        """
        Checks if the broker is long.

        Parameters:
            name: The name of the data source to check. If None, the first data source in the context will be used.

        Returns:
            True if the broker is long, False otherwise.
        """
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        if len(self.openPositions[name]) == 0:
            return False
        return self.openPositions[name][0].direction == OrderDirection.BUY

    def is_short(self, name: str | None = None) -> bool:
        """
        Checks if the broker is short.

        Parameters:
            name: The name of the data source to check. If None, the first data source in the context will be used.

        Returns:
            True if the broker is short, False otherwise.
        """
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        if len(self.openPositions[name]) == 0:
            return False
        return self.openPositions[name][0].direction == OrderDirection.SELL

    def is_closed(self, name: str | None = None) -> bool:
        """
        Checks if the broker is closed.

        Parameters:
            name: The name of the data source to check. If None, the first data source in the context will be used.

        Returns:
            True if the broker is closed, False otherwise.
        """
        if name is None:
            name = list(self.__context__.datas.keys())[0]
        if len(self.openPositions[name]) == 0:
            return True
        return False
