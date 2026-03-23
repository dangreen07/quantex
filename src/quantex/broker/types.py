from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np


class OrderSide(Enum):
    """
    Enumeration for order side (buy/sell direction).
    
    This enum defines whether an order is a buy (long) or sell (short) order.
    
    Attributes:
        BUY (int): Represents a buy order (1).
        SELL (int): Represents a sell order (-1).
    """
    BUY = 1
    SELL = -1

class OrderType(Enum):
    """
    Enumeration for order execution types.
    
    This enum defines how an order should be executed in the market.
    
    Attributes:
        MARKET (int): Market order that executes immediately at current market price (0).
        LIMIT (int): Limit order that executes only at specified price or better (1).
    """
    MARKET = 0
    LIMIT = 1

class OrderStatus(Enum):
    """
    Enumeration for order execution status.
    
    This enum tracks the current state of an order during its lifecycle.
    
    Attributes:
        ACTIVE (int): Order is active with stop loss or take profit conditions (0).
        COMPLETE (int): Order has been fully executed and no further actions needed (1).
        PENDING (int): Order is pending execution, waiting for price or time trigger (2).
    """
    ACTIVE = 0 ## Will go here if there is a stop loss or take profit
    COMPLETE = 1 ## Will go to this state if there are no more actions that can be done with the order
    PENDING = 2 ## Will be here when created, waiting for either the right price, or for time step

@dataclass
class Order:
    """
    Dataclass representing a trading order.
    
    This class encapsulates all the information needed to define and track
    a trading order, including its side, quantity, type, execution conditions,
    and current status.
    
    Attributes:
        side (OrderSide): Whether this is a buy or sell order.
        quantity (np.float64): Number of shares/contracts to trade.
        type (OrderType): Order execution type (market or limit).
        price (np.float64 | None): Limit price for limit orders, None for market orders.
        stop_loss (np.float64 | None): Stop loss price, if any.
        take_profit (np.float64 | None): Take profit price, if any.
        status (OrderStatus): Current status of the order.
        timestamp (datetime): Time when the order was created.
    """
    side: OrderSide
    quantity: np.float64
    type: OrderType
    price: np.float64 | None
    stop_loss: np.float64 | None
    take_profit: np.float64 | None
    status: OrderStatus
    timestamp: datetime
    reduce_only: bool = False