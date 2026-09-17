from dataclasses import dataclass
from enum import Enum
import pandas as pd


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


class SearchType(Enum):
    GRID = 1
    OPTUNA = 2


class ApplicationType(Enum):
    """
    The application type for a commission.
    """

    PER_ORDER = 1  ## Applies for each order
    PER_CONTRACT = 2  ## Applies for each contract


class MarginType(Enum):
    FIXED_PER_CONTRACT = 1
    PERCENTAGE = 2
