from enum import Enum


class CommissionType(Enum):
    """
    Enumeration for commission calculation types.
    
    This enum defines how commissions are calculated for trades
    in the backtesting framework.
    
    Attributes:
        PERCENTAGE (int): Commission calculated as percentage of trade value (0).
            Commission = trade_value * commission_rate
        CASH (int): Commission calculated as fixed cash amount per lot (1).
            Commission = quantity * commission_rate / lot_size
    """
    PERCENTAGE = 0
    CASH = 1