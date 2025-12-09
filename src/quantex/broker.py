from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import final
import numpy as np
import pandas as pd
from .datasource import DataSource
from .enums import CommissionType


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


def same_sign(num1, num2):
    """
    Check if two numbers have the same sign (both positive or both negative).
    
    This utility function is used to determine if two numerical values
    are on the same side of zero, which is important for position
    management and averaging price calculations.
    
    Args:
        num1: First number to compare.
        num2: Second number to compare.
        
    Returns:
        bool: True if both numbers are positive or both are negative,
              False if they have opposite signs or either is zero.
              
    Example:
        >>> same_sign(5, 10)
        True
        >>> same_sign(-3, -7)
        True
        >>> same_sign(2, -5)
        False
        >>> same_sign(0, 5)
        False
    """
    if (num1 > 0 and num2 > 0):
        return True
    elif (num1 < 0 and num2 < 0):
        return True
    return False

class Broker:
    """
    Broker class for managing trading positions and order execution.
    
    This class simulates a brokerage environment, handling position management,
    order processing, commission calculations, and P&L tracking. It provides
    the core trading infrastructure for backtesting strategies.
    
    The broker manages:
    - Current position and average price
    - Cash management and margin calls
    - Order execution (market and limit orders)
    - Commission calculations
    - Stop loss and take profit order management
    - P&L record tracking
    
    Example:
        >>> source = CSVDataSource("data.csv")  
        >>> broker = Broker(source)  
        >>> broker.buy(quantity=0.1)  # Buy 10% of available cash  
        >>> broker.sell(quantity=0.5)  # Sell 50% of available position  
    """
    def __init__(self, source: DataSource):
        """
        Initialize the broker with a data source.
        
        Args:
            source (DataSource): The data source providing market data
                (OHLCV prices and timestamps) for the broker to operate on.
        """
        self.position: np.float64 = np.float64(0)
        self.position_avg_price: np.float64 = np.float64(0)
        self.cash: np.float64 = np.float64(10_000)
        self.commision: np.float64 = np.float64(0.002)
        self.commision_type: CommissionType = CommissionType.PERCENTAGE
        self.lot_size: int = 1
        self.margin_call: float = 0.5 ## 50% of the cash value
        self.share_decimals = 1
        self.orders: list[Order] = []
        self.complete_orders = []
        self._i = 0
        self.source = source
        self.PnLRecord = np.full(len(self.source.data['Close']), self.cash, dtype=np.float64)

    @final
    def buy(self, quantity: float = 1, limit: np.float64 | None = None, amount: np.float64 | None = None, stop_loss: np.float64 | None = None, take_profit: np.float64 | None = None):
        """
        Place a buy order for a specified quantity.
        
        This method creates a buy order with optional limit price, stop loss,
        and take profit conditions. The quantity can be specified as a
        percentage of available cash or as an absolute amount.
        
        Args:
            quantity (float, optional): Quantity to buy as fraction of available
                cash (0 < quantity <= 1). Defaults to 1 (full cash amount).
                For example: 0.5 = buy with 50% of available cash.
            limit (np.float64 | None, optional): Limit price for the order.
                If None, creates a market order. Defaults to None.
            amount (np.float64 | None, optional): Absolute number of shares
                to buy. If provided, overrides quantity calculation.
                Defaults to None.
            stop_loss (np.float64 | None, optional): Stop loss price.
                If provided, creates a stop loss order that executes when
                price falls to this level. Defaults to None.
            take_profit (np.float64 | None, optional): Take profit price.
                If provided, creates a take profit order that executes when
                price rises to this level. Defaults to None.
                
        Raises:
            ValueError: If quantity is not in (0, 1] range.
            ValueError: If limit price is negative.
            ValueError: If amount is negative.
            ValueError: If attempting to buy more than available cash balance.
            
        Note:
            - If both quantity and amount are provided, amount takes precedence.
            - Orders are added to the broker's order queue and processed
              during the next iteration.
            - Market orders execute immediately at current open price.
            - Limit orders only execute when price reaches the specified level.
              
        Example:
            >>> broker = Broker(source)  
            >>> # Buy with 25% of available cash  
            >>> broker.buy(quantity=0.25)  
            >>> # Buy exactly 100 shares at limit price $50  
            >>> broker.buy(amount=100, limit=50.0)  
            >>> # Buy with stop loss and take profit  
            >>> broker.buy(quantity=0.1, stop_loss=45.0, take_profit=55.0)  
        """
        ## Default to full account buy
        if (quantity > 1 or quantity <= 0):
            raise ValueError("Quantity must be between 0 and 1")
        if (limit and limit < 0):
            raise ValueError("Cannot have a negative limit price")
        if (amount and amount < 0):
            raise ValueError("Cannot have a negative amount!")
        if (limit):
            type = OrderType.LIMIT
        else:
            type = OrderType.MARKET
        current_price = self.source.Close[-1]
        total_shares = round((self.cash * quantity) / current_price, self.share_decimals)
        if (amount):
            total_shares = amount
        order = Order(
            side=OrderSide.BUY, 
            quantity=total_shares, 
            type=type,
            price=limit,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=OrderStatus.PENDING,
            timestamp=self.source.Index[self._i]
            )
        ## Transmit the order
        self.orders.append(order)
            

    @final
    def sell(self, quantity: float = 1, limit = None, amount: np.float64 | None = None, stop_loss: np.float64 | None = None, take_profit: np.float64 | None = None):
        """
        Place a sell order for a specified quantity.
        
        This method creates a sell order with optional limit price, stop loss,
        and take profit conditions. The quantity can be specified as a
        percentage of current position or as an absolute amount.
        
        Args:
            quantity (float, optional): Quantity to sell as fraction of current
                position (0 < quantity <= 1). Defaults to 1 (full position).
                For example: 0.5 = sell 50% of current position.
            limit (optional): Limit price for the order (same as limit parameter
                in buy method). If None, creates a market order. Defaults to None.
            amount (np.float64 | None, optional): Absolute number of shares
                to sell. If provided, overrides quantity calculation.
                Defaults to None.
            stop_loss (np.float64 | None, optional): Stop loss price for
                the position. Defaults to None.
            take_profit (np.float64 | None, optional): Take profit price for
                the position. Defaults to None.
                
        Raises:
            ValueError: If quantity is not in (0, 1] range.
            ValueError: If limit price is negative.
            ValueError: If amount is negative.
            ValueError: If attempting to sell more shares than currently held.
            
        Note:
            - If both quantity and amount are provided, amount takes precedence.
            - Orders are added to the broker's order queue and processed
              during the next iteration.
            - Selling reduces the current position and increases cash balance.
            - Stop loss and take profit apply to remaining position after sale.
              
        Example:
            >>> broker = Broker(source)  
            >>> broker.buy(quantity=1)  # First buy full position  
            >>> # Sell 50% of position  
            >>> broker.sell(quantity=0.5)  
            >>> # Sell exactly 100 shares at limit price $52  
            >>> broker.sell(amount=100, limit=52.0)  
        """
        ## Default to full account size sell
        if (quantity > 1 or quantity <= 0):
            raise ValueError("Quantity must be between 0 and 1")
        if (limit and limit < 0):
            raise ValueError("Cannot have a negative limit price")
        if (amount and amount < 0):
            raise ValueError("Cannot have a negative amount!")
        if (limit):
            type = OrderType.LIMIT
        else:
            type = OrderType.MARKET
        current_price = self.source.Close[-1]
        total_shares = round((self.cash * quantity) / current_price, self.share_decimals)
        if (amount):
            total_shares = amount
        order = Order(
            side=OrderSide.SELL, 
            quantity=total_shares, 
            type=type,
            price=limit,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=OrderStatus.PENDING,
            timestamp=self.source.Index[self._i]
            )
        ## Transmit the order
        self.orders.append(order)
    
    @final
    def close(self):
        """
        Close all current positions with market orders.
        
        This method creates market orders to liquidate the entire current
        position, regardless of whether it's long or short. It's commonly
        used at the end of a backtest or when exiting all positions.
        
        The method creates:
        - A sell order if holding a long position (positive shares)
        - A buy order if holding a short position (negative shares)
        
        Note:
            - This method only creates orders; they are executed during
              the next iteration cycle.
            - No parameters are required as it closes the entire position.
            - Commission and slippage will apply to the closing trades.
            - Existing stop loss and take profit orders remain active
              unless explicitly cancelled.
              
        Example:
            >>> broker = Broker(source)  
            >>> broker.buy(quantity=1)  # Open long position  
            >>> # Later, close all positions  
            >>> broker.close()  
        """
        ## Close active and complete positions
        if (self.position > 0):
            order = Order(
                side=OrderSide.SELL, 
                quantity=self.position, 
                type=OrderType.MARKET,
                price=None,
                stop_loss=None,
                take_profit=None,
                status=OrderStatus.PENDING,
                timestamp=self.source.Index[self._i]
                )
            self.orders.append(order)
        elif (self.position < 0):
            order = Order(
                side=OrderSide.BUY, 
                quantity=-self.position, 
                type=OrderType.MARKET,
                price=None,
                stop_loss=None,
                take_profit=None,
                status=OrderStatus.PENDING,
                timestamp=self.source.Index[self._i]
                )
            self.orders.append(order)
    
    def is_long(self):
        """
        Check if currently holding a long position.
        
        Returns:
            bool: True if position is positive (long), False otherwise.
            
        Example:
            >>> broker.buy(quantity=0.5)  
            >>> broker.is_long()  
            True
        """
        return self.position > 0
    
    def is_short(self):
        """
        Check if currently holding a short position.
        
        Returns:
            bool: True if position is negative (short), False otherwise.
            
        Example:
            >>> broker.sell(quantity=0.5)  
            >>> broker.is_short()  
            True
        """
        return self.position < 0
    
    def is_closed(self):
        """
        Check if currently not holding any position.
        
        Returns:
            bool: True if position is zero (no open position), False otherwise.
            
        Example:
            >>> broker.close()  
            >>> broker.is_closed()  
            True
        """
        return self.position == 0

    def _debit(self, amount: np.float64): ## Give money to the market (buy shares)
        """
        Deduct amount from cash balance (internal method).
        
        This is an internal method used to reduce the cash balance when
        purchasing shares or paying commissions.
        
        Args:
            amount (np.float64): Amount to deduct from cash balance.
            
        Raises:
            ValueError: If attempting to deduct more than available cash.
            
        Note:
            This is an internal method and should not be called directly
            by strategy code. Use public methods like buy() instead.
        """
        if (self.cash - amount < 0):
            ## Order fail
            raise ValueError("Tried to purchase more than account balance")
        self.cash -= amount

    def _credit(self, amount: np.float64): ## Take money from the market (sell shares)
        """
        Add amount to cash balance (internal method).
        
        This is an internal method used to increase the cash balance when
        selling shares or receiving funds.
        
        Args:
            amount (np.float64): Amount to add to cash balance.
            
        Note:
            This is an internal method and should not be called directly
            by strategy code. Use public methods like sell() instead.
        """
        self.cash += amount
    

    def _calc_commission(self, quantity: np.float64, price: np.float64):
        """
        Calculate commission for a trade (internal method).
        
        This method computes the commission based on the configured
        commission type and rate.
        
        Args:
            quantity (np.float64): Number of shares/contracts traded.
            price (np.float64): Execution price per share/contract.
            
        Returns:
            np.float64: Commission amount to be charged.
            
        Note:
            - This is an internal method used by _apply_commission.
            - Commission type can be PERCENTAGE (percentage of trade value)
              or CASH (fixed amount per lot).
        """
        if self.commision_type == CommissionType.CASH:
            debit = quantity * self.commision / self.lot_size
        else:
            debit = quantity * price * self.commision
        return debit

    def _apply_commission(self, quantity: np.float64, price: np.float64):
        """
        Apply commission to the trade (internal method).
        
        This method calculates and deducts commission from the cash balance.
        
        Args:
            quantity (np.float64): Number of shares/contracts traded.
            price (np.float64): Execution price per share/contract.
            
        Note:
            This is an internal method and should not be called directly
            by strategy code. Commission is automatically applied during
            order execution.
        """
        debit = self._calc_commission(quantity, price)
        self._debit(debit)

    def _iterate(self, current_index: int):
        """
        Process orders and update broker state for current time step (internal method).
        
        This is the core method that handles order execution, position updates,
        P&L calculations, and margin call checks. It's called for each time step
        in the backtest loop.
        
        The method performs:
        1. Order processing and execution (market and limit orders)
        2. Position and average price updates
        3. Stop loss and take profit order management
        4. P&L and equity calculations
        5. Margin call checks and enforcement
        6. P&L record updates
        
        Args:
            current_index (int): Current time step index in the data source.
            
        Note:
            This is an internal method called automatically by the backtester
            and should not be called directly by strategy code.
        """
        self._i = current_index
        to_delete = []
        ## Do one loop to see if you can execute any orders
        for order in self.orders:
            ## Check for new orders
            match order.status:
                case OrderStatus.PENDING:
                    if (order.type == OrderType.LIMIT):
                        if (order.side == OrderSide.BUY):
                            if (not order.price == None and self.source.COpen <= order.price):
                                ## We can buy it
                                old_pos = self.position
                                new_pos = old_pos + order.quantity
                                if (old_pos == 0):
                                    self.position_avg_price = order.price
                                elif same_sign(old_pos, new_pos):
                                    if (abs(new_pos) > abs(old_pos)):
                                        self.position_avg_price = (old_pos * self.position_avg_price + order.quantity * order.price) / new_pos
                                else:
                                    self.position_avg_price = order.price
                                self._debit(order.price * order.quantity)
                                self._apply_commission(order.quantity, order.price)
                                self.position = new_pos
                        else:
                            if (not order.price == None and self.source.COpen >= order.price):
                                ## We can sell it
                                old_pos = self.position
                                new_pos = old_pos - order.quantity
                                if (old_pos == 0):
                                    self.position_avg_price = order.price
                                elif same_sign(old_pos, new_pos):
                                    if (abs(new_pos) > abs(old_pos)):
                                        self.position_avg_price = (old_pos * self.position_avg_price + order.quantity * order.price) / new_pos
                                else:
                                    self.position_avg_price = order.price
                                self._credit(order.price * order.quantity)
                                self._apply_commission(order.quantity, order.price)
                                self.position = new_pos
                        if (order.stop_loss or order.take_profit):
                            order.status = OrderStatus.ACTIVE ## Will need to be checked on for each update
                        else:
                            order.status = OrderStatus.COMPLETE ## We are done with it
                            to_delete.append(order)
                    else:
                        try:
                            if (order.side == OrderSide.BUY):
                                old_pos = self.position
                                new_pos = old_pos + order.quantity
                                price = self.source.COpen
                                if (old_pos == 0):
                                    self.position_avg_price = price
                                elif same_sign(old_pos, new_pos):
                                    if (abs(new_pos) > abs(old_pos)):
                                        self.position_avg_price = (old_pos * self.position_avg_price + order.quantity * price) / new_pos
                                else:
                                    self.position_avg_price = price
                                self._debit(self.source.COpen * order.quantity)
                                self._apply_commission(order.quantity, self.source.COpen)
                                self.position = new_pos
                            else:
                                old_pos = self.position
                                new_pos = old_pos - order.quantity
                                price = self.source.COpen
                                if (old_pos == 0):
                                    self.position_avg_price = price
                                elif same_sign(old_pos, new_pos):
                                    if (abs(new_pos) > abs(old_pos)):
                                        self.position_avg_price = (old_pos * self.position_avg_price + order.quantity * price) / new_pos
                                else:
                                    self.position_avg_price = price
                                self._credit(self.source.COpen * order.quantity)
                                self._apply_commission(order.quantity, self.source.COpen)
                                self.position = new_pos
                            if (order.stop_loss or order.take_profit):
                                order.status = OrderStatus.ACTIVE
                            else:
                                order.status = OrderStatus.COMPLETE
                                self.complete_orders.append(order)
                                to_delete.append(order)
                        except:
                            pass
                case OrderStatus.ACTIVE:
                    if (
                        order.side == OrderSide.BUY 
                        and (
                            (order.take_profit and self.source.COpen >= order.take_profit)
                            or (order.stop_loss and self.source.COpen <= order.stop_loss)
                            )):
                            close_order = Order(
                                side=OrderSide.SELL, 
                                quantity=order.quantity, 
                                type=OrderType.MARKET, 
                                price= None, 
                                stop_loss= None, 
                                take_profit= None, 
                                status=OrderStatus.PENDING,
                                timestamp=self.source.Index[self._i]
                                )
                            self.orders.append(close_order)
                            order.status = OrderStatus.COMPLETE
                            self.complete_orders.append(order)
                            to_delete.append(order)
                    elif(order.side == OrderSide.SELL
                         and (
                             (order.take_profit and self.source.COpen <= order.take_profit) 
                             or (order.stop_loss and self.source.COpen >= order.stop_loss)
                             )):
                            close_order = Order(
                                side=OrderSide.BUY,
                                quantity=order.quantity,
                                type=OrderType.MARKET,
                                price=None,
                                stop_loss=None,
                                take_profit=None,
                                status=OrderStatus.PENDING,
                                timestamp=self.source.Index[self._i]
                            )
                            self.orders.append(close_order)
                            order.status = OrderStatus.COMPLETE
                            self.complete_orders.append(order)
                            to_delete.append(order)
        for item in to_delete:
            self.orders.remove(item)
        unrealized = self.position * self.source.CClose
        equity = self.cash + unrealized
        margin_call = self.margin_call * abs(self.position) * self.source.CClose
        if equity < margin_call and self.position < 0:
            self.close() ## Close all positions immediately, margin call
        self.PnLRecord[self._i] = equity