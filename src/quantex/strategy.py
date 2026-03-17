from abc import ABC, abstractmethod
from typing import final
import numpy as np
from .broker import Broker
from .datasource import DataSource
from .helpers import TimeNDArray
from .indicators import indicators as builtin_indicators


class Strategy(ABC):
    """
    Abstract base class for trading strategies in the backtesting framework.
    
    This class defines the interface that all trading strategies must implement
    and provides core functionality for managing data sources, positions, and
    technical indicators. Strategies inherit from this class to implement
    their specific trading logic.
    
    The Strategy class manages:
    - Multiple data sources and their associated broker positions
    - Technical indicator arrays using TimeNDArray
    - Initialization and iteration over market data
    - Abstract methods for strategy-specific logic
    
    Example:
        >>> class MyStrategy(Strategy):  
        ...     def init(self):  
        ...         # Initialize indicators and parameters  
        ...         self.fast_ma = None  
        ...           
        ...     def next(self):  
        ...         # Trading logic for each time step  
        ...         if len(self.data['AAPL'].Close) > 20:  
        ...             fast = np.mean(self.data['AAPL'].Close[-10:])  
        ...             slow = np.mean(self.data['AAPL'].Close[-20:])  
        ...             if fast > slow:  
        ...                 self.positions['AAPL'].buy(quantity=0.1)  
        ...             else:  
        ...                 self.positions['AAPL'].sell(quantity=0.1)  
    """
    def __init__(self):
        """
        Initialize the strategy with empty data structures.
        
        This constructor sets up the core containers for data sources,
        positions, and indicators that the strategy will use.
        """
        self.positions: dict[str, Broker] = {}
        self.data: dict[str, DataSource] = {}
        self.indicators: list[TimeNDArray] = []
        self.ta = builtin_indicators
    
    @abstractmethod
    def init(self):
        """
        Initialize the strategy with indicators and parameters.
        
        This method is called once at the beginning of the backtest
        before any data iteration begins. Strategies should use this
        method to:
        - Initialize technical indicators
        - Set strategy parameters
        - Prepare any pre-computed values
        - Setup any required data structures
        
        Note:
            This method must be implemented by concrete strategy classes.
            It is called automatically by the backtester.
        """
        pass

    @abstractmethod
    def next(self):
        """
        Execute strategy logic for the current time step.
        
        This method is called for each time step in the backtest data
        and should contain the main trading logic of the strategy.
        At the time this method is called, all data up to the current
        time step is available in the data sources and positions.
        
        The method can access:
        - self.data: Dictionary of DataSource objects with current market data
        - self.positions: Dictionary of Broker objects for position management
        - self.indicators: List of TimeNDArray objects for technical analysis
        
        Typical actions include:
        - Reading current market data (e.g., self.data['AAPL'].CClose)
        - Updating technical indicators
        - Making buy/sell decisions based on indicators
        - Managing position sizes
        - Setting stop loss or take profit levels
        
        Note:
            This method must be implemented by concrete strategy classes.
            It is called automatically by the backtester for each time step.
        """
        pass

    @final
    def add_data(self, source: DataSource, symbol: str):
        """
        Add a data source and corresponding broker position to the strategy.
        
        This method associates a data source (containing market data for a
        specific symbol) with a broker position for that same symbol.
        It's typically called during strategy initialization.
        
        Args:
            source (DataSource): DataSource object containing OHLCV data
                for the specified symbol.
            symbol (str): Identifier for the symbol (e.g., 'AAPL', 'EURUSD').
                
        Note:
            This method is final and cannot be overridden by strategy classes.
            Each symbol can only have one data source and broker position.
            
        Example:
            >>> class MyStrategy(Strategy):  
            ...     def init(self):  
            ...         self.add_data(CSVDataSource("AAPL.csv"), "AAPL")  
            ...         self.add_data(CSVDataSource("EURUSD.csv"), "EURUSD")  
        """
        self.data[symbol] = source
        self.positions[symbol] = Broker(source)
    
    @final
    def Indicator(self, arr: np.typing.NDArray):
        """
        Create a time-aware indicator array from input data.
        
        This method converts a numpy array into a TimeNDArray, which
        maintains time awareness for backtesting. The returned array
        can be used to store technical indicators that need to be
        progressively built during the backtest.
        
        Args:
            arr (np.typing.NDArray): Input array containing indicator values.
            
        Returns:
            TimeNDArray: Time-aware array that can be used for indicators.
            
        Note:
            This method is final and cannot be overridden by strategy classes.
            The indicator is automatically added to the strategy's indicators
            list and will be properly managed during backtest iteration.
            
        Example:
            >>> class MyStrategy(Strategy):  
            ...     def init(self):  
            ...         # Create empty indicator array  
            ...         self.sma = self.Indicator(np.zeros(len(self.data['AAPL'])))  
            ...           
            ...     def next(self):  
            ...         # Update indicator  
            ...         if len(self.data['AAPL'].Close) >= 20:  
            ...             self.sma[-1] = np.mean(self.data['AAPL'].Close[-20:])  
        """
        data = TimeNDArray.from_array(arr)
        self.indicators.append(data)
        return data
