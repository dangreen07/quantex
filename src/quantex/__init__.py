"""
QuantEx: A comprehensive backtesting framework for quantitative trading strategies.

This package provides a complete framework for developing, testing, and optimizing
trading strategies using historical market data. It includes tools for data
management, strategy development, backtesting, and performance optimization.

Key Components:
- Data Sources: Load and manage OHLCV market data from various formats
- Strategy Framework: Abstract base class for implementing trading strategies
- Broker Simulation: Realistic order execution and position management
- Backtesting Engine: Run historical simulations with performance tracking
- Optimization Tools: Grid search and parallel optimization for strategy parameters
- Time-Aware Arrays: Specialized numpy arrays for progressive data access

Features:
- Support for multiple data sources (CSV, Parquet)
- Realistic order execution (market, limit, stop-loss, take-profit)
- Commission modeling (percentage or fixed amount)
- Position management with margin calls
- Performance metrics (Sharpe ratio, max drawdown, total return)
- Parallel parameter optimization
- Time-series data validation and handling
- Walk-forward analysis with train/test splits

Example Usage:
    >>> from quantex import CSVDataSource, Strategy, SimpleBacktester
    >>> 
    >>> class MovingAverageStrategy(Strategy):
    ...     def init(self):
    ...         self.add_data(CSVDataSource("AAPL.csv"), "AAPL")
    ...         self.fast_ma = self.Indicator(np.zeros(len(self.data["AAPL"])))
    ...         self.slow_ma = self.Indicator(np.zeros(len(self.data["AAPL"])))
    ...     
    ...     def next(self):
    ...         if len(self.data["AAPL"].Close) >= 20:
    ...             self.fast_ma[-1] = np.mean(self.data["AAPL"].Close[-10:])
    ...             self.slow_ma[-1] = np.mean(self.data["AAPL"].Close[-20:])
    ...             
    ...             if self.fast_ma[-1] > self.slow_ma[-1] and not self.positions["AAPL"].is_long():
    ...                 self.positions["AAPL"].buy(quantity=0.1)
    ...             elif self.fast_ma[-1] < self.slow_ma[-1] and not self.positions["AAPL"].is_short():
    ...                 self.positions["AAPL"].sell(quantity=0.1)
    >>> 
    >>> # Run backtest
    >>> strategy = MovingAverageStrategy()
    >>> backtester = SimpleBacktester(strategy)
    >>> report = backtester.run()
    >>> print(report)

Classes:
    - Strategy: Abstract base class for trading strategies
    - SimpleBacktester: Main backtesting engine with optimization capabilities
    - CSVDataSource: Load market data from CSV files
    - ParquetDataSource: Load market data from Parquet files
    - CommissionType: Enumeration for commission calculation types
    
Data Requirements:
    All data sources must contain OHLCV columns:
    - 'Open': Opening prices
    - 'High': High prices  
    - 'Low': Low prices
    - 'Close': Closing prices
    - 'Volume': Trading volume
    
    Index should be datetime values for proper period calculations.

Performance Metrics:
    - Total Return: Overall strategy performance
    - Sharpe Ratio: Risk-adjusted returns with confidence intervals
    - Maximum Drawdown: Largest peak-to-trough decline
    - Number of Trades: Total executed trades
    - Commission Impact: Total trading costs

Author: QuantEx Development Team
License: See LICENSE file for details
"""

from .datasource import CSVDataSource as CSVDataSource, ParquetDataSource as ParquetDataSource
from .strategy import Strategy as Strategy
from .backtester import SimpleBacktester as SimpleBacktester
from .enums import CommissionType as CommissionType