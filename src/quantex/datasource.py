import math
from typing import final
import numpy as np
import pandas as pd

class DataSource:
    """
    Base class for handling market data sources in backtesting.
    
    This class provides the interface for loading and accessing OHLCV
    (Open, High, Low, Close, Volume) market data. It supports time series
    data with pandas DataFrame input and provides efficient array-based
    access to price data during backtesting.
    
    The class handles:
    - Data validation (ensuring required columns are present)
    - Optional train/test splitting for walk-forward analysis
    - Efficient numpy array conversion for fast access
    - Current data point tracking for time series iteration
    
    Attributes:
        required_columns (list): List of required column names ['Open', 'High', 'Low', 'Close', 'Volume'].
        
    Example:
        >>> import pandas as pd  
        >>> df = pd.read_csv('data.csv')  
        >>> source = DataSource(df)  
        >>> print(f"Data points: {len(source)}")  
        >>> print(f"Current close: {source.CClose}")  
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    def __init__(self, df: pd.DataFrame, train_test_split: bool = False, mode: str = "train"):
        """
        Initialize the data source with a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data with columns
                'Open', 'High', 'Low', 'Close', 'Volume'. Index should be
                datetime values for proper period calculation.
            train_test_split (bool, optional): Whether to split data into
                train and test sets. Defaults to False.
            mode (str, optional): Mode for data splitting ('train' or 'test').
                Only used when train_test_split=True. Defaults to "train".
                
        Raises:
            ValueError: If DataFrame doesn't contain required columns.
            
        Note:
            - Data is split 80/20 for train/test when enabled.
            - All data is converted to numpy arrays for efficient access.
            - Current index starts at the end for proper iteration.
        """
        self.data = df
        if (train_test_split and mode == "train"):
            index = math.floor(len(df.index) * 0.8)
            self.data = self.data.iloc[:index]
        elif (train_test_split and mode == "test"):
            index = math.floor(len(df.index) * 0.8)
            self.data = self.data.iloc[index:]
        if not all(col in self.data.columns for col in self.required_columns):
            raise ValueError(f"Dataframe requires the following columns: {self.required_columns}")
        self.current_index = len(self.data)
        self.open_data = np.ascontiguousarray(self.data['Open'].to_numpy(), dtype=np.float64)
        self.high_data = np.ascontiguousarray(self.data['High'].to_numpy(), dtype=np.float64)
        self.low_data = np.ascontiguousarray(self.data['Low'].to_numpy(), dtype=np.float64)
        self.close_data = np.ascontiguousarray(self.data['Close'].to_numpy(), dtype=np.float64)
        self.volume_data = np.ascontiguousarray(self.data['Volume'].to_numpy(), dtype=np.float64)

    @final
    def __len__(self):
        """
        Get the total number of data points in the source.
        
        Returns:
            int: Number of rows in the DataFrame.
        """
        return len(self.data)
    
    @property
    def Index(self):
        """
        Get the datetime index of the data source.
        
        Returns:
            pd.Index: Index containing the timestamps for each data point.
        """
        return self.data.index
    
    @property
    def Open(self):
        """
        Get the historical open prices up to current index.
        
        Returns:
            np.ndarray: Array of open prices for all historical data points
                up to the current iteration index.
        """
        return self.open_data[:self.current_index+1]
    
    @property
    def High(self):
        """
        Get the historical high prices up to current index.
        
        Returns:
            np.ndarray: Array of high prices for all historical data points
                up to the current iteration index.
        """
        return self.high_data[:self.current_index+1]
    
    @property
    def Low(self):
        """
        Get the historical low prices up to current index.
        
        Returns:
            np.ndarray: Array of low prices for all historical data points
                up to the current iteration index.
        """
        return self.low_data[:self.current_index+1]
    
    @property
    def Close(self):
        """
        Get the historical close prices up to current index.
        
        Returns:
            np.ndarray: Array of close prices for all historical data points
                up to the current iteration index.
        """
        return self.close_data[:self.current_index+1]
    
    @property
    def Volume(self):
        """
        Get the historical volume data up to current index.
        
        Returns:
            np.ndarray: Array of volume values for all historical data points
                up to the current iteration index.
        """
        return self.volume_data[:self.current_index+1]
    
    @property
    def COpen(self) -> np.float64:
        """
        Get the current open price at the current index.
        
        Returns:
            np.float64: Open price for the current data point.
        """
        return self.open_data[self.current_index]
    
    @property
    def CHigh(self) -> np.float64:
        """
        Get the current high price at the current index.
        
        Returns:
            np.float64: High price for the current data point.
        """
        return self.high_data[self.current_index]
    
    @property
    def CLow(self) -> np.float64:
        """
        Get the current low price at the current index.
        
        Returns:
            np.float64: Low price for the current data point.
        """
        return self.low_data[self.current_index]
    
    @property
    def CClose(self) -> np.float64:
        """
        Get the current close price at the current index.
        
        Returns:
            np.float64: Close price for the current data point.
        """
        return self.close_data[self.current_index]
    
    @property
    def CVolume(self) -> np.float64:
        """
        Get the current volume at the current index.
        
        Returns:
            np.float64: Volume for the current data point.
        """
        return self.volume_data[self.current_index]

class CSVDataSource(DataSource):
    """
    Data source for loading CSV files into the backtesting framework.
    
    This class extends DataSource to specifically handle CSV file loading,
    providing convenient initialization from CSV files with automatic
    parsing and validation.
    
    Args:
        pathname (str): Path to the CSV file to load.
        train_test_split (bool, optional): Whether to split data into
            train and test sets. Defaults to False.
        mode (str, optional): Mode for data splitting ('train' or 'test').
            Only used when train_test_split=True. Defaults to "train".
        index_col (str, optional): Name of the column to use as index.
            Defaults to '0' (first column).
            
    Example:
        >>> source = CSVDataSource("data/AAPL.csv")  
        >>> print(f"Loaded {len(source)} data points")  
    """
    def __init__(self, pathname: str, train_test_split: bool = False, mode: str = "train", index_col: str | int = 0):
        """
        Initialize CSV data source from file path.
        
        Args:
            pathname (str): Path to the CSV file.
            train_test_split (bool, optional): Whether to split data.
            mode (str, optional): Split mode ('train' or 'test').
            index_col (str, optional): Column to use as datetime index.
        """
        data = pd.read_csv(pathname, index_col=index_col, parse_dates=[index_col]) # type: ignore
        super().__init__(data, train_test_split, mode)

class ParquetDataSource(DataSource):
    """
    Data source for loading Parquet files into the backtesting framework.
    
    This class extends DataSource to specifically handle Parquet file loading,
    providing efficient binary format loading with automatic parsing and validation.
    
    Args:
        pathname (str): Path to the Parquet file to load.
        train_test_split (bool, optional): Whether to split data into
            train and test sets. Defaults to False.
        mode (str, optional): Mode for data splitting ('train' or 'test').
            Only used when train_test_split=True. Defaults to "train".
            
    Example:
        >>> source = ParquetDataSource("data/market_data.parquet")  
        >>> print(f"Loaded {len(source)} data points")  
        
    Note:
        Parquet files are generally faster to load than CSV files and
        preserve data types more accurately.
    """
    def __init__(self, pathname: str, train_test_split: bool = False, mode: str = "train"):
        """
        Initialize Parquet data source from file path.
        
        Args:
            pathname (str): Path to the Parquet file.
            train_test_split (bool, optional): Whether to split data.
            mode (str, optional): Split mode ('train' or 'test').
        """
        data = pd.read_parquet(pathname)
        super().__init__(data, train_test_split, mode)