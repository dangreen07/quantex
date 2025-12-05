# Data Sources API Reference

This page provides comprehensive API documentation for QuantEx's data source system.

## DataSource Class

The `DataSource` class is the base class for all market data sources in QuantEx.

```python
class DataSource:
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    def __init__(self, df: pd.DataFrame):
        self.data = df
        if not all(col in self.data.columns for col in self.required_columns):
            raise ValueError(f"Dataframe requires the following columns: {self.required_columns}")
        self.current_index = len(self.data)
        self.open_data = np.ascontiguousarray(self.data['Open'].to_numpy(), dtype=np.float64)
        self.high_data = np.ascontiguousarray(self.data['High'].to_numpy(), dtype=np.float64)
        self.low_data = np.ascontiguousarray(self.data['Low'].to_numpy(), dtype=np.float64)
        self.close_data = np.ascontiguousarray(self.data['Close'].to_numpy(), dtype=np.float64)
        self.volume_data = np.ascontiguousarray(self.data['Volume'].to_numpy(), dtype=np.float64)
```

### Constructor Parameters

#### `df: pd.DataFrame`
Pandas DataFrame containing OHLCV data with datetime index.

### Class Attributes

#### `required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']`
Required column names for valid data source.

### Instance Attributes

#### `data: pd.DataFrame`
Original pandas DataFrame with market data.

#### `current_index: int`
Current position in the data (used during backtesting).

#### `open_data, high_data, low_data, close_data, volume_data: np.ndarray`
NumPy arrays containing price/volume data for fast access.

### Properties

#### `Index`

Get the DataFrame index (datetime index).

```python
@property
def Index(self):
    """Get datetime index"""
    return self.data.index
```

**Usage:**
```python
# Access datetime index
for timestamp in data_source.Index:
    print(f"Data point at: {timestamp}")
```

#### `Open, High, Low, Close, Volume`

Get historical price/volume data up to current index.

```python
@property
def Open(self):
    """Get historical Open prices"""
    return self.open_data[:self.current_index]

@property
def Close(self):
    """Get historical Close prices"""
    return self.close_data[:self.current_index]
```

**Usage:**
```python
# Access historical data
open_prices = data_source.Open      # All Open prices up to current_index
close_prices = data_source.Close    # All Close prices up to current_index
recent_highs = data_source.High[-100:]  # Last 100 High prices
```

#### `COpen, CHigh, CLow, CClose, CVolume`

Get current (most recent) price/volume values.

```python
@property
def COpen(self) -> np.float64:
    """Get current Open price"""
    return self.open_data[self.current_index]

@property
def CClose(self) -> np.float64:
    """Get current Close price"""
    return self.close_data[self.current_index]
```

**Usage:**
```python
# Access current market data
current_price = data_source.CClose
current_high = data_source.CHigh
current_volume = data_source.CVolume
```

### Instance Methods

#### `__len__(self) -> int`

Return the total length of the data.

```python
def __len__(self):
    """Return total number of data points"""
    return len(self.data)
```

**Usage:**
```python
total_bars = len(data_source)
print(f"Total data points: {total_bars}")
```

## CSVDataSource Class

Specialized data source for loading data from CSV files.

```python
class CSVDataSource(DataSource):
    def __init__(self, pathname: str, train_test_split: bool = False, mode: str = "train"):
        data = pd.read_csv(pathname, index_col=0, parse_dates=[0])
        super().__init__(data, train_test_split, mode)
```

## ParquetDataSource Class

Specialized data source for loading data from Parquet files.

```python
class ParquetDataSource(DataSource):
    def __init__(self, pathname: str, train_test_split: bool = False, mode: str = "train"):
        data = pd.read_parquet(pathname, index_col=0, parse_dates=[0])
        super().__init__(data, train_test_split, mode)
```

### Constructor Parameters

#### `pathname: str`
Path to CSV file containing market data.

### Usage

```python
# Load data from CSV file
data_source = CSVDataSource('data/EURUSD_M1.csv')

# Access data
print(f"Data length: {len(data_source)}")
print(f"Date range: {data_source.Index[0]} to {data_source.Index[-1]}")
print(f"Current price: {data_source.CClose}")
```

## Complete Data Source Example

```python
import pandas as pd
import numpy as np
from quantex import DataSource, CSVDataSource, Strategy

class DataAnalysisStrategy(Strategy):
    """Strategy demonstrating data source usage"""

    def init(self):
        # Load data using CSV data source
        self.data_source = CSVDataSource('data/EURUSD.csv')
        self.add_data(self.data_source, 'EURUSD')

        # Alternative: Create data source from DataFrame
        # custom_df = self.create_custom_dataframe()
        # self.custom_source = DataSource(custom_df)
        # self.add_data(self.custom_source, 'CUSTOM')

    def next(self):
        # Access current market data
        current_price = self.data['EURUSD'].CClose
        current_high = self.data['EURUSD'].CHigh
        current_low = self.data['EURUSD'].CLow
        current_volume = self.data['EURUSD'].CVolume

        # Access historical data
        historical_prices = self.data['EURUSD'].Close
        recent_prices = self.data['EURUSD'].Close[-20:]  # Last 20 bars

        # Calculate indicators using historical data
        if len(historical_prices) > 50:
            sma_50 = self.calculate_sma(historical_prices, 50)
            current_sma = sma_50[-1] if len(sma_50) > 0 else current_price

            # Trading logic
            if current_price > current_sma * 1.02:  # 2% above SMA
                self.positions['EURUSD'].buy(0.3)
            elif current_price < current_sma * 0.98:  # 2% below SMA
                self.positions['EURUSD'].sell(0.3)

    def calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        return pd.Series(prices).rolling(window=period).mean().values

# Usage
strategy = DataAnalysisStrategy()
backtester = SimpleBacktester(strategy, cash=10000)
report = backtester.run()
```

## Advanced Data Source Techniques

### Custom Data Source Implementation

```python
class CustomAPI_DataSource(DataSource):
    """Custom data source for API data"""

    def __init__(self, api_data, symbol='CUSTOM'):
        """
        Initialize from API data

        Args:
            api_data: List of dictionaries with OHLCV data
            symbol: Symbol name
        """

        # Convert API data to DataFrame
        df_data = []

        for item in api_data:
            df_data.append({
                'Open': float(item['open']),
                'High': float(item['high']),
                'Low': float(item['low']),
                'Close': float(item['close']),
                'Volume': int(item['volume'])
            })

        # Create DataFrame
        df = pd.DataFrame(df_data)

        # Set datetime index
        if 'timestamp' in api_data[0]:
            df.index = pd.to_datetime([item['timestamp'] for item in api_data])

        # Initialize parent class
        super().__init__(df)

# Usage with API data
api_data = [
    {'timestamp': '2023-01-01T00:00:00Z', 'open': 1.234, 'high': 1.236,
     'low': 1.233, 'close': 1.235, 'volume': 1000},
    {'timestamp': '2023-01-01T00:01:00Z', 'open': 1.235, 'high': 1.237,
     'low': 1.234, 'close': 1.236, 'volume': 1100},
    # ... more data
]

custom_source = CustomAPI_DataSource(api_data, 'EURUSD')
```

### Multi-Symbol Data Management

```python
class MultiSymbolDataManager:
    """Manage multiple data sources efficiently"""

    def __init__(self):
        self.data_sources = {}
        self.symbols = []

    def add_symbol(self, symbol, data_source):
        """Add a symbol and its data source"""

        if not isinstance(data_source, DataSource):
            raise ValueError("data_source must be a DataSource instance")

        self.data_sources[symbol] = data_source
        self.symbols.append(symbol)

    def get_current_prices(self):
        """Get current prices for all symbols"""

        current_prices = {}

        for symbol in self.symbols:
            try:
                current_prices[symbol] = self.data_sources[symbol].CClose
            except (AttributeError, IndexError):
                print(f"Warning: Could not get current price for {symbol}")
                current_prices[symbol] = None

        return current_prices

    def get_price_history(self, symbol, lookback=100):
        """Get price history for a symbol"""

        if symbol not in self.data_sources:
            raise ValueError(f"Symbol {symbol} not found")

        data_source = self.data_sources[symbol]

        if len(data_source.Close) < lookback:
            return data_source.Close
        else:
            return data_source.Close[-lookback:]

    def synchronize_data_sources(self):
        """Synchronize all data sources to common timeline"""

        if not self.data_sources:
            return

        # Find common time period
        all_timestamps = set()

        for symbol, source in self.data_sources.items():
            all_timestamps.update(source.Index)

        common_timestamps = sorted(all_timestamps)

        # Resample each data source
        synchronized_sources = {}

        for symbol, source in self.data_sources.items():
            # Reindex to common timestamps
            synchronized_df = source.data.reindex(common_timestamps).fillna(method='ffill')
            synchronized_sources[symbol] = DataSource(synchronized_df)

        self.data_sources = synchronized_sources
```

### Data Quality Validation

```python
class DataQualityValidator:
    """Validate data source quality"""

    @staticmethod
    def validate_data_source(data_source):
        """Comprehensive data validation"""

        issues = []

        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns
                          if col not in data_source.data.columns]

        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")

        # Check for NaN values
        nan_counts = data_source.data.isnull().sum()
        columns_with_nan = nan_counts[nan_counts > 0]

        if not columns_with_nan.empty:
            for col, count in columns_with_nan.items():
                issues.append(f"Column {col} has {count} NaN values")

        # Check for data gaps
        if hasattr(data_source.Index, 'freq'):
            expected_points = len(data_source.Index)
            actual_points = len(data_source.data.dropna())

            if actual_points != expected_points:
                issues.append(f"Data gaps detected: {expected_points - actual_points} missing points")

        # Check price consistency
        price_issues = DataQualityValidator.check_price_consistency(data_source)
        issues.extend(price_issues)

        # Check volume consistency
        volume_issues = DataQualityValidator.check_volume_consistency(data_source)
        issues.extend(volume_issues)

        return issues

    @staticmethod
    def check_price_consistency(data_source):
        """Check for price consistency issues"""

        issues = []

        # Check for negative prices
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data_source.data.columns:
                negative_prices = (data_source.data[col] <= 0).sum()
                if negative_prices > 0:
                    issues.append(f"Column {col} has {negative_prices} negative values")

        # Check High >= Low
        if 'High' in data_source.data.columns and 'Low' in data_source.data.columns:
            invalid_hl = (data_source.data['High'] < data_source.data['Low']).sum()
            if invalid_hl > 0:
                issues.append(f"High < Low in {invalid_hl} instances")

        # Check OHLC relationships
        if all(col in data_source.data.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_ohlc = (
                (data_source.data['High'] < data_source.data['Open']) |
                (data_source.data['High'] < data_source.data['Close']) |
                (data_source.data['Low'] > data_source.data['Open']) |
                (data_source.data['Low'] > data_source.data['Close'])
            ).sum()

            if invalid_ohlc > 0:
                issues.append(f"OHLC relationship violations: {invalid_ohlc}")

        return issues

    @staticmethod
    def check_volume_consistency(data_source):
        """Check for volume consistency issues"""

        issues = []

        if 'Volume' in data_source.data.columns:
            negative_volumes = (data_source.data['Volume'] < 0).sum()
            if negative_volumes > 0:
                issues.append(f"Negative volume in {negative_volumes} instances")

            zero_volumes = (data_source.data['Volume'] == 0).sum()
            if zero_volumes > len(data_source.data) * 0.1:  # More than 10% zero volume
                issues.append(f"High zero volume: {zero_volumes} instances")

        return issues
```

### Data Preprocessing

```python
class DataPreprocessor:
    """Preprocess and clean market data"""

    @staticmethod
    def clean_data(data_source, fill_gaps=True, remove_outliers=True):
        """Clean and preprocess data source"""

        df = data_source.data.copy()

        # Fill small gaps
        if fill_gaps:
            df = DataPreprocessor.fill_small_gaps(df)

        # Remove outliers
        if remove_outliers:
            df = DataPreprocessor.remove_price_outliers(df)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Sort by time
        df = df.sort_index()

        return DataSource(df)

    @staticmethod
    def fill_small_gaps(df, max_gap_minutes=5):
        """Fill small gaps in time series"""

        df = df.copy()

        # Forward fill small gaps
        df = df.fillna(method='ffill')

        # If still have gaps, consider interpolation for very small gaps
        if df.isnull().any().any():
            # Simple interpolation for remaining gaps
            df = df.interpolate(method='linear', limit=2)

        # Drop any remaining NaN values
        df = df.dropna()

        return df

    @staticmethod
    def remove_price_outliers(df, z_threshold=3.0):
        """Remove price outliers using z-score method"""

        df = df.copy()

        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                # Calculate rolling z-score
                rolling_mean = df[col].rolling(window=50).mean()
                rolling_std = df[col].rolling(window=50).std()

                z_scores = np.abs((df[col] - rolling_mean) / rolling_std)

                # Mark outliers
                outliers = z_scores > z_threshold

                # Replace outliers with rolling median
                df.loc[outliers, col] = df[col].rolling(window=10, center=True).median()

        return df
```

### Data Transformation

```python
class DataTransformer:
    """Transform and enrich market data"""

    @staticmethod
    def add_technical_indicators(data_source):
        """Add technical indicators to data source"""

        df = data_source.data.copy()

        # Add common indicators
        df['SMA_20'] = pd.Series(data_source.Close).rolling(window=20).mean()
        df['SMA_50'] = pd.Series(data_source.Close).rolling(window=50).mean()

        # RSI
        df['RSI_14'] = DataTransformer.calculate_rsi(data_source.Close, 14)

        # Bollinger Bands
        sma_20 = pd.Series(data_source.Close).rolling(window=20).mean()
        std_20 = pd.Series(data_source.Close).rolling(window=20).std()

        df['BB_Upper'] = sma_20 + 2 * std_20
        df['BB_Lower'] = sma_20 - 2 * std_20

        # Volume indicators
        df['Volume_SMA'] = pd.Series(data_source.Volume).rolling(window=20).mean()

        return DataSource(df)

    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator"""

        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50).values

    @staticmethod
    def resample_data(data_source, frequency):
        """Resample data to different frequency"""

        df = data_source.data.resample(frequency).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        return DataSource(df)
```

## Performance Optimization

### Memory-Efficient Data Loading

```python
class MemoryEfficientLoader:
    """Load large datasets efficiently"""

    @staticmethod
    def load_large_csv(file_path, chunk_size=10000):
        """Load large CSV files in chunks"""

        # Read first chunk to get structure
        first_chunk = pd.read_csv(file_path, nrows=chunk_size)

        # Process in chunks
        chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)

        processed_chunks = []

        for chunk in chunk_iter:
            # Process chunk (validate, clean, etc.)
            processed_chunk = MemoryEfficientLoader.process_chunk(chunk)
            processed_chunks.append(processed_chunk)

        # Combine chunks
        combined_df = pd.concat(processed_chunks, ignore_index=True)

        # Set datetime index
        if 'DateTime' in combined_df.columns:
            combined_df.index = pd.to_datetime(combined_df['DateTime'])
            combined_df = combined_df.drop('DateTime', axis=1)

        return DataSource(combined_df)

    @staticmethod
    def process_chunk(chunk):
        """Process individual chunk"""

        # Validate chunk
        if not all(col in chunk.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("Chunk missing required columns")

        # Clean chunk
        chunk = chunk.dropna()  # Remove rows with NaN

        return chunk
```

### Fast Data Access

```python
class FastDataAccessor:
    """Optimized data access patterns"""

    def __init__(self, data_source):
        self.data_source = data_source
        self._cache = {}

    def get_recent_data(self, symbol, bars=100):
        """Get recent data with caching"""

        cache_key = f"{symbol}_{bars}"

        if cache_key not in self._cache:
            # Cache recent data for fast access
            self._cache[cache_key] = {
                'close': self.data_source.Close[-bars:],
                'high': self.data_source.High[-bars:],
                'low': self.data_source.Low[-bars:],
                'volume': self.data_source.Volume[-bars:]
            }

        return self._cache[cache_key]

    def get_price_at_time(self, target_time):
        """Get price at specific time"""

        try:
            # Find index for timestamp
            time_index = self.data_source.Index.get_loc(target_time)
            return self.data_source.Close[time_index]
        except KeyError:
            return None
```

## Error Handling

### Data Source Errors

```python
class DataSourceError(Exception):
    """Base exception for data source errors"""
    pass

class InvalidDataError(DataSourceError):
    """Raised when data format is invalid"""
    pass

class MissingDataError(DataSourceError):
    """Raised when required data is missing"""
    pass

class DataAccessError(DataSourceError):
    """Raised when data access fails"""
    pass

# Safe data source creation
def safe_create_data_source(file_path):
    """Safely create data source with error handling"""

    try:
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Create data source
        data_source = CSVDataSource(file_path)

        # Validate data quality
        issues = DataQualityValidator.validate_data_source(data_source)

        if issues:
            print(f"Data quality issues found: {issues}")
            # Decide whether to continue or raise error

        return data_source

    except FileNotFoundError as e:
        raise MissingDataError(f"Data file missing: {e}")

    except pd.errors.EmptyDataError:
        raise InvalidDataError("Data file is empty")

    except pd.errors.ParserError as e:
        raise InvalidDataError(f"Data file parsing error: {e}")

    except Exception as e:
        raise DataSourceError(f"Unexpected error creating data source: {e}")
```

## Best Practices

### 1. Data Organization

```python
# Recommended data directory structure
data/
├── raw/                    # Raw downloaded data
│   ├── EURUSD_M1.csv
│   ├── EURUSD_H1.csv
│   └── ...
├── processed/              # Cleaned and validated data
│   ├── EURUSD_M1_clean.csv
│   └── ...
├── indicators/             # Pre-calculated indicators
│   └── ...
└── metadata/               # Data information
    ├── EURUSD_info.json
    └── ...
```

### 2. Data Pipeline

```python
class DataPipeline:
    """Complete data processing pipeline"""

    def __init__(self, raw_data_path, processed_data_path):
        self.raw_path = raw_data_path
        self.processed_path = processed_data_path

    def process_symbol_data(self, symbol):
        """Process data for a single symbol"""

        print(f"Processing {symbol}...")

        # 1. Load raw data
        raw_file = os.path.join(self.raw_path, f"{symbol}.csv")
        data_source = safe_create_data_source(raw_file)

        # 2. Validate data
        issues = DataQualityValidator.validate_data_source(data_source)
        if issues:
            print(f"Data issues for {symbol}: {issues}")

        # 3. Clean data
        cleaned_source = DataPreprocessor.clean_data(data_source)

        # 4. Add indicators
        enriched_source = DataTransformer.add_technical_indicators(cleaned_source)

        # 5. Save processed data
        processed_file = os.path.join(self.processed_path, f"{symbol}_processed.csv")
        enriched_source.data.to_csv(processed_file)

        print(f"Saved processed data to {processed_file}")

        return enriched_source
```

### 3. Memory Management

```python
def manage_data_memory(data_sources, max_sources=10):
    """Manage memory usage for multiple data sources"""

    if len(data_sources) > max_sources:
        # Remove oldest data sources
        symbols_to_remove = list(data_sources.keys())[:-max_sources]

        for symbol in symbols_to_remove:
            del data_sources[symbol]
            print(f"Removed {symbol} from memory")

    # Force garbage collection
    import gc
    gc.collect()
```

## Integration Examples

### Using with pandas

```python
def data_source_to_pandas(data_source):
    """Convert DataSource to pandas DataFrame"""

    data_dict = {
        'Open': data_source.Open,
        'High': data_source.High,
        'Low': data_source.Close,
        'Close': data_source.Close,
        'Volume': data_source.Volume
    }

    return pd.DataFrame(data_dict, index=data_source.Index[:len(data_source.Close)])

# Usage
df = data_source_to_pandas(data_source)
sma_20 = df['Close'].rolling(window=20).mean()
```

### Batch Processing

```python
def process_multiple_symbols(symbols, data_directory):
    """Process multiple symbols efficiently"""

    processed_sources = {}

    for symbol in symbols:
        try:
            file_path = os.path.join(data_directory, f"{symbol}.csv")
            data_source = CSVDataSource(file_path)

            # Process data
            processed_source = DataPreprocessor.clean_data(data_source)
            processed_sources[symbol] = processed_source

            print(f"Processed {symbol}: {len(data_source)} bars")

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    return processed_sources
```

## Related Documentation

- **[Usage Guide](../../../usage/data-sources.md)**: Learn how to use data sources
- **[Strategy API](../../quantex.strategy.md)**: Understand strategy integration
- **[Backtester API](../../quantex.backtest.md)**: See how data flows through backtesting
- **[Technical Indicators Guide](../../../usage/indicators.md)**: Work with indicators and derived data