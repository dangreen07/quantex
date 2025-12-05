# Data Sources Guide

This guide covers how to work with market data in QuantEx, including loading, managing, and processing financial time series data.

## Overview

QuantEx provides flexible data source classes to handle various types of market data:

- **CSVDataSource**: Load data from CSV files
- **DataSource**: Base class for custom data sources
- **Time-aware data access**: Efficient handling of time series data

## Basic Data Loading

### CSV Data Source

The most common way to load data is using `CSVDataSource`:

```python
from quantex import CSVDataSource

# Load data from CSV file
data_source = CSVDataSource('data/EURUSD_M1.csv')

# Access basic information
print(f"Data length: {len(data_source)}")
print(f"Columns: {list(data_source.data.columns)}")
print(f"Date range: {data_source.Index[0]} to {data_source.Index[-1]}")
```

### CSV File Format Requirements

Your CSV files should follow this format:

```csv
DateTime,Open,High,Low,Close,Volume
2023-01-01 00:00:00,1.23456,1.23478,1.23434,1.23467,1000
2023-01-01 00:01:00,1.23467,1.23489,1.23445,1.23478,1200
...
```

### Parquet Data Source

Another method is to use the `ParquetDataSource`:

```python
from quantex import ParquetDataSource

# Load data from CSV file
data_source = ParquetDataSource('data/EURUSD_M1.csv')

# Access basic information
print(f"Data length: {len(data_source)}")
print(f"Columns: {list(data_source.data.columns)}")
print(f"Date range: {data_source.Index[0]} to {data_source.Index[-1]}")
```

**Requirements:**
- First column must be datetime (index column)
- Required columns: `Open`, `High`, `Low`, `Close`, `Volume`
- Datetime format: `YYYY-MM-DD HH:MM:SS`
- Numeric values should use dot (`.`) as decimal separator

## Data Access Methods

### Current Values (Real-time)

Access current market data using the `C` prefix:

```python
# Current OHLCV values
current_open = data_source.COpen      # Current Open price
current_high = data_source.CHigh      # Current High price
current_low = data_source.CLow        # Current Low price
current_close = data_source.CClose    # Current Close price
current_volume = data_source.CVolume  # Current Volume

# Current timestamp
current_time = data_source.Index[data_source.current_index]
```

### Historical Data Access

Access historical data using array-like indexing:

```python
# Access specific historical bars
first_close = data_source.Close[0]      # First Close price
last_close = data_source.Close[-1]      # Last Close price
recent_closes = data_source.Close[-100:] # Last 100 Close prices

# Access other OHLCV data
opens = data_source.Open
highs = data_source.High
lows = data_source.Low
volumes = data_source.Volume

# Access by index
close_at_index = data_source.Close[1000]  # Close at index 1000
```

### Time-based Access

```python
# Access data by datetime (if you have the datetime index)
# Note: This requires pandas datetime indexing
import pandas as pd

# Get close price at specific time
target_time = pd.Timestamp('2023-01-01 12:00:00')
if target_time in data_source.Index:
    idx = data_source.Index.get_loc(target_time)
    price = data_source.Close[idx]
```

## Working with Multiple Data Sources

### Multiple Symbols

```python
from quantex import Strategy

class MultiSymbolStrategy(Strategy):
    def init(self):
        # Load multiple symbols
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

        for symbol in symbols:
            data = CSVDataSource(f'data/{symbol}_M1.csv')
            self.add_data(data, symbol)

    def next(self):
        # Access data for each symbol
        for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
            current_price = self.data[symbol].CClose
            print(f"{symbol}: {current_price}")
```

### Multiple Timeframes

```python
class MultiTimeframeStrategy(Strategy):
    def init(self):
        # Load same symbol on different timeframes
        self.add_data(CSVDataSource('data/EURUSD_M1.csv'), 'EURUSD_M1')
        self.add_data(CSVDataSource('data/EURUSD_H1.csv'), 'EURUSD_H1')
        self.add_data(CSVDataSource('data/EURUSD_D1.csv'), 'EURUSD_D1')

    def next(self):
        # Use different timeframes for different purposes
        m1_close = self.data['EURUSD_M1'].CClose  # Entry timing
        h1_trend = self.data['EURUSD_H1'].Close[-1]  # Trend direction
        d1_support = self.data['EURUSD_D1'].Low[-20:].min()  # Support level
```

## Data Validation and Quality Checks

### Basic Data Validation

```python
def validate_data_source(data_source):
    """Validate data source integrity"""

    # Check required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns
                      if col not in data_source.data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check for NaN values
    if data_source.data.isnull().any().any():
        print("Warning: NaN values found in data")

    # Check for data gaps
    if hasattr(data_source.Index, 'freq'):
        expected_length = len(data_source.Index)
        actual_length = len(data_source.data)
        if expected_length != actual_length:
            print(f"Warning: Possible gaps in data. Expected: {expected_length}, Got: {actual_length}")

    return True
```

### Data Quality Metrics

```python
def analyze_data_quality(data_source):
    """Analyze data quality metrics"""

    close_prices = data_source.Close

    print(f"Data points: {len(close_prices)}")
    print(f"Date range: {data_source.Index[0]} to {data_source.Index[-1]}")
    print(f"Price range: {close_prices.min():.5f} - {close_prices.max():.5f}")

    # Check for outliers
    mean_price = close_prices.mean()
    std_price = close_prices.std()
    outliers = close_prices[(close_prices - mean_price).abs() > 3 * std_price]

    print(f"Outliers detected: {len(outliers)}")

    # Check for gaps in time series
    if len(data_source.Index) > 1:
        time_diffs = pd.Series(data_source.Index[1:]) - pd.Series(data_source.Index[:-1])
        gaps = time_diffs[time_diffs > time_diffs.median() * 1.5]
        print(f"Time gaps detected: {len(gaps)}")

    return {
        'length': len(close_prices),
        'price_range': (close_prices.min(), close_prices.max()),
        'outliers': len(outliers),
        'gaps': len(gaps)
    }
```

## Custom Data Sources

### Creating Custom Data Sources

```python
import pandas as pd
import numpy as np
from quantex.datasource import DataSource

class CustomDataSource(DataSource):
    def __init__(self, api_data, symbol='CUSTOM'):
        # Process your API data into the required format
        df = self.process_api_data(api_data, symbol)

        # Validate required columns
        super().__init__(df)

    def process_api_data(self, api_data, symbol):
        """Convert API data to QuantEx format"""

        # Example: Convert from API format to OHLCV
        processed_data = []

        for item in api_data:
            processed_data.append({
                'Open': float(item['open']),
                'High': float(item['high']),
                'Low': float(item['low']),
                'Close': float(item['close']),
                'Volume': int(item['volume'])
            })

        # Create DataFrame with datetime index
        df = pd.DataFrame(processed_data)

        # Convert timestamp if needed
        if 'timestamp' in api_data[0]:
            df.index = pd.to_datetime([item['timestamp'] for item in api_data])

        return df
```

### Using Custom Data Sources

```python
# Example with financial API data
api_data = [
    {'timestamp': '2023-01-01T00:00:00Z', 'open': 1.234, 'high': 1.236,
     'low': 1.233, 'close': 1.235, 'volume': 1000},
    # ... more data
]

# Create custom data source
custom_source = CustomDataSource(api_data, symbol='EURUSD')

# Use in strategy
strategy = MyStrategy()
strategy.add_data(custom_source, 'EURUSD')
```

## Data Preprocessing

### Handling Missing Data

```python
def clean_data(data_source):
    """Clean and prepare data for backtesting"""

    df = data_source.data.copy()

    # Forward fill small gaps (less than 5 minutes)
    df = df.fillna(method='ffill', limit=5)

    # Remove remaining NaN values
    df = df.dropna()

    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]

    # Sort by time (important!)
    df = df.sort_index()

    return CSVDataSource.from_dataframe(df)
```

### Data Normalization

```python
def normalize_data(data_source):
    """Normalize price data for certain strategies"""

    df = data_source.data.copy()

    # Log returns for stationary analysis
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Rolling statistics
    df['Rolling_Mean'] = df['Close'].rolling(window=20).mean()
    df['Rolling_Std'] = df['Close'].rolling(window=20).std()

    # Z-score normalization
    df['Z_Score'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']

    return df
```

## Performance Optimization

### Memory-Efficient Data Loading

```python
def load_large_dataset(file_path, chunksize=10000):
    """Load large datasets in chunks"""

    # Read first chunk to get structure
    first_chunk = pd.read_csv(file_path, nrows=chunksize)

    # Process in chunks
    chunk_iter = pd.read_csv(file_path, chunksize=chunksize)

    processed_chunks = []
    for chunk in chunk_iter:
        # Process each chunk
        processed_chunk = process_chunk(chunk)
        processed_chunks.append(processed_chunk)

    # Combine all chunks
    combined_data = pd.concat(processed_chunks, ignore_index=True)
    combined_data.index = pd.to_datetime(combined_data['DateTime'])
    combined_data = combined_data.drop('DateTime', axis=1)

    return CSVDataSource(combined_data)
```

### Data Sampling for Faster Backtesting

```python
def create_sampled_data(original_data, sample_rate='1H'):
    """Create lower frequency data for faster testing"""

    df = original_data.data.resample(sample_rate).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    return CSVDataSource(df)
```

## Integration with pandas

### Converting to pandas DataFrame

```python
# Convert QuantEx data to pandas for analysis
def data_to_pandas(data_source):
    """Convert QuantEx DataSource to pandas DataFrame"""

    data_dict = {
        'Open': data_source.Open,
        'High': data_source.High,
        'Low': data_source.Low,
        'Close': data_source.Close,
        'Volume': data_source.Volume
    }

    return pd.DataFrame(data_dict, index=data_source.Index[:len(data_source.Close)])
```

### Using pandas Operations

```python
# Advanced pandas operations on QuantEx data
def analyze_with_pandas(data_source):
    """Perform complex analysis using pandas"""

    df = data_to_pandas(data_source)

    # Calculate technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Calculate returns
    df['Returns'] = df['Close'].pct_change()

    # Calculate volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()

    # Find patterns
    df['Above_SMA20'] = df['Close'] > df['SMA_20']

    return df
```

## Best Practices

### 1. Data Organization

```
data/
├── EURUSD/
│   ├── EURUSD_M1.csv    # 1-minute data
│   ├── EURUSD_H1.csv    # 1-hour data
│   └── EURUSD_D1.csv    # Daily data
├── GBPUSD/
│   └── GBPUSD_M1.csv
└── indicators/
    └── custom_indicators.csv
```

### 2. Data Validation Pipeline

```python
def create_data_pipeline(file_paths):
    """Create a robust data loading pipeline"""

    validated_sources = []

    for path in file_paths:
        try:
            # Load data
            source = CSVDataSource(path)

            # Validate
            if validate_data_source(source):

            # Clean
            cleaned_source = clean_data(source)

            # Quality check
            quality = analyze_data_quality(cleaned_source)

            if quality['outliers'] < 100 and quality['gaps'] < 10:
                validated_sources.append(cleaned_source)
            else:
                print(f"Skipping {path}: Poor data quality")

        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    return validated_sources
```

### 3. Error Handling

```python
def safe_data_access(data_source, index):
    """Safely access data with bounds checking"""

    try:
        if index >= len(data_source):
            raise IndexError("Index out of bounds")

        return {
            'open': data_source.Open[index],
            'high': data_source.High[index],
            'low': data_source.Low[index],
            'close': data_source.Close[index],
            'volume': data_source.Volume[index]
        }

    except (IndexError, KeyError) as e:
        print(f"Data access error: {e}")
        return None
```

## Common Patterns

### Rolling Window Calculations

```python
def calculate_rolling_metrics(data_source, window=20):
    """Calculate rolling window metrics"""

    closes = data_source.Close

    # Rolling mean
    rolling_mean = pd.Series(closes).rolling(window=window).mean().values

    # Rolling standard deviation
    rolling_std = pd.Series(closes).rolling(window=window).std().values

    # Bollinger Bands
    upper_band = (rolling_mean + 2 * rolling_std)
    lower_band = (rolling_mean - 2 * rolling_std)

    return {
        'mean': rolling_mean,
        'std': rolling_std,
        'upper_bb': upper_band,
        'lower_bb': lower_band
    }
```

### Data Synchronization

```python
def synchronize_data_sources(*data_sources):
    """Synchronize multiple data sources to common timeline"""

    # Find common time period
    all_timestamps = set()
    for source in data_sources:
        all_timestamps.update(source.Index)

    common_timestamps = sorted(all_timestamps)

    # Resample each source to common timeline
    synchronized_sources = []

    for source in data_sources:
        # Resample to common timestamps
        synchronized_df = source.data.reindex(common_timestamps).fillna(method='ffill')
        synchronized_sources.append(CSVDataSource(synchronized_df))

    return synchronized_sources
```

## Next Steps

Now that you understand data management in QuantEx, explore these related topics:

- **[Strategy Guide](./strategy.md)**: Learn how to use data in trading strategies
- **[Backtesting Guide](./backtesting.md)**: Understand how data flows through the backtesting engine
- **[Technical Indicators Guide](./indicators.md)**: Work with indicators and derived data

For complete API reference, see the [Data Sources API documentation](../../reference/quantex.sources.md).