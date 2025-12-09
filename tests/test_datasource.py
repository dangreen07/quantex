import pytest
import pandas as pd
import numpy as np
from quantex.datasource import DataSource, CSVDataSource


class TestDataSource:
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        return data

    def test_init_valid_data(self, sample_data):
        """Test DataSource initialization with valid data."""
        ds = DataSource(sample_data)
        assert len(ds) == 100
        assert ds.current_index == 100
        assert isinstance(ds.Open, np.ndarray)
        assert isinstance(ds.Close, np.ndarray)

    def test_init_missing_columns(self):
        """Test DataSource initialization with missing required columns."""
        data = pd.DataFrame({'Open': [1, 2], 'Close': [1, 2]})
        with pytest.raises(ValueError, match="Dataframe requires the following columns"):
            DataSource(data)

    def test_train_test_split_train(self, sample_data):
        """Test train/test split in train mode."""
        ds = DataSource(sample_data, train_test_split=True, mode="train")
        expected_len = int(len(sample_data) * 0.8)
        assert len(ds) == expected_len

    def test_train_test_split_test(self, sample_data):
        """Test train/test split in test mode."""
        ds = DataSource(sample_data, train_test_split=True, mode="test")
        expected_len = len(sample_data) - int(len(sample_data) * 0.8)
        assert len(ds) == expected_len

    def test_properties(self, sample_data):
        """Test DataSource properties return correct data."""
        ds = DataSource(sample_data)
        ds.current_index = 50

        assert len(ds.Open) == 51
        assert len(ds.High) == 51
        assert len(ds.Low) == 51
        assert len(ds.Close) == 51
        assert len(ds.Volume) == 51

        assert isinstance(ds.COpen, np.float64)
        assert isinstance(ds.CClose, np.float64)
        assert ds.COpen == sample_data.iloc[50]['Open']
        assert ds.CClose == sample_data.iloc[50]['Close']

    def test_index_property(self, sample_data):
        """Test Index property."""
        ds = DataSource(sample_data)
        assert isinstance(ds.Index, pd.Index)
        assert len(ds.Index) == 100


class TestCSVDataSource:
    def test_init_from_csv(self, tmp_path):
        """Test CSVDataSource initialization from CSV file."""
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'Low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Volume': [1000] * 10
        }, index=dates)

        csv_path = tmp_path / "test_data.csv"
        data.to_csv(csv_path)

        ds = CSVDataSource(str(csv_path))
        assert len(ds) > 0
        assert hasattr(ds, 'Open')
        assert hasattr(ds, 'Close')