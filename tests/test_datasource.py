import pytest
import pandas as pd
from quantex.datasource import DataSource, CSVDataSource


class TestDataSource:
    def test_init_valid_data(self, ohlcv_data):
        """Test DataSource initialization with valid data."""
        ds = DataSource(ohlcv_data)
        assert len(ds) == 20
        assert ds.current_index == 20
        assert ds.Open[0] == 100
        assert ds.Close[-1] == 121

    def test_init_missing_columns(self):
        """Test DataSource initialization with missing required columns."""
        data = pd.DataFrame({'Open': [1, 2], 'Close': [1, 2]})
        with pytest.raises(ValueError, match="Dataframe requires the following columns"):
            DataSource(data)

    def test_train_test_split_train(self, ohlcv_data):
        """Test train/test split in train mode."""
        ds = DataSource(ohlcv_data, train_test_split=True, mode="train")
        expected_len = int(len(ohlcv_data) * 0.8)
        assert len(ds) == expected_len

    def test_train_test_split_test(self, ohlcv_data):
        """Test train/test split in test mode."""
        ds = DataSource(ohlcv_data, train_test_split=True, mode="test")
        expected_len = len(ohlcv_data) - int(len(ohlcv_data) * 0.8)
        assert len(ds) == expected_len

    def test_properties(self, ohlcv_data):
        """Test DataSource properties return correct data."""
        ds = DataSource(ohlcv_data)
        ds.current_index = len(ohlcv_data) - 1

        assert len(ds.Open) == len(ohlcv_data)
        assert len(ds.High) == len(ohlcv_data)
        assert len(ds.Low) == len(ohlcv_data)
        assert len(ds.Close) == len(ohlcv_data)
        assert len(ds.Volume) == len(ohlcv_data)

        assert ds.COpen == ohlcv_data.iloc[-1]['Open']
        assert ds.CClose == ohlcv_data.iloc[-1]['Close']

    def test_index_property(self, ohlcv_data):
        """Test Index property."""
        ds = DataSource(ohlcv_data)
        assert isinstance(ds.Index, pd.Index)
        assert len(ds.Index) == 20

    def test_empty_dataframe_rejected(self):
        """Empty data should fail before any price access occurs."""
        data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        ds = DataSource(data)

        with pytest.raises(IndexError):
            _ = ds.COpen


class TestCSVDataSource:
    def test_init_from_csv(self, tmp_path):
        """Test CSVDataSource initialization from CSV file."""
        csv_path = tmp_path / "test_data.csv"
        pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "High": [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
                "Low": [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
                "Close": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "Volume": [1000] * 10,
            },
            index=pd.date_range("2020-01-01", periods=10, freq="D"),
        ).to_csv(csv_path)

        ds = CSVDataSource(str(csv_path))
        assert len(ds) > 0
        assert hasattr(ds, 'Open')
        assert hasattr(ds, 'Close')
