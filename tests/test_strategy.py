import pytest
import pandas as pd
import numpy as np
from quantex.strategy import Strategy
from quantex.datasource import DataSource


class TestStrategy:
    def test_abstract_methods(self):
        """Test that Strategy is abstract and requires implementation."""
        # We can't instantiate Strategy directly, but we can check it's abstract
        assert hasattr(Strategy, '__abstractmethods__')
        assert 'init' in Strategy.__abstractmethods__
        assert 'next' in Strategy.__abstractmethods__

    def test_strategy_base_attributes(self):
        """Test that DummyStrategy has the correct base attributes."""
        strategy = DummyStrategy()
        assert isinstance(strategy.positions, dict)
        assert isinstance(strategy.data, dict)
        assert isinstance(strategy.indicators, list)
        assert len(strategy.positions) == 0
        assert len(strategy.data) == 0
        assert len(strategy.indicators) == 0

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'Low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Volume': [1000] * 10
        }, index=dates)
        return data

    @pytest.fixture
    def datasource(self, sample_data):
        """Create a DataSource instance."""
        return DataSource(sample_data)

    def test_add_data(self, datasource):
        """Test adding data to strategy."""
        strategy = DummyStrategy()
        strategy.add_data(datasource, "EURUSD")

        assert "EURUSD" in strategy.data
        assert "EURUSD" in strategy.positions
        assert strategy.data["EURUSD"] is datasource
        assert hasattr(strategy.positions["EURUSD"], 'cash')

    def test_indicator(self):
        """Test adding an indicator."""
        strategy = DummyStrategy()
        arr = np.array([1, 2, 3, 4, 5])
        indicator = strategy.Indicator(arr)

        assert len(strategy.indicators) == 1
        assert hasattr(indicator, '_i')
        assert indicator._i == 5


class DummyStrategy(Strategy):
    """Concrete implementation of Strategy for testing."""

    def init(self):
        pass

    def next(self):
        pass


class TestDummyStrategy:
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'Low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Volume': [1000] * 10
        }, index=dates)
        return data

    @pytest.fixture
    def datasource(self, sample_data):
        """Create a DataSource instance."""
        return DataSource(sample_data)

    def test_concrete_implementation(self, datasource):
        """Test a concrete strategy implementation."""
        strategy = DummyStrategy()
        strategy.add_data(datasource, "EURUSD")

        # Should not raise NotImplementedError
        strategy.init()
        strategy.next()

        assert len(strategy.data) == 1
        assert len(strategy.positions) == 1