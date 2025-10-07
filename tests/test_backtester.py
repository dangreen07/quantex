import pytest
import pandas as pd
import numpy as np
from quantex.strategy import Strategy
from quantex.datasource import DataSource
from quantex.backtester import SimpleBacktester, BacktestReport, max_drawdown, _infer_periods_per_year
from quantex.enums import CommissionType


class DummyStrategy(Strategy):
    """Simple strategy for testing - buys on first bar, sells on last."""

    def init(self):
        pass

    def next(self):
        if len(self.data["EURUSD"].Close) == 2:  # Second bar
            self.positions["EURUSD"].buy(quantity=1.0)
        elif len(self.data["EURUSD"].Close) == len(self.data["EURUSD"].data) - 1:  # Second to last bar
            self.positions["EURUSD"].close()


class TestBacktester:
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            'Open': np.linspace(100, 110, 20),
            'High': np.linspace(105, 115, 20),
            'Low': np.linspace(95, 105, 20),
            'Close': np.linspace(102, 112, 20),
            'Volume': [1000] * 20
        }, index=dates)
        return data

    @pytest.fixture
    def datasource(self, sample_data):
        """Create a DataSource instance."""
        return DataSource(sample_data)

    @pytest.fixture
    def strategy(self, datasource):
        """Create a strategy instance."""
        strat = DummyStrategy()
        strat.add_data(datasource, "EURUSD")
        return strat

    @pytest.fixture
    def backtester(self, strategy):
        """Create a SimpleBacktester instance."""
        return SimpleBacktester(strategy)

    def test_init(self, backtester):
        """Test SimpleBacktester initialization."""
        assert backtester.cash == 10000
        assert backtester.commission == 0.002
        assert backtester.commission_type == CommissionType.PERCENTAGE
        assert backtester.lot_size == 1
        assert isinstance(backtester.PnLRecord, np.ndarray)

    def test_run(self, backtester):
        """Test running a backtest."""
        report = backtester.run(progress_bar=False)

        assert isinstance(report, BacktestReport)
        assert isinstance(report.starting_cash, np.float64)
        assert isinstance(report.final_cash, np.float64)
        assert isinstance(report.PnlRecord, pd.Series)
        assert isinstance(report.orders, list)

        # Check that PnL record has correct length
        assert len(report.PnlRecord) == len(backtester.strategy.data["EURUSD"].data)

    def test_backtest_report_str(self, backtester):
        """Test BacktestReport string representation."""
        report = backtester.run(progress_bar=False)
        report_str = str(report)

        assert "Starting Cash" in report_str
        assert "Final Cash" in report_str
        assert "Total Return" in report_str
        assert "Sharpe Ratio" in report_str
        assert "Max Drawdown" in report_str
        assert "Total Trades" in report_str

    def test_optimize_empty_params(self, backtester):
        """Test optimization with empty parameters."""
        with pytest.raises(ValueError, match="params must not be empty"):
            backtester.optimize({})

    def test_optimize_invalid_param_type(self, backtester):
        """Test optimization with non-iterable parameter."""
        with pytest.raises(TypeError, match="Parameter 'test' must be iterable"):
            backtester.optimize({"test": 5})

    def test_optimize_empty_param_values(self, backtester):
        """Test optimization with empty parameter values."""
        with pytest.raises(ValueError, match="Parameter 'test' has no candidate values"):
            backtester.optimize({"test": []})

    def test_optimize_basic(self, backtester):
        """Test basic optimization functionality."""
        # Simple parameter that doesn't affect the strategy
        params = {"dummy_param": [1, 2]}

        # Add dummy_param to strategy
        backtester.strategy.dummy_param = 1

        best_params, best_report, results_df = backtester.optimize(params)

        assert isinstance(best_params, dict)
        assert isinstance(best_report, BacktestReport)
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 2  # Two parameter combinations

    def test_max_drawdown(self):
        """Test max_drawdown function."""
        # Equity that goes up then down
        equity = pd.Series([100, 110, 105, 95, 100])
        mdd = max_drawdown(equity)
        expected_mdd = (95 - 110) / 110  # Drawdown from peak to trough
        assert abs(mdd - abs(expected_mdd)) < 1e-6

    def test_infer_periods_per_year(self):
        """Test _infer_periods_per_year function."""
        # Daily data
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        periods = _infer_periods_per_year(dates)
        assert periods == 252  # 252 trading days/year for daily data

        # Empty index
        periods = _infer_periods_per_year(pd.DatetimeIndex([]))
        assert periods == 252 * 24 * 60  # Default for minute data

    def test_periods_per_year_property(self, backtester):
        """Test periods_per_year property of BacktestReport."""
        report = backtester.run(progress_bar=False)
        periods = report.periods_per_year
        assert isinstance(periods, int)
        assert periods > 0