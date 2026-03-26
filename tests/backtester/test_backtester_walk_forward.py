"""
Tests for walk-forward optimization functionality.
"""

import pandas as pd
import pytest

from quantex.backtester import SimpleBacktester
from quantex.backtester.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardResult,
    WalkForwardWindow,
    walk_forward_analyze,
)
from quantex.datasource import DataSource
from tests.strategies.common import (
    IndicatorResetStrategy,
    ParametrizedHoldPeriodStrategy,
)


@pytest.fixture
def ohlcv_data_large():
    """Larger OHLCV data for walk-forward testing.
    
    100 periods allows for multiple windows with train_periods=30 and test_periods=20.
    """
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "Open": list(range(100, 200)),
            "High": list(range(105, 205)),
            "Low": list(range(95, 195)),
            "Close": list(range(102, 202)),
            "Volume": [1000] * 100,
        },
        index=dates,
    )


@pytest.fixture
def large_strategy(ohlcv_data_large):
    """Strategy backed by larger deterministic data."""
    datasource = DataSource(ohlcv_data_large)
    strat = IndicatorResetStrategy()
    strat.add_data(datasource, "EURUSD")
    return strat


@pytest.fixture
def large_backtester(large_strategy):
    """Backtester with larger data."""
    return SimpleBacktester(large_strategy)


@pytest.fixture
def hold_strategy(ohlcv_data_large):
    """Parametrized strategy for hold periods."""
    datasource = DataSource(ohlcv_data_large)
    strat = ParametrizedHoldPeriodStrategy()
    strat.add_data(datasource, "EURUSD")
    return strat


@pytest.fixture
def hold_backtester(hold_strategy):
    """Backtester with hold strategy."""
    return SimpleBacktester(hold_strategy)


class TestWalkForwardAnalyzer:
    """Tests for WalkForwardAnalyzer class."""

    def test_analyzer_initialization(self, large_backtester):
        """Analyzer should initialize with valid parameters."""
        analyzer = WalkForwardAnalyzer(
            backtester=large_backtester,
            train_periods=30,
            test_periods=10,
        )
        assert analyzer.train_periods == 30
        assert analyzer.test_periods == 10
        assert analyzer.step_periods == 10  # Default to test_periods

    def test_analyzer_with_custom_step(self, large_backtester):
        """Analyzer should accept custom step periods."""
        analyzer = WalkForwardAnalyzer(
            backtester=large_backtester,
            train_periods=30,
            test_periods=10,
            step_periods=5,
        )
        assert analyzer.step_periods == 5

    def test_analyzer_validates_train_periods(self, large_backtester):
        """Analyzer should reject train_periods below minimum."""
        with pytest.raises(ValueError, match="train_periods must be at least"):
            WalkForwardAnalyzer(
                backtester=large_backtester,
                train_periods=5,  # Below minimum of 30
                test_periods=10,
            )

    def test_analyzer_validates_test_periods(self, large_backtester):
        """Analyzer should reject test_periods below minimum."""
        with pytest.raises(ValueError, match="test_periods must be at least"):
            WalkForwardAnalyzer(
                backtester=large_backtester,
                train_periods=30,
                test_periods=5,  # Below minimum of 10
            )

    def test_creates_correct_number_of_windows(self, large_backtester):
        """Analyzer should create correct number of walk-forward windows."""
        # 100 periods, train=30, test=10, step=10 -> 7 windows
        # Window 0: train 0-30, test 30-40 (test_end=40 <= 100 ✓)
        # Window 1: train 10-40, test 40-50 (test_end=50 <= 100 ✓)
        # Window 2: train 20-50, test 50-60 (test_end=60 <= 100 ✓)
        # Window 3: train 30-60, test 60-70 (test_end=70 <= 100 ✓)
        # Window 4: train 40-70, test 70-80 (test_end=80 <= 100 ✓)
        # Window 5: train 50-80, test 80-90 (test_end=90 <= 100 ✓)
        # Window 6: train 60-90, test 90-100 (test_end=100 <= 100 ✓)
        # Window 7: train 70-100, test 100-110 (test_end=110 > 100 ✗)
        analyzer = WalkForwardAnalyzer(
            backtester=large_backtester,
            train_periods=30,
            test_periods=10,
            step_periods=10,
        )
        splits = analyzer._create_window_splits()
        assert len(splits) == 7

    def test_windows_are_sequential(self, large_backtester):
        """Windows should be properly sequential."""
        analyzer = WalkForwardAnalyzer(
            backtester=large_backtester,
            train_periods=30,
            test_periods=10,
            step_periods=10,
        )
        splits = analyzer._create_window_splits()
        
        for i in range(len(splits) - 1):
            current = splits[i]
            next_window = splits[i + 1]
            # Train start of next window should equal step from current
            assert next_window[0] == current[0] + 10


class TestWalkForwardResult:
    """Tests for WalkForwardResult class."""

    def test_result_properties(self):
        """Result should expose computed properties."""
        result = WalkForwardResult(
            n_windows=3,
            train_periods=30,
            test_periods=10,
            window_results=[
                WalkForwardWindow(
                    window_index=i,
                    train_start=i * 10,
                    train_end=i * 10 + 30,
                    test_start=i * 10 + 30,
                    test_end=i * 10 + 40,
                    train_periods=30,
                    test_periods=10,
                    best_params={"fast": 5, "slow": 10},
                    train_metrics={"sharpe": 1.0 + i * 0.1, "total_return": 0.05},
                    test_metrics={"sharpe": 0.8 + i * 0.1, "total_return": 0.03},
                )
                for i in range(3)
            ],
        )
        
        assert len(result.in_sample_sharpe) == 3
        assert len(result.out_of_sample_sharpe) == 3
        assert result.in_sample_sharpe[0] == 1.0
        assert result.out_of_sample_sharpe[0] == 0.8

    def test_param_stability_analysis(self):
        """Result should compute parameter stability correctly."""
        result = WalkForwardResult(
            n_windows=3,
            train_periods=30,
            test_periods=10,
            window_results=[
                WalkForwardWindow(
                    window_index=i,
                    train_start=0,
                    train_end=30,
                    test_start=30,
                    test_end=40,
                    train_periods=30,
                    test_periods=10,
                    best_params={"fast": 5.0 + i},
                    train_metrics={"sharpe": 1.0},
                    test_metrics={"sharpe": 0.8},
                )
                for i in range(3)
            ],
        )
        
        stability = result.get_param_stability("fast")
        assert stability["mean"] == 6.0
        assert stability["std"] > 0
        assert stability["min"] == 5.0
        assert stability["max"] == 7.0


class TestWalkForwardAnalyze:
    """Tests for the walk_forward_analyze function."""

    def test_basic_walk_forward(self, hold_backtester):
        """Walk-forward should run without errors."""
        result = walk_forward_analyze(
            backtester=hold_backtester,
            optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
            params={"hold_period": [1, 2, 3]},
            train_periods=30,
            test_periods=10,
            progress_bar=False,
        )
        
        assert isinstance(result, WalkForwardResult)
        assert result.n_windows >= 1
        assert len(result.window_results) == result.n_windows

    def test_returns_best_params_per_window(self, hold_backtester):
        """Each window should have its own best parameters."""
        result = walk_forward_analyze(
            backtester=hold_backtester,
            optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
            params={"hold_period": [1, 2, 3]},
            train_periods=30,
            test_periods=10,
            progress_bar=False,
        )
        
        for window in result.window_results:
            assert "hold_period" in window.best_params
            assert window.best_params["hold_period"] in [1, 2, 3]

    def test_aggregated_metrics_computed(self, hold_backtester):
        """Aggregated metrics should be computed."""
        result = walk_forward_analyze(
            backtester=hold_backtester,
            optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
            params={"hold_period": [1, 2]},
            train_periods=30,
            test_periods=10,
            progress_bar=False,
        )
        
        assert "in_sample_sharpe_mean" in result.aggregated_metrics
        assert "out_of_sample_sharpe_mean" in result.aggregated_metrics
        assert "oos_to_is_ratio_mean" in result.aggregated_metrics

    def test_results_dataframe_has_all_columns(self, hold_backtester):
        """Results DataFrame should contain expected columns."""
        result = walk_forward_analyze(
            backtester=hold_backtester,
            optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
            params={"hold_period": [1, 2]},
            train_periods=30,
            test_periods=10,
            progress_bar=False,
        )
        
        df = result.all_windows_results_df
        assert not df.empty
        assert "window" in df.columns
        assert "train_start" in df.columns
        assert "test_start" in df.columns
        assert "param_hold_period" in df.columns
        assert "train_sharpe" in df.columns
        assert "test_sharpe" in df.columns

    def test_out_of_sample_metrics_different_from_in_sample(self, hold_backtester):
        """Out-of-sample metrics should generally differ from in-sample."""
        result = walk_forward_analyze(
            backtester=hold_backtester,
            optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
            params={"hold_period": [1, 2, 3]},
            train_periods=30,
            test_periods=20,  # Larger test window
            progress_bar=False,
        )
        
        # Check that some windows have different in-sample vs out-of-sample metrics
        # (This is expected due to overfitting in training)
        is_sharpe = result.in_sample_sharpe
        oos_sharpe = result.out_of_sample_sharpe
        
        # At least one window should have different metrics
        differences = [abs(a - b) > 0.001 for a, b in zip(is_sharpe, oos_sharpe)]
        assert any(differences), "Expected different in-sample vs out-of-sample metrics"

    def test_stability_ratio_computed(self, hold_backtester):
        """Stability ratio (OOS/IS) should be computed."""
        result = walk_forward_analyze(
            backtester=hold_backtester,
            optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
            params={"hold_period": [1, 2]},
            train_periods=30,
            test_periods=10,
            progress_bar=False,
        )
        
        agg = result.aggregated_metrics
        assert "oos_to_is_ratio_mean" in agg
        assert "oos_to_is_ratio_std" in agg
        # Ratio can be positive, negative, or NaN depending on data
        assert agg["oos_to_is_ratio_mean"] is not None


class TestWalkForwardWithDifferentOptimizers:
    """Tests for walk-forward with different optimizer functions."""

    def test_with_parallel_optimizer(self, hold_backtester):
        """Walk-forward should work with parallel optimizer."""
        result = walk_forward_analyze(
            backtester=hold_backtester,
            optimizer=lambda bt, params, **kwargs: bt.optimize_parallel(
                params,
                workers=1,  # Use single worker for testing
                **kwargs,
            ),
            params={"hold_period": [1, 2, 3]},
            train_periods=30,
            test_periods=10,
            progress_bar=False,
        )
        
        assert isinstance(result, WalkForwardResult)
        assert result.n_windows >= 1

    def test_custom_optimizer_function(self, hold_backtester):
        """Walk-forward should work with custom optimizer function."""
        def custom_optimizer(bt, params, **kwargs):
            """Simple custom optimizer that always returns same params."""
            return bt.optimize(params, **kwargs)
        
        result = walk_forward_analyze(
            backtester=hold_backtester,
            optimizer=custom_optimizer,
            params={"hold_period": [1, 2]},
            train_periods=30,
            test_periods=10,
            progress_bar=False,
        )
        
        assert isinstance(result, WalkForwardResult)
        assert len(result.window_results) >= 1


class TestWalkForwardEdgeCases:
    """Tests for edge cases and error handling."""

    def test_data_too_short(self, ohlcv_data):
        """Should raise error when data is too short for configuration."""
        # Create backtester with small data (20 periods)
        datasource = DataSource(ohlcv_data)
        strat = ParametrizedHoldPeriodStrategy()
        strat.add_data(datasource, "EURUSD")
        bt = SimpleBacktester(strat)
        
        # train=30, test=10, but data is only 20 periods
        with pytest.raises(ValueError, match="Data length"):
            walk_forward_analyze(
                backtester=bt,
                optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
                params={"hold_period": [1, 2]},
                train_periods=30,
                test_periods=10,
                progress_bar=False,
            )

    def test_overlapping_windows(self, large_backtester):
        """Should support overlapping windows (step < test_periods)."""
        result = walk_forward_analyze(
            backtester=large_backtester,
            optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
            params={"hold_period": [1, 2]},
            train_periods=30,
            test_periods=10,
            step_periods=5,  # Overlapping
            progress_bar=False,
        )
        
        assert result.n_windows >= 1
        # More windows when step is smaller
        assert result.n_windows > 6  # Non-overlapping would have 6

    def test_no_params_that_produce_results(self, large_backtester):
        """Should handle parameter space with no valid combinations."""
        result = walk_forward_analyze(
            backtester=large_backtester,
            optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
            params={"dummy_param": [1]},  # Single value
            train_periods=30,
            test_periods=10,
            progress_bar=False,
        )
        
        assert isinstance(result, WalkForwardResult)
        # Should still produce results even with single param value


class TestWalkForwardAnalyzerClass:
    """Tests for using WalkForwardAnalyzer class directly."""

    def test_class_interface(self, hold_backtester):
        """Analyzer class should work correctly."""
        analyzer = WalkForwardAnalyzer(
            backtester=hold_backtester,
            train_periods=30,
            test_periods=10,
        )
        
        result = analyzer.analyze(
            optimizer=lambda bt, params, **kwargs: bt.optimize(params, **kwargs),
            params={"hold_period": [1, 2]},
            progress_bar=False,
        )
        
        assert isinstance(result, WalkForwardResult)
        assert result.n_windows == analyzer._create_window_splits().__len__()

    def test_class_passes_kwargs_to_optimizer(self, hold_backtester):
        """Analyzer should pass additional kwargs to optimizer."""
        analyzer = WalkForwardAnalyzer(
            backtester=hold_backtester,
            train_periods=30,
            test_periods=10,
        )
        
        # Using parallel optimizer with workers kwarg
        result = analyzer.analyze(
            optimizer=lambda bt, params, **kw: bt.optimize_parallel(params, **kw),
            params={"hold_period": [1, 2]},
            workers=1,
            progress_bar=False,
        )
        
        assert isinstance(result, WalkForwardResult)


class TestWalkForwardStringRepresentation:
    """Tests for string representations."""

    def test_result_str(self):
        """WalkForwardResult should have readable string representation."""
        result = WalkForwardResult(
            n_windows=3,
            train_periods=30,
            test_periods=10,
            window_results=[
                WalkForwardWindow(
                    window_index=i,
                    train_start=0,
                    train_end=30,
                    test_start=30,
                    test_end=40,
                    train_periods=30,
                    test_periods=10,
                    best_params={"fast": 5},
                    train_metrics={"sharpe": 1.0, "total_return": 0.05},
                    test_metrics={"sharpe": 0.8, "total_return": 0.03},
                )
                for i in range(3)
            ],
            aggregated_metrics={
                "in_sample_sharpe_mean": 1.0,
                "in_sample_sharpe_std": 0.1,
                "in_sample_return_mean": 0.05,
                "in_sample_return_std": 0.01,
                "out_of_sample_sharpe_mean": 0.8,
                "out_of_sample_sharpe_std": 0.1,
                "out_of_sample_return_mean": 0.03,
                "out_of_sample_return_std": 0.01,
                "oos_to_is_ratio_mean": 0.8,
                "oos_to_is_ratio_std": 0.1,
            },
        )
        
        result_str = str(result)
        assert "Walk-Forward Optimization Results" in result_str
        assert "Total Windows: 3" in result_str
        assert "Train Periods per Window: 30" in result_str
        assert "Test Periods per Window: 10" in result_str
        assert "In-Sample" in result_str
        assert "Out-of-Sample" in result_str
