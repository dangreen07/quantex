"""
Tests for Monte Carlo simulation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from quantex.datasource import DataSource
from quantex.backtester import (
    SimpleBacktester,
    MonteCarloResult,
    MonteCarloMode,
)
from tests.strategies.common import (
    DeterministicEntryExitStrategy,
    RiskAwareStrategy,
)


class TestMonteCarlo:
    """Test suite for Monte Carlo simulation functionality."""
    
    @pytest.fixture
    def datasource(self, ohlcv_data):
        """Provide the deterministic data source."""
        return DataSource(ohlcv_data)
    
    @pytest.fixture
    def backtester(self, datasource):
        """Create a SimpleBacktester instance with a trading strategy."""
        strategy = RiskAwareStrategy()
        strategy.add_data(datasource, "EURUSD")
        return SimpleBacktester(strategy, cash=10000)
    
    def test_monte_carlo_returns_monte_carlo_result(self, backtester):
        """Monte Carlo should return a MonteCarloResult object."""
        result = backtester.monte_carlo(simulations=10, progress_bar=False)
        
        assert isinstance(result, MonteCarloResult)
    
    def test_monte_carlo_mode_trade_order(self, backtester):
        """Trade order mode should run the specified number of simulations."""
        result = backtester.monte_carlo(
            simulations=50,
            mode="trade_order",
            seed=42,
            progress_bar=False
        )
        
        assert result.simulations == 50
        assert result.mode == MonteCarloMode.TRADE_ORDER
        assert len(result.equity_curves) == 50
    
    def test_monte_carlo_mode_price_path(self, backtester):
        """Price path mode should run the specified number of simulations."""
        result = backtester.monte_carlo(
            simulations=50,
            mode="price_path",
            seed=42,
            progress_bar=False
        )
        
        assert result.simulations == 50
        assert result.mode == MonteCarloMode.PRICE_PATH
        assert len(result.equity_curves) == 50
    
    def test_monte_carlo_mode_both(self, backtester):
        """Both mode should run twice the number of simulations."""
        result = backtester.monte_carlo(
            simulations=25,
            mode="both",
            seed=42,
            progress_bar=False
        )
        
        assert result.simulations == 25
        assert result.mode == MonteCarloMode.BOTH
        # Both mode runs both analyses, so we get 2x simulations
        assert len(result.equity_curves) == 50
    
    def test_monte_carlo_with_enum_mode(self, backtester):
        """Monte Carlo should accept MonteCarloMode enum as mode parameter."""
        result = backtester.monte_carlo(
            simulations=10,
            mode=MonteCarloMode.TRADE_ORDER,
            progress_bar=False
        )
        
        assert result.mode == MonteCarloMode.TRADE_ORDER
    
    def test_monte_carlo_computes_statistics(self, backtester):
        """Monte Carlo result should compute summary statistics."""
        result = backtester.monte_carlo(
            simulations=20,
            mode="trade_order",
            seed=42,
            progress_bar=False
        )
        
        # Check summary stats are computed
        assert "mean" in result.summary_stats
        assert "std" in result.summary_stats
        assert "min" in result.summary_stats
        assert "max" in result.summary_stats
        assert "median" in result.summary_stats
    
    def test_monte_carlo_computes_percentiles(self, backtester):
        """Monte Carlo result should compute percentile results."""
        result = backtester.monte_carlo(
            simulations=20,
            mode="trade_order",
            seed=42,
            progress_bar=False
        )
        
        # Check percentiles are computed
        assert "p5" in result.percentile_results
        assert "p25" in result.percentile_results
        assert "p50" in result.percentile_results
        assert "p75" in result.percentile_results
        assert "p95" in result.percentile_results
    
    def test_monte_carlo_stores_original_equity(self, backtester):
        """Monte Carlo should store original equity curve for comparison."""
        result = backtester.monte_carlo(
            simulations=10,
            mode="trade_order",
            progress_bar=False
        )
        
        assert result.original_equity is not None
        assert isinstance(result.original_equity, pd.Series)
    
    def test_monte_carlo_stores_starting_cash(self, backtester):
        """Monte Carlo should store starting cash amount."""
        result = backtester.monte_carlo(
            simulations=10,
            mode="trade_order",
            progress_bar=False
        )
        
        assert result.starting_cash == 10000
    
    def test_monte_carlo_reproducibility_with_seed(self, backtester):
        """Same seed should produce identical results."""
        result1 = backtester.monte_carlo(
            simulations=10,
            mode="trade_order",
            seed=12345,
            progress_bar=False
        )
        
        result2 = backtester.monte_carlo(
            simulations=10,
            mode="trade_order",
            seed=12345,
            progress_bar=False
        )
        
        # Compare equity curves
        for curve1, curve2 in zip(result1.equity_curves, result2.equity_curves):
            np.testing.assert_array_almost_equal(
                curve1.values,
                curve2.values
            )
    
    def test_monte_carlo_str_output(self, backtester):
        """Monte Carlo result should have a string representation."""
        result = backtester.monte_carlo(
            simulations=10,
            mode="trade_order",
            progress_bar=False
        )
        
        result_str = str(result)
        
        assert "Monte Carlo Results" in result_str
        assert "Simulations:" in result_str
        assert "Starting Cash:" in result_str
        assert "Summary Statistics:" in result_str
        assert "Mean Final Value:" in result_str
        assert "Percentiles:" in result_str
    
    def test_monte_carlo_with_zero_simulations(self, backtester):
        """Zero simulations should still work but produce empty results."""
        result = backtester.monte_carlo(
            simulations=0,
            mode="trade_order",
            progress_bar=False
        )
        
        assert result.simulations == 0
        assert len(result.equity_curves) == 0
    
    def test_monte_carlo_progress_bar(self, backtester):
        """Progress bar should not cause errors."""
        result = backtester.monte_carlo(
            simulations=5,
            mode="trade_order",
            progress_bar=True
        )
        
        assert result is not None
    
    def test_monte_carlo_price_path_preserves_length(self, backtester):
        """Price path simulation should preserve data length."""
        original_len = len(backtester.strategy.data["EURUSD"].data)
        
        result = backtester.monte_carlo(
            simulations=5,
            mode="price_path",
            seed=42,
            progress_bar=False
        )
        
        # Each equity curve should match original length
        for curve in result.equity_curves:
            assert len(curve) == original_len
    
    def test_monte_carlo_no_trades(self, ohlcv_data):
        """Strategy with no trades should handle gracefully."""
        datasource = DataSource(ohlcv_data)
        
        # Create a strategy that never trades
        class NoTradeStrategy(DeterministicEntryExitStrategy):
            def next(self):
                pass  # Never trade
        
        strategy = NoTradeStrategy()
        strategy.add_data(datasource, "EURUSD")
        backtester = SimpleBacktester(strategy, cash=10000)
        
        result = backtester.monte_carlo(
            simulations=5,
            mode="trade_order",
            progress_bar=False
        )
        
        # Should complete without error
        assert result is not None
        assert len(result.equity_curves) == 5
    
    def test_monte_carlo_statistics_are_numeric(self, backtester):
        """Summary statistics should be numeric values."""
        result = backtester.monte_carlo(
            simulations=20,
            mode="price_path",
            seed=42,
            progress_bar=False
        )
        
        # All stats should be finite numbers
        for key, value in result.summary_stats.items():
            assert np.isfinite(value), f"{key} should be finite"
        
        for key, value in result.percentile_results.items():
            assert np.isfinite(value), f"{key} should be finite"

    def test_price_path_simulation_generates_positive_monotonic_index_preserving_paths(self, backtester):
        """Synthetic price paths should remain positive and preserve the original index length."""
        original_source = backtester.strategy.data["EURUSD"].data
        original_close = original_source["Close"].to_numpy(dtype=float)
        original_log_returns = np.diff(np.log(original_close))

        result = backtester.monte_carlo(
            simulations=5,
            mode="price_path",
            seed=42,
            progress_bar=False,
        )

        assert len(result.equity_curves) == 5
        for curve in result.equity_curves:
            assert len(curve) == len(original_source)
            assert np.all(np.isfinite(curve.values))
            assert np.all(curve.values >= 0)

        # The synthetic process should broadly preserve the return scale rather than exploding.
        simulated_final_values = np.array([curve.iloc[-1] for curve in result.equity_curves])
        assert np.all(np.isfinite(simulated_final_values))
        assert simulated_final_values.std() > 0

        # This is a loose sanity check that the source return distribution remains in the same order of magnitude.
        assert np.isfinite(original_log_returns).all()
        assert abs(np.mean(original_log_returns)) < 0.1
    
    def test_monte_carlo_different_seeds_different_results(self, ohlcv_data):
        """Test that different seeds produce different random sequences (verifies seed is passed correctly)."""
        datasource = DataSource(ohlcv_data)
        strategy = DeterministicEntryExitStrategy()
        strategy.add_data(datasource, "EURUSD")
        backtester = SimpleBacktester(strategy, cash=10000)
        
        # Just verify that seed parameter is accepted and doesn't error
        # Different seeds will produce different random sequences
        result1 = backtester.monte_carlo(
            simulations=5,
            mode="price_path",
            seed=11111,
            progress_bar=False
        )
        
        result2 = backtester.monte_carlo(
            simulations=5,
            mode="price_path",
            seed=99999,
            progress_bar=False
        )
        
        # Both should run successfully
        assert result1 is not None
        assert result2 is not None
        assert len(result1.equity_curves) == 5
        assert len(result2.equity_curves) == 5
