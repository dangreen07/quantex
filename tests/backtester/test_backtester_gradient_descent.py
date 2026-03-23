import pandas as pd
import pytest

from quantex.backtester import OptimizationResult, SimpleBacktester
from tests.strategies.common import (
    DeterministicEntryExitStrategy,
    RiskAwareStrategy,
)


def test_optimize_gradient_descent_basic(datasource):
    """Test basic gradient descent optimization."""
    strategy = DeterministicEntryExitStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    result = backtester.optimize_gradient_descent(
        param_init={"dummy_param": 1.0},
        param_bounds={"dummy_param": (0.5, 2.5)},
        max_iterations=5,
        progress_bar=False,
    )

    assert isinstance(result, OptimizationResult)
    assert isinstance(result.best_params, dict)
    assert "dummy_param" in result.best_params


def test_optimize_gradient_descent_bounds_enforced(datasource):
    """Test that parameter bounds are enforced during gradient descent."""
    strategy = DeterministicEntryExitStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    result = backtester.optimize_gradient_descent(
        param_init={"dummy_param": 1.5},
        param_bounds={"dummy_param": (1.0, 2.0)},
        max_iterations=5,
        progress_bar=False,
    )

    assert result.best_params["dummy_param"] >= 1.0
    assert result.best_params["dummy_param"] <= 2.0


def test_optimize_gradient_descent_iteration_history(datasource):
    """Test that gradient descent returns iteration history."""
    strategy = DeterministicEntryExitStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    result = backtester.optimize_gradient_descent(
        param_init={"dummy_param": 1.0},
        param_bounds={"dummy_param": (0.5, 2.0)},
        max_iterations=5,
        tolerance=0.0,
        progress_bar=False,
    )

    assert isinstance(result.all_results, pd.DataFrame)
    if not result.all_results.empty:
        assert "iteration" in result.all_results.columns
        assert "train_score" in result.all_results.columns
        assert "validate_score" in result.all_results.columns
        assert "test_score" in result.all_results.columns
        assert "gradient_magnitude" in result.all_results.columns


def test_optimize_gradient_descent_invalid_criterion(datasource):
    """Test that invalid selection criterion raises ValueError."""
    strategy = RiskAwareStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    with pytest.raises(
        ValueError,
        match="selection_criterion must be one of",
    ):
        backtester.optimize_gradient_descent(
            param_init={"fast": 2.0},
            param_bounds={"fast": (1.0, 5.0)},
            selection_criterion="invalid",
            progress_bar=False,
        )


def test_optimize_gradient_descent_mismatched_params(datasource):
    """Test that mismatched param_init and param_bounds raises ValueError."""
    strategy = RiskAwareStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    with pytest.raises(
        ValueError,
        match="param_init and param_bounds must have the same keys",
    ):
        backtester.optimize_gradient_descent(
            param_init={"fast": 2.0},
            param_bounds={"slow": (1.0, 5.0)},
            progress_bar=False,
        )


def test_optimize_gradient_descent_empty_params(datasource):
    """Test that empty param_init raises ValueError."""
    strategy = RiskAwareStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    with pytest.raises(ValueError, match="param_init must not be empty"):
        backtester.optimize_gradient_descent(
            param_init={},
            param_bounds={},
            progress_bar=False,
        )


def test_optimize_gradient_descent_integer_params(datasource):
    """Test that integer_params rounds parameters to integers."""
    strategy = DeterministicEntryExitStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    result = backtester.optimize_gradient_descent(
        param_init={"dummy_param": 1.5},
        param_bounds={"dummy_param": (1.0, 3.0)},
        integer_params={"dummy_param"},
        max_iterations=5,
        tolerance=0.0,
        progress_bar=False,
    )

    assert result.best_params["dummy_param"] == int(
        result.best_params["dummy_param"]
    )