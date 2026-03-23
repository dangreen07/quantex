import pandas as pd
import pytest

from quantex.backtester import (
    BacktestReport,
    OptimizationResult,
    SimpleBacktester,
    TrainValidateTestSplit,
    create_train_validate_test_split,
)
from tests.strategies.common import DeterministicEntryExitStrategy


def test_create_train_validate_test_split_basic():
    """Test basic train/validate/test split creation."""
    split = create_train_validate_test_split(1000, 0.6, 0.2, 0.2)

    assert isinstance(split, TrainValidateTestSplit)
    assert split.train_start == 0
    assert split.train_end == 600
    assert split.validate_start == 600
    assert split.validate_end == 800
    assert split.test_start == 800
    assert split.test_end == 1000


def test_create_train_validate_test_split_invalid_ratios():
    """Test that invalid ratios raise ValueError."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        create_train_validate_test_split(1000, 0.5, 0.3, 0.1)

    with pytest.raises(
        ValueError,
        match="All split ratios must be positive",
    ):
        create_train_validate_test_split(1000, 0.0, 0.5, 0.5)


def test_optimize_with_split_basic(datasource):
    """Test basic optimize_with_split functionality."""
    strategy = DeterministicEntryExitStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    result = backtester.optimize_with_split(
        {"dummy_param": [1, 2]},
        selection_criterion="validate",
    )

    assert isinstance(result, OptimizationResult)
    assert isinstance(result.best_params, dict)
    assert isinstance(result.train_metrics, dict)
    assert isinstance(result.validate_metrics, dict)
    assert isinstance(result.test_metrics, dict)
    assert isinstance(result.all_results, pd.DataFrame)


def test_optimize_with_split_selection_criteria(datasource):
    """Test different selection criteria."""
    strategy = DeterministicEntryExitStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    result_train = backtester.optimize_with_split(
        {"dummy_param": [1, 2]},
        selection_criterion="train",
    )
    assert result_train.best_params is not None

    result_validate = backtester.optimize_with_split(
        {"dummy_param": [1, 2]},
        selection_criterion="validate",
    )
    assert result_validate.best_params is not None

    result_test = backtester.optimize_with_split(
        {"dummy_param": [1, 2]},
        selection_criterion="test",
    )
    assert result_test.best_params is not None


def test_optimize_with_split_invalid_selection_criterion(datasource):
    """Test that invalid selection criterion raises ValueError."""
    strategy = DeterministicEntryExitStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    with pytest.raises(
        ValueError,
        match="selection_criterion must be one of",
    ):
        backtester.optimize_with_split(
            {"dummy_param": [1]},
            selection_criterion="invalid",
        )


def test_optimize_with_split_custom_ratios(datasource):
    """Test optimize_with_split with custom split ratios."""
    strategy = DeterministicEntryExitStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    result = backtester.optimize_with_split(
        {"dummy_param": [1, 2]},
        train_ratio=0.7,
        validate_ratio=0.15,
        test_ratio=0.15,
        selection_criterion="validate",
    )

    assert isinstance(result, OptimizationResult)
    assert result.best_params is not None


def test_optimize_with_split_reports(datasource):
    """Test that optimize_with_split returns proper BacktestReports."""
    strategy = DeterministicEntryExitStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    result = backtester.optimize_with_split(
        {"dummy_param": [1, 2]},
        selection_criterion="validate",
    )

    assert result.train_report is None or isinstance(
        result.train_report,
        BacktestReport,
    )
    assert result.validate_report is None or isinstance(
        result.validate_report,
        BacktestReport,
    )
    assert result.test_report is None or isinstance(
        result.test_report,
        BacktestReport,
    )