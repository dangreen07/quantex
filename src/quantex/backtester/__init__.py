# Re-export all public API from submodules for backward compatibility
from .constants import DataSplitMode
from ..broker.types import CommissionType
from .data_splits import TrainValidateTestSplit, create_train_validate_test_split
from .metrics import (
    max_drawdown,
    _infer_periods_per_year,
    _compute_backtest_metrics,
    _extract_metric_value,
    _risk_tolerance_passes,
)
from .reports import BacktestReport, OptimizationResult
from .backtester import SimpleBacktester

# Keep parallel exports for advanced users
from .parallel import _worker_init, _worker_eval

__all__ = [
    # Constants
    "DataSplitMode",
    "CommissionType",
    # Data splits
    "TrainValidateTestSplit",
    "create_train_validate_test_split",
    # Metrics
    "max_drawdown",
    "_infer_periods_per_year",
    "_compute_backtest_metrics",
    "_extract_metric_value",
    "_risk_tolerance_passes",
    # Reports
    "BacktestReport",
    "OptimizationResult",
    # Main classes
    "SimpleBacktester",
    # Parallel (advanced)
    "_worker_init",
    "_worker_eval",
]
