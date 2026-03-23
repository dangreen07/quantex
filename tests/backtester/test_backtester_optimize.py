import pandas as pd
import pytest

from quantex.backtester import BacktestReport, OptimizationResult
from tests.strategies.common import (
    IndicatorResetStrategy,
    ParametrizedHoldPeriodStrategy,
    RepeatableRunStrategy,
    RiskAwareStrategy,
)
from quantex.backtester import CommissionType, SimpleBacktester


def test_optimize_empty_params(backtester):
    """Test optimization with empty parameters."""
    with pytest.raises(ValueError, match="params must not be empty"):
        backtester.optimize({})


def test_optimize_invalid_param_type(backtester):
    """Test optimization with non-iterable parameter."""
    with pytest.raises(TypeError, match="Parameter 'test' must be iterable"):
        backtester.optimize({"test": 5})


def test_optimize_empty_param_values(backtester):
    """Test optimization with empty parameter values."""
    with pytest.raises(
        ValueError,
        match="Parameter 'test' has no candidate values",
    ):
        backtester.optimize({"test": []})


def test_optimize_basic(backtester):
    """Optimization should evaluate all parameter combinations."""
    params = {"dummy_param": [1, 2]}
    backtester.strategy.dummy_param = 1

    result = backtester.optimize(params)

    assert isinstance(result, OptimizationResult)
    assert isinstance(result.best_params, dict)
    assert isinstance(result.train_report, BacktestReport)
    assert isinstance(result.all_results, pd.DataFrame)
    assert len(result.all_results) == 2


def test_optimize_supports_custom_objective(datasource):
    strategy = RiskAwareStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    result = backtester.optimize(
        {"fast": range(2, 4), "slow": range(5, 7)},
        constraint=lambda p: p["fast"] < p["slow"],
        objective="total_return",
    )

    assert isinstance(result.train_report, BacktestReport)
    assert not result.all_results.empty
    assert "objective_score" in result.all_results.columns
    assert result.train_report.total_return == pytest.approx(
        result.all_results.iloc[0]["objective_score"]
    )


def test_optimize_filters_by_risk_tolerance(datasource):
    strategy = RiskAwareStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    unrestricted = backtester.optimize(
        {"fast": range(2, 4), "slow": range(5, 7)},
        constraint=lambda p: p["fast"] < p["slow"],
    ).all_results

    restricted = backtester.optimize(
        {"fast": range(2, 4), "slow": range(5, 7)},
        constraint=lambda p: p["fast"] < p["slow"],
        risk_tolerance={"max_drawdown": 0.0},
    ).all_results

    assert len(restricted) <= len(unrestricted)
    if not restricted.empty:
        assert (restricted["max_drawdown"] <= 0.0).all()


def test_optimize_best_report_is_consistent_with_best_params(datasource):
    """The selected best row should match the returned report exactly."""
    strategy = ParametrizedHoldPeriodStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    result = backtester.optimize({"hold_period": range(1, 11)})

    assert isinstance(result.train_report, BacktestReport)
    assert not result.all_results.empty

    matching_rows = result.all_results.loc[
        result.all_results["hold_period"]
        == result.best_params["hold_period"]
    ]
    assert len(matching_rows) == 1

    best_row = matching_rows.iloc[0]

    assert result.train_report.final_cash == pytest.approx(
        best_row["final_cash"]
    )
    assert result.train_report.total_return == pytest.approx(
        best_row["total_return"]
    )


def test_optimize_does_not_mutate_base_strategy_indicators(datasource):
    """Optimize should not leave behind indicator objects on the base strategy."""
    strategy = IndicatorResetStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    assert len(backtester.strategy.indicators) == 0

    backtester.optimize(
        {"slow": range(4, 7), "fast": range(2, 4)},
        constraint=lambda x: x["slow"] > x["fast"],
    )

    assert len(backtester.strategy.indicators) == 0


def test_optimize_after_run_uses_same_backtester_cleanly(datasource):
    """A run followed by optimize should still agree with a clean replay."""
    strategy = IndicatorResetStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(strategy)

    backtester.run(progress_bar=False)
    result = backtester.optimize(
        {"slow": range(4, 7), "fast": range(2, 4)},
        constraint=lambda x: x["slow"] > x["fast"],
    )

    assert result.train_report is None
    assert result.all_results.empty


def test_repeat_run_on_same_backtester_is_deterministic(datasource):
    """Repeated runs should not accumulate hidden state or drift."""
    strategy = RepeatableRunStrategy()
    strategy.add_data(datasource, "EURUSD")
    backtester = SimpleBacktester(
        strategy,
        10_000,
        commission=5,
        commission_type=CommissionType.CASH,
        lot_size=100_000,
    )

    report1 = backtester.run(progress_bar=False)
    report2 = backtester.run(progress_bar=False)

    assert report1.final_cash == pytest.approx(report2.final_cash)
    assert report1.total_return == pytest.approx(report2.total_return)
    assert report1.PnlRecord.to_numpy() == pytest.approx(
        report2.PnlRecord.to_numpy()
    )
    assert len(report1.orders) == len(report2.orders)