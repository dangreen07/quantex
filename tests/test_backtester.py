import pytest
import pandas as pd
import numpy as np
from quantex.datasource import DataSource
from quantex.backtester import SimpleBacktester, BacktestReport, max_drawdown, _infer_periods_per_year
from quantex.enums import CommissionType
from tests.strategies.common import (
    DeterministicEntryExitStrategy,
    IndicatorResetStrategy,
    ParametrizedHoldPeriodStrategy,
    RepeatableRunStrategy,
    RiskAwareStrategy,
)


class TestBacktester:
    @pytest.fixture
    def strategy(self, ohlcv_data):
        """Create a strategy instance backed by deterministic sample data."""
        datasource = DataSource(ohlcv_data)
        strat = DeterministicEntryExitStrategy()
        strat.add_data(datasource, "EURUSD")
        return strat

    @pytest.fixture
    def datasource(self, ohlcv_data):
        """Provide the same deterministic source used by strategy fixtures.

        Several optimizer tests need direct access to the source so they can
        create fresh strategies without depending on implicit fixture wiring.
        """
        return DataSource(ohlcv_data)

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
        """A normal run should produce a full report with one equity series."""
        report = backtester.run(progress_bar=False)

        assert isinstance(report, BacktestReport)
        assert isinstance(report.starting_cash, np.float64)
        assert isinstance(report.final_cash, np.float64)
        assert isinstance(report.PnlRecord, pd.Series)
        assert isinstance(report.orders, list)

        # Check that PnL record has correct length
        assert len(report.PnlRecord) == len(backtester.strategy.data["EURUSD"].data)

    def test_backtest_report_str(self, backtester):
        """String formatting should surface the key performance summary fields."""
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
        """Optimization should evaluate all parameter combinations."""
        params = {"dummy_param": [1, 2]}
        backtester.strategy.dummy_param = 1

        best_params, best_report, results_df = backtester.optimize(params)

        assert isinstance(best_params, dict)
        assert isinstance(best_report, BacktestReport)
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 2  # Two parameter combinations

    def test_optimize_supports_custom_objective(self, datasource):
        strategy = RiskAwareStrategy()
        strategy.add_data(datasource, "EURUSD")
        backtester = SimpleBacktester(strategy)

        best_params, best_report, results_df = backtester.optimize(
            {"fast": range(2, 4), "slow": range(5, 7)},
            constraint=lambda p: p["fast"] < p["slow"],
            objective="total_return",
        )

        assert isinstance(best_report, BacktestReport)
        assert not results_df.empty
        assert "objective_score" in results_df.columns
        assert best_report.total_return == pytest.approx(results_df.iloc[0]["objective_score"])

    def test_optimize_filters_by_risk_tolerance(self, datasource):
        strategy = RiskAwareStrategy()
        strategy.add_data(datasource, "EURUSD")
        backtester = SimpleBacktester(strategy)

        _, _, unrestricted = backtester.optimize(
            {"fast": range(2, 4), "slow": range(5, 7)},
            constraint=lambda p: p["fast"] < p["slow"],
        )
        _, _, restricted = backtester.optimize(
            {"fast": range(2, 4), "slow": range(5, 7)},
            constraint=lambda p: p["fast"] < p["slow"],
            risk_tolerance={"max_drawdown": 0.0},
        )

        assert len(restricted) <= len(unrestricted)
        if not restricted.empty:
            assert (restricted["max_drawdown"] <= 0.0).all()

    def test_optimize_best_report_is_consistent_with_best_params(self, datasource):
        """The selected best row should match the returned report exactly."""
        strategy = ParametrizedHoldPeriodStrategy()
        strategy.add_data(datasource, "EURUSD")
        backtester = SimpleBacktester(strategy)

        best_params, best_report, results_df = backtester.optimize(
            {"hold_period": range(1, 11)}
        )

        assert isinstance(best_report, BacktestReport)
        assert not results_df.empty

        matching_rows = results_df.loc[
            results_df["hold_period"] == best_params["hold_period"]
        ]
        assert len(matching_rows) == 1

        best_row = matching_rows.iloc[0]

        assert best_report.final_cash == pytest.approx(best_row["final_cash"])
        assert best_report.total_return == pytest.approx(best_row["total_return"])

    def test_optimize_does_not_mutate_base_strategy_indicators(self, datasource):
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

    def test_optimize_after_run_uses_same_backtester_cleanly(self, datasource):
        """A run followed by optimize should still agree with a clean replay."""
        strategy = IndicatorResetStrategy()
        strategy.add_data(datasource, "EURUSD")
        backtester = SimpleBacktester(strategy)

        backtester.run(progress_bar=False)
        best_params, best_report, results_df = backtester.optimize(
            {"slow": range(4, 7), "fast": range(2, 4)},
            constraint=lambda x: x["slow"] > x["fast"],
        )

        assert best_report is None
        assert results_df.empty

    def test_repeat_run_on_same_backtester_is_deterministic(self, datasource):
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
        assert np.allclose(report1.PnlRecord.to_numpy(), report2.PnlRecord.to_numpy())
        assert len(report1.orders) == len(report2.orders)

    def test_max_drawdown(self):
        """Max drawdown should report the peak-to-trough decline as a positive number."""
        equity = pd.Series([100, 110, 105, 95, 100])
        mdd = max_drawdown(equity)
        expected_mdd = (95 - 110) / 110
        assert abs(mdd - abs(expected_mdd)) < 1e-6

    def test_infer_periods_per_year(self):
        """The period inference helper should distinguish daily and empty indexes."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        periods = _infer_periods_per_year(dates)
        assert periods == 252

        periods = _infer_periods_per_year(pd.DatetimeIndex([]))
        assert periods == 252 * 24 * 60

    def test_periods_per_year_property(self, backtester):
        """Test periods_per_year property of BacktestReport."""
        report = backtester.run(progress_bar=False)
        periods = report.periods_per_year
        assert isinstance(periods, int)
        assert periods > 0
