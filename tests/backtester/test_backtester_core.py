import numpy as np
import pandas as pd

from quantex.backtester import BacktestReport, CommissionType


def test_init(backtester):
    """Test SimpleBacktester initialization."""
    assert backtester.cash == 10000
    assert backtester.commission == 0.002
    assert backtester.commission_type == CommissionType.PERCENTAGE
    assert backtester.lot_size == 1
    assert isinstance(backtester.PnLRecord, np.ndarray)


def test_run(backtester):
    """A normal run should produce a full report with one equity series."""
    report = backtester.run(progress_bar=False)

    assert isinstance(report, BacktestReport)
    assert isinstance(report.starting_cash, np.float64)
    assert isinstance(report.final_cash, np.float64)
    assert isinstance(report.PnlRecord, pd.Series)
    assert isinstance(report.orders, list)

    assert len(report.PnlRecord) == len(
        backtester.strategy.data["EURUSD"].data
    )


def test_backtest_report_str(backtester):
    """String formatting should surface the key performance summary fields."""
    report = backtester.run(progress_bar=False)
    report_str = str(report)

    assert "Starting Cash" in report_str
    assert "Final Cash" in report_str
    assert "Total Return" in report_str
    assert "Sharpe Ratio" in report_str
    assert "Max Drawdown" in report_str
    assert "Total Trades" in report_str


def test_periods_per_year_property(backtester):
    """Test periods_per_year property of BacktestReport."""
    report = backtester.run(progress_bar=False)
    periods = report.periods_per_year
    assert isinstance(periods, int)
    assert periods > 0