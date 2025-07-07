import pytest
from quantex.strategy import Strategy
from quantex.sources import DataSource


class DummyDataSource(DataSource):
    """Minimal concrete DataSource for testing the abstract Strategy base class."""

    def get_row(self, idx):  # pragma: no cover
        return (idx, 0, 0, 0, 0, 0)

    def get_data_before_idx(self, idx, lookback_period):  # pragma: no cover
        return []


def test_strategy_stores_data_sources():
    ds = DummyDataSource()
    strat = Strategy({"dummy": ds})
    assert strat.data_sources["dummy"] is ds


def test_strategy_run_strategy_raises():
    ds = DummyDataSource()
    strat = Strategy({"dummy": ds})
    with pytest.raises(NotImplementedError):
        strat.run_strategy(0)
