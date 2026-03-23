import pytest

from quantex.datasource import DataSource
from quantex.backtester import SimpleBacktester
from tests.strategies.common import DeterministicEntryExitStrategy


@pytest.fixture
def strategy(ohlcv_data):
    """Create a strategy instance backed by deterministic sample data."""
    datasource = DataSource(ohlcv_data)
    strat = DeterministicEntryExitStrategy()
    strat.add_data(datasource, "EURUSD")
    return strat


@pytest.fixture
def datasource(ohlcv_data):
    """Provide the same deterministic source used by strategy fixtures.

    Several optimizer tests need direct access to the source so they can
    create fresh strategies without depending on implicit fixture wiring.
    """
    return DataSource(ohlcv_data)


@pytest.fixture
def backtester(strategy):
    """Create a SimpleBacktester instance."""
    return SimpleBacktester(strategy)