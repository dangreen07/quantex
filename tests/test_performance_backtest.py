import pathlib

import pytest

from quantex.backtest import BacktestRunner
from quantex.strategy import Strategy
from quantex.sources import ParquetDataSource


class PassiveStrategy(Strategy):
    """A do-nothing strategy used solely for performance benchmarking."""

    def run(self):  # noqa: D401 – simple stub, no docstring needed beyond class
        # Intentionally perform no trading logic – we only want engine overhead
        pass


data_root = pathlib.Path(__file__).resolve().parents[2] / "data"
BTC_PARQUET = data_root / "bitcoin_minute_bar_data_2022-2025-03.parquet"


@pytest.mark.benchmark(group="backtest")
def test_backtest_engine_speed(benchmark):
    """Benchmark the BacktestRunner over a large historical dataset.

    The benchmark prints execution time and ops/sec; it will not fail unless
    the engine raises. We can later add performance thresholds once we have
    baseline numbers.
    """

    ds = ParquetDataSource(path=BTC_PARQUET, symbol="BTC-USD")

    # Strategy and DataSource mappings use the same instance; this mirrors
    # real-world usage while avoiding unnecessary copies.
    strategy = PassiveStrategy(data_sources={"btc": ds})

    runner = BacktestRunner(strategy=strategy, data_sources={"btc": ds})

    # The benchmark fixture runs the function multiple times and reports stats
    result = benchmark(runner.run)

    # Record sizes for informational purposes (do not fail benchmark)
    benchmark.extra_info["nav_len"] = len(result.nav)
    benchmark.extra_info["bars_len"] = len(ds)
