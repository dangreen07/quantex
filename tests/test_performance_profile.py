import io
import cProfile
import pstats

from quantex.sources import RandomWalkDataSource
from quantex.strategy import Strategy
from quantex.backtest import BacktestRunner


class NoOpStrategy(Strategy):
    """Strategy that deliberately does *nothing* – useful for engine profiling."""

    def run(self):
        pass


def test_random_walk_backtest_profile():
    """Run a short synthetic back-test under *cProfile* and ensure it completes."""

    ds = RandomWalkDataSource(symbol="RND", periods=5_000_000, seed=42)
    strategy = NoOpStrategy({"rw": ds}, initial_cash=100_000)

    runner = BacktestRunner(strategy, {"rw": ds})

    pr = cProfile.Profile()
    pr.enable()
    runner.run()
    pr.disable()

    pr.dump_stats("random_walk_profile.prof")

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats()
    output = s.getvalue()

    # Basic sanity checks
    assert "cumtime" in output
    # NAV should have an entry for every bar processed
    assert len(runner.event_bus.nav) == len(ds)
