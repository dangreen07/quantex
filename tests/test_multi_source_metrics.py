import pandas as pd
import pytest

from quantex.backtest import BacktestRunner
from quantex.strategy import Strategy
from quantex.sources import BacktestingDataSource
from quantex.models import Bar


class DFDataSource(BacktestingDataSource):
    """BacktestingDataSource backed by an in-memory pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, symbol: str):
        self._df = df
        self.symbol = symbol
        self.index = 0

    def __len__(self):
        return len(self._df)

    def get_raw_data(self):
        return self._df

    def peek_timestamp(self):
        if self.index < len(self):
            return self._df.index[self.index]
        return None

    # --- DataSource API -------------------------------------------------
    def get_current_bar(self):
        row = self._df.iloc[self.index]
        ts = self._df.index[self.index]
        return Bar(ts, row.open, row.high, row.low, row.close, row.volume, self.symbol)

    def get_lookback_data(self, lookback_period: int):
        start = max(0, self.index - lookback_period + 1)
        return self._df.iloc[start : self.index + 1]


class BuyAndHoldBoth(Strategy):
    """Buy one unit of each symbol on the first bar and hold."""

    def run(self):
        if self.index == 0:
            for sym in self.symbols:
                self.buy(sym, 1)


class BuyAndHoldIncFlat(Strategy):
    """Buy one unit of each symbol; one appreciates, one flat."""

    def run(self):
        if self.index == 0:
            for sym in self.symbols:
                self.buy(sym, 1)


def _make_ohlcv(prices: list[float]) -> pd.DataFrame:
    """Helper to make OHLCV frame where all OHLC are the same."""
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1000] * len(prices),
        }
    )


def test_multi_source_metrics():
    """Ensure NAV and metrics are computed correctly with 2 symbols / sources."""

    # --- create synthetic OHLCV for 10 minutes ---
    idx = pd.date_range(start="2022-01-01 00:00", periods=10, freq="min", tz="UTC")
    template = {
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000,
    }
    df_btc = pd.DataFrame(template, index=idx)
    df_coin = pd.DataFrame(
        {k: v * 0.5 if k != "volume" else v for k, v in template.items()}, index=idx
    )

    ds_btc = DFDataSource(df_btc, symbol="BTC")
    ds_coin = DFDataSource(df_coin, symbol="COIN")

    strat = BuyAndHoldBoth(symbols=["BTC", "COIN"], initial_cash=100_000)
    runner = BacktestRunner(strat, {"btc": ds_btc, "coin": ds_coin})
    result = runner.run()

    # Metric should exist
    assert "total_return" in result.metrics
    assert result.metrics["total_return"] == pytest.approx(0.0, abs=1e-6)

    # NAV length matches timeline
    assert len(result.nav) == len(df_btc)

    # Two buy orders
    assert len(result.orders) == 2
    assert {o.symbol for o in result.orders} == {"BTC", "COIN"}


def test_multi_source_positive_return():
    """Strategy over two symbols where one rises in price should yield >0 return."""

    idx = pd.date_range("2022-01-01 00:00", periods=10, freq="min", tz="UTC")

    # Inc symbol rises $1 per minute starting at 100
    inc_prices = [float(100 + i) for i in range(len(idx))]
    flat_prices = [50.0] * len(idx)

    df_inc = _make_ohlcv(inc_prices).set_index(idx)
    df_flat = _make_ohlcv(flat_prices).set_index(idx)

    ds_inc = DFDataSource(df_inc, symbol="INC")
    ds_flat = DFDataSource(df_flat, symbol="FLT")

    strat = BuyAndHoldIncFlat(symbols=["INC", "FLT"], initial_cash=100_000)
    result = BacktestRunner(strat, {"inc": ds_inc, "flt": ds_flat}).run()

    assert result.metrics["total_return"] > 0

    # Orders are executed on the **next** bar's open, so the buy happens at
    # 101 (not 100). Profit = 109 − 101.
    expected = (109 - 101) / 100_000
    assert pytest.approx(result.metrics["total_return"], abs=1e-6) == expected
