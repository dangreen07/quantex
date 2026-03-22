import numpy as np
import pandas as pd
from typing import cast

from quantex import indicators
from quantex.backtester import SimpleBacktester
from quantex.datasource import DataSource
from quantex.strategy import Strategy


class StaticIndicatorStrategy(Strategy):
    def init(self):
        values = np.array([np.nan, np.nan, 3.0, 4.0, 5.0, 6.0], dtype=float)
        self.test_indicator = self.Indicator(values)
        self.seen_values = []

    def next(self):
        if len(self.data["EURUSD"].Close) >= 3:
            self.seen_values.append(float(self.test_indicator[-1]))


class InitDataVisibilityStrategy(Strategy):
    def init(self):
        self.init_close_values = list(self.data["EURUSD"].Close)

    def next(self):
        pass


class BuiltinIndicatorStrategy(Strategy):
    def init(self):
        close = self.data["EURUSD"].Close
        self.sma_3 = self.Indicator(self.ta.sma(close, 3))
        self.rsi_3 = self.Indicator(self.ta.rsi(close, 3))
        macd_line, macd_signal, macd_hist = self.ta.macd(close, 3, 5, 2)
        self.macd_line = self.Indicator(macd_line)
        self.macd_signal = self.Indicator(macd_signal)
        self.macd_hist = self.Indicator(macd_hist)
        self.last_snapshot = None

    def next(self):
        if len(self.sma_3) >= 1:
            self.last_snapshot = {
                "sma": float(self.sma_3[-1]) if not np.isnan(self.sma_3[-1]) else np.nan,
                "rsi": float(self.rsi_3[-1]) if not np.isnan(self.rsi_3[-1]) else np.nan,
                "macd": float(self.macd_line[-1]) if not np.isnan(self.macd_line[-1]) else np.nan,
            }


def test_indicator_values_progress_through_backtest_without_reset_corruption():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5, 6],
            "High": [1, 2, 3, 4, 5, 6],
            "Low": [1, 2, 3, 4, 5, 6],
            "Close": [1, 2, 3, 4, 5, 6],
            "Volume": [100] * 6,
        },
        index=dates,
    )

    original_strategy = StaticIndicatorStrategy()
    original_strategy.add_data(DataSource(df), "EURUSD")

    backtester = SimpleBacktester(original_strategy)
    backtester.run(progress_bar=False)

    strategy = cast(StaticIndicatorStrategy, backtester.strategy)
    assert strategy.seen_values == [3.0, 4.0, 5.0, 6.0]


def test_init_has_access_to_full_datasource_history():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Open": [10, 11, 12, 13],
            "High": [10, 11, 12, 13],
            "Low": [10, 11, 12, 13],
            "Close": [10, 11, 12, 13],
            "Volume": [100] * 4,
        },
        index=dates,
    )

    original_strategy = InitDataVisibilityStrategy()
    original_strategy.add_data(DataSource(df), "EURUSD")

    backtester = SimpleBacktester(original_strategy)
    backtester.run(progress_bar=False)

    strategy = cast(InitDataVisibilityStrategy, backtester.strategy)
    assert strategy.init_close_values == [10, 11, 12, 13]


def test_builtin_indicator_catalog_matches_known_values():
    close = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    high = close + 1.0
    low = close - 1.0
    volume = np.array([100, 110, 120, 130, 140, 150], dtype=float)

    sma_3 = indicators.sma(close, 3)
    ema_3 = indicators.ema(close, 3)
    rsi_3 = indicators.rsi(close, 3)
    lower, middle, upper = indicators.bollinger_bands(close, 3)
    macd_line, macd_signal, macd_hist = indicators.macd(close, 3, 5, 2)
    atr_3 = indicators.atr(high, low, close, 3)
    obv_values = indicators.obv(close, volume)
    vwap_values = indicators.vwap(high, low, close, volume)
    adx_values, plus_di, minus_di = indicators.adx(high, low, close, 3)

    np.testing.assert_allclose(sma_3, np.array([np.nan, np.nan, 11.0, 12.0, 13.0, 14.0]), equal_nan=True)
    np.testing.assert_allclose(ema_3, np.array([np.nan, np.nan, 11.0, 12.0, 13.0, 14.0]), equal_nan=True)
    np.testing.assert_allclose(rsi_3[3:], np.array([100.0, 100.0, 100.0]), atol=1e-8)
    np.testing.assert_allclose(middle, sma_3, equal_nan=True)
    np.testing.assert_allclose(lower[2:], np.array([9.36700684, 10.36700684, 11.36700684, 12.36700684]), atol=1e-8)
    np.testing.assert_allclose(upper[2:], np.array([12.63299316, 13.63299316, 14.63299316, 15.63299316]), atol=1e-8)
    np.testing.assert_allclose(atr_3[2:], np.array([2.0, 2.0, 2.0, 2.0]), atol=1e-8)
    np.testing.assert_allclose(obv_values, np.array([0.0, 110.0, 230.0, 360.0, 500.0, 650.0]))
    np.testing.assert_allclose(
        vwap_values,
        np.array([10.0, 10.52381, 11.060606, 11.608696, 12.166667, 12.733333]),
        atol=1e-6,
    )
    assert np.isfinite(macd_line[-1])
    assert np.isfinite(macd_signal[-1])
    assert np.isfinite(macd_hist[-1])
    assert np.isfinite(adx_values[-1])
    assert np.isfinite(plus_di[-1])
    assert np.isfinite(minus_di[-1])


def test_advanced_indicator_shapes_and_ranges():
    close = np.array([100.0, 102.0, 101.0, 104.0, 103.0, 107.0, 109.0, 108.0, 112.0, 115.0])
    high = close + np.array([1.5, 2.0, 1.0, 2.5, 1.5, 2.0, 1.2, 1.3, 1.1, 1.0])
    low = close - np.array([1.0, 1.2, 1.5, 1.3, 1.0, 1.1, 1.4, 1.0, 1.5, 1.2])
    volume = np.array([1000, 1050, 980, 1100, 1150, 1120, 1200, 1180, 1250, 1300], dtype=float)

    percent_k, percent_d = indicators.stochastic_oscillator(high, low, close, 5, 3)
    cci_values = indicators.cci(high, low, close, 5)
    williams = indicators.williams_r(high, low, close, 5)
    mfi_values = indicators.mfi(high, low, close, volume, 5)
    conversion, base, span_a, span_b, lagging = indicators.ichimoku_cloud(high, low, close, 3, 5, 7, 2)
    keltner_lower, keltner_middle, keltner_upper = indicators.keltner_channels(high, low, close, 3, 3, 1.5)
    donchian_lower, donchian_middle, donchian_upper = indicators.donchian_channels(high, low, 5)
    aroon_up, aroon_down, aroon_osc = indicators.aroon(high, low, 5)
    vi_plus, vi_minus = indicators.vortex(high, low, close, 5)
    ultimate = indicators.ultimate_oscillator(high, low, close, 3, 5, 7)
    trix_values = indicators.trix(close, 2)
    kama_values = indicators.kama(close, 3, 2, 5)
    zscore_values = indicators.zscore(close, 5)
    sharpe_values = indicators.sharpe_ratio(close, 4)
    sortino_values = indicators.sortino_ratio(close, 4)
    hurst_values = indicators.hurst_exponent(np.linspace(100.0, 120.0, 120), 100)
    fisher_values = indicators.fisher_transform(close, 5)
    slope_values = indicators.linear_regression_slope(close, 5)

    for series in [
        percent_k,
        percent_d,
        cci_values,
        williams,
        mfi_values,
        conversion,
        base,
        span_a,
        span_b,
        lagging,
        keltner_lower,
        keltner_middle,
        keltner_upper,
        donchian_lower,
        donchian_middle,
        donchian_upper,
        aroon_up,
        aroon_down,
        aroon_osc,
        vi_plus,
        vi_minus,
        ultimate,
        trix_values,
        kama_values,
        zscore_values,
        sharpe_values,
        sortino_values,
        fisher_values,
        slope_values,
    ]:
        assert len(series) == len(close)

    assert np.nanmax(percent_k) <= 100.0 + 1e-8
    assert np.nanmin(percent_k) >= -1e-8
    assert np.nanmax(williams) <= 0.0 + 1e-8
    assert np.nanmin(williams) >= -100.0 - 1e-8
    assert np.nanmax(aroon_up) <= 100.0 + 1e-8
    assert np.nanmax(aroon_down) <= 100.0 + 1e-8
    assert np.nanmax(mfi_values) <= 100.0 + 1e-8
    assert np.nanmin(mfi_values) >= -1e-8
    assert np.isnan(hurst_values[:99]).all()
    assert np.isfinite(hurst_values[-1]) or np.isnan(hurst_values[-1])


def test_strategy_can_use_builtin_indicator_catalog():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    df = pd.DataFrame(
        {
            "Open": [10, 11, 12, 13, 14, 15, 16, 17],
            "High": [11, 12, 13, 14, 15, 16, 17, 18],
            "Low": [9, 10, 11, 12, 13, 14, 15, 16],
            "Close": [10, 11, 12, 13, 14, 15, 16, 17],
            "Volume": [100] * 8,
        },
        index=dates,
    )

    original_strategy = BuiltinIndicatorStrategy()
    original_strategy.add_data(DataSource(df), "EURUSD")

    backtester = SimpleBacktester(original_strategy)
    backtester.run(progress_bar=False)

    strategy = cast(BuiltinIndicatorStrategy, backtester.strategy)
    assert strategy.last_snapshot is not None
    assert strategy.last_snapshot["sma"] == 16.0
    assert np.isfinite(strategy.last_snapshot["rsi"])
    assert np.isfinite(strategy.last_snapshot["macd"])
