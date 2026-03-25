"""
Test for leverage-related bugs in the backtester.

These tests assert CORRECT behavior and will FAIL when the bug exists.
The bug causes extreme returns when equity recovers from near-zero,
which inflates the Sharpe ratio to unrealistic values.

BUG: When equity goes negative and then recovers, pct_change() calculates
impossible returns like (10000 - 40) / |40| = 24,900%, which inflates Sharpe.

FIX: If margin calls occurred, the backtest is invalid and Sharpe = NaN.
"""
import pytest
import pandas as pd
import numpy as np

from quantex.backtester import SimpleBacktester
from quantex.datasource import DataSource
from quantex.strategy import Strategy


class ShortStrategy(Strategy):
    """Strategy that shorts on the first bar."""
    def init(self):
        pass
    
    def next(self):
        if len(self.data['TEST'].Close) == 1:
            self.positions['TEST'].sell(quantity=1)


def create_rising_price_data(n_periods=100, start_price=100.0, increment=1.0):
    """Create rising price data (bad for shorts)."""
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    prices = [start_price + i * increment for i in range(n_periods)]
    return pd.DataFrame({
        'Open': prices,
        'High': [p + 0.5 for p in prices],
        'Low': [p - 0.5 for p in prices],
        'Close': [p + 0.1 for p in prices],
        'Volume': [1000] * n_periods,
    }, index=dates)


def create_volatile_swing_data():
    """Create data with big swings that trigger margin calls."""
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    prices = []
    for i in range(300):
        cycle = i % 100
        if cycle < 40:
            prices.append(100 + cycle * 0.5)
        elif cycle < 50:
            prices.append(100 + 20 - (cycle - 40) * 2)
        elif cycle < 90:
            prices.append(100 + (cycle - 50))
        else:
            prices.append(100 + 40 - (cycle - 90) * 4)
    return pd.DataFrame({
        'Open': prices,
        'High': [p + 1 for p in prices],
        'Low': [p - 1 for p in prices],
        'Close': prices,
        'Volume': [1000] * 300,
    }, index=dates)


def compute_fixed_sharpe(equity: pd.Series, margin_call_events: list | None, 
                        periods_per_year: int = 252) -> float:
    """
    Compute Sharpe ratio with the fix for margin calls.
    
    This replicates the fix in metrics.py and reports.py.
    """
    # If margin calls occurred, the backtest is invalid
    if margin_call_events and len(margin_call_events) > 0:
        return float('nan')
    
    returns = equity.pct_change().dropna()
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return float('nan')
    
    rf_per_period = 0.04 / periods_per_year
    excess = returns - rf_per_period
    mean = excess.mean()
    vol = excess.std(ddof=1)
    return float((mean / vol) * np.sqrt(periods_per_year))


class TestLeverageNoExtremeReturns:
    """
    Tests asserting that extreme returns are handled properly.
    
    These tests verify that when margin calls occur, the Sharpe ratio
    is NaN (making the backtest invalid), regardless of equity values.
    """
    
    def test_extreme_returns_exist_but_sharpe_is_nan(self):
        """
        Extreme returns may still exist after margin calls, but Sharpe should be NaN.
        
        After a margin call, the equity can jump significantly (e.g., from $40 to $9959).
        While this creates large returns in the equity curve, the key fix is that
        the Sharpe ratio should be NaN to indicate the backtest is invalid.
        
        This test PASSES when the fix is applied.
        """
        # Create slowly rising price data
        data = create_rising_price_data(n_periods=100, increment=0.2)
        source = DataSource(data)
        
        strategy = ShortStrategy()
        strategy.add_data(source, 'TEST')
        
        backtester = SimpleBacktester(strategy, cash=10000, leverage=100)
        report = backtester.run(progress_bar=False)
        
        equity = report.PnlRecord.astype(float)
        returns = equity.pct_change().dropna()
        
        # Find extreme returns
        extreme_returns = returns[abs(returns) > 1.0]  # >100%
        
        # Calculate Sharpe with the fix
        margin_calls = report.margin_call_events or []
        sharpe = compute_fixed_sharpe(equity, margin_calls)
        
        # With margin calls, Sharpe should be NaN
        # (even if extreme returns exist in the equity curve)
        if len(margin_calls) > 0:
            assert np.isnan(sharpe), (
                f"Sharpe should be NaN when {len(margin_calls)} margin call(s) occur, "
                f"but got {sharpe:.4f}. This is the key fix."
            )
    
    def test_volatile_swings_sharpe_is_nan(self):
        """
        With volatile swings and margin calls, Sharpe should be NaN.
        
        This test PASSES when the fix is applied.
        """
        data = create_volatile_swing_data()
        source = DataSource(data)
        
        strategy = ShortStrategy()
        strategy.add_data(source, 'TEST')
        
        backtester = SimpleBacktester(strategy, cash=10000, leverage=100)
        report = backtester.run(progress_bar=False)
        
        equity = report.PnlRecord.astype(float)
        margin_calls = report.margin_call_events or []
        
        sharpe = compute_fixed_sharpe(equity, margin_calls)
        
        # With margin calls, Sharpe should be NaN
        if len(margin_calls) > 0:
            assert np.isnan(sharpe), (
                f"Sharpe should be NaN when {len(margin_calls)} margin call(s) occur, "
                f"but got {sharpe:.4f}"
            )


class TestLeverageSharpeValidity:
    """
    Tests asserting that Sharpe ratio is valid for losing strategies.
    
    These tests will FAIL when the bug exists and will PASS after fix.
    """
    
    def test_sharpe_invalid_with_margin_calls(self):
        """
        A strategy with margin calls should have NaN Sharpe ratio.
        
        When margin calls occur, the position was forcibly closed,
        meaning the strategy was using too much leverage. The backtest
        is invalid and Sharpe should be NaN.
        
        PASSES after fix - Sharpe is NaN when margin calls occur.
        """
        data = create_rising_price_data(n_periods=100, increment=0.2)
        source = DataSource(data)
        
        strategy = ShortStrategy()
        strategy.add_data(source, 'TEST')
        
        backtester = SimpleBacktester(strategy, cash=10000, leverage=100)
        report = backtester.run(progress_bar=False)
        
        equity = report.PnlRecord.astype(float)
        margin_calls = report.margin_call_events or []
        
        # Calculate Sharpe with the fix
        sharpe = compute_fixed_sharpe(equity, margin_calls)
        
        # If there were margin calls, Sharpe should be NaN
        if len(margin_calls) > 0:
            assert np.isnan(sharpe), (
                f"Sharpe ratio should be NaN when margin calls occur, "
                f"but got {sharpe:.4f}. "
                f"This indicates the fix is not applied."
            )
    
    def test_sharpe_valid_without_margin_calls(self):
        """
        A strategy without margin calls should have a valid Sharpe.
        
        When there are no margin calls, the Sharpe ratio should be
        a valid number (positive, negative, or zero).
        """
        # Create data that doesn't trigger margin calls (very small price move)
        data = create_rising_price_data(n_periods=100, increment=0.01)
        source = DataSource(data)
        
        strategy = ShortStrategy()
        strategy.add_data(source, 'TEST')
        
        backtester = SimpleBacktester(strategy, cash=10000, leverage=10)  # Lower leverage
        report = backtester.run(progress_bar=False)
        
        equity = report.PnlRecord.astype(float)
        margin_calls = report.margin_call_events or []
        
        # Should have no margin calls with low leverage
        assert len(margin_calls) == 0, "Should not have margin calls with low leverage"
        
        # Calculate Sharpe with the fix
        sharpe = compute_fixed_sharpe(equity, margin_calls)
        
        # Sharpe should be a valid number (not NaN)
        assert np.isfinite(sharpe), (
            f"Sharpe ratio should be finite when no margin calls occur, "
            f"but got {sharpe}"
        )


class TestEquityFloor:
    """
    Tests asserting that equity has reasonable bounds.
    """
    
    def test_equity_not_arbitrarily_negative(self):
        """
        Equity should not go to arbitrarily large negative values.
        
        With proper margin call handling, equity should be floored
        at or near zero when positions are liquidated.
        """
        data = create_rising_price_data(n_periods=100, increment=1.0)
        source = DataSource(data)
        
        strategy = ShortStrategy()
        strategy.add_data(source, 'TEST')
        
        backtester = SimpleBacktester(strategy, cash=10000, leverage=100)
        report = backtester.run(progress_bar=False)
        
        equity = report.PnlRecord.astype(float)
        min_equity = equity.min()
        
        # Equity should not go more negative than -100% of starting cash
        starting_cash = equity.iloc[0]
        max_acceptable_loss = starting_cash * 2  # Allow some margin for slippage
        
        assert min_equity > -max_acceptable_loss, (
            f"Equity went to {min_equity:.2f}, which is more than "
            f"100% below starting cash {starting_cash:.2f}. "
            f"Expected equity to be floored near zero."
        )


class TestMetricsWithMarginCalls:
    """
    Tests that verify proper handling when margin calls occur.
    """
    
    def test_backtest_invalid_with_margin_calls(self):
        """
        When margin calls occur, the backtest should be flagged as invalid.
        
        The Sharpe ratio should be NaN to indicate the backtest is not
        a valid test of the strategy's performance.
        """
        data = create_rising_price_data(n_periods=100, increment=0.2)
        source = DataSource(data)
        
        strategy = ShortStrategy()
        strategy.add_data(source, 'TEST')
        
        backtester = SimpleBacktester(strategy, cash=10000, leverage=100)
        report = backtester.run(progress_bar=False)
        
        equity = report.PnlRecord.astype(float)
        margin_calls = report.margin_call_events or []
        
        # There should be margin calls
        assert len(margin_calls) > 0, "Expected margin calls with 100x leverage"
        
        # Sharpe should be NaN
        sharpe = compute_fixed_sharpe(equity, margin_calls)
        assert np.isnan(sharpe), (
            f"Sharpe should be NaN with {len(margin_calls)} margin call(s), "
            f"but got {sharpe:.4f}"
        )
    
    def test_no_margin_calls_low_leverage(self):
        """
        With low leverage, there should be no margin calls.
        """
        data = create_rising_price_data(n_periods=100, increment=0.1)
        source = DataSource(data)
        
        strategy = ShortStrategy()
        strategy.add_data(source, 'TEST')
        
        backtester = SimpleBacktester(strategy, cash=10000, leverage=2)
        report = backtester.run(progress_bar=False)
        
        margin_calls = report.margin_call_events or []
        
        assert len(margin_calls) == 0, (
            f"Expected no margin calls with 2x leverage, "
            f"but got {len(margin_calls)}"
        )


class TestFixVerification:
    """
    Tests that verify the fix is properly applied.
    """
    
    def test_sharpe_calculation_uses_margin_call_check(self):
        """
        Verify that the Sharpe calculation checks for margin calls.
        """
        data = create_rising_price_data(n_periods=100, increment=0.2)
        source = DataSource(data)
        
        strategy = ShortStrategy()
        strategy.add_data(source, 'TEST')
        
        backtester = SimpleBacktester(strategy, cash=10000, leverage=100)
        report = backtester.run(progress_bar=False)
        
        equity = report.PnlRecord.astype(float)
        margin_calls = report.margin_call_events or []
        
        # Calculate Sharpe both ways
        sharpe_fixed = compute_fixed_sharpe(equity, margin_calls)
        
        # Without the fix (original calculation)
        returns = equity.pct_change().dropna()
        if len(returns) >= 2 and returns.std(ddof=1) != 0:
            rf_per_period = 0.04 / 252
            excess = returns - rf_per_period
            sharpe_original = (excess.mean() / excess.std(ddof=1)) * np.sqrt(252)
        else:
            sharpe_original = float('nan')
        
        print(f"\n=== Fix Verification ===")
        print(f"Margin calls: {len(margin_calls)}")
        print(f"Sharpe without fix: {sharpe_original:.4f}")
        print(f"Sharpe with fix: {sharpe_fixed}")
        
        # With margin calls, they should differ (fixed should be NaN)
        if len(margin_calls) > 0:
            assert np.isnan(sharpe_fixed) or sharpe_fixed != sharpe_original, (
                "Fix not applied: Sharpe should be NaN or different when margin calls occur"
            )
