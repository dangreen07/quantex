from __future__ import annotations

import pandas as pd

from quantex.strategy import Strategy


class DeterministicEntryExitStrategy(Strategy):
    """Buys on the second bar and closes on the penultimate bar."""

    def init(self):
        pass

    def next(self):
        if len(self.data["EURUSD"].Close) == 2:
            self.positions["EURUSD"].buy(quantity=1.0)
        elif len(self.data["EURUSD"].Close) == len(self.data["EURUSD"].data) - 1:
            self.positions["EURUSD"].close()


class ParametrizedHoldPeriodStrategy(Strategy):
    """Strategy used to verify optimize() returns self-consistent results."""

    hold_period = 2

    def init(self):
        self._entry_index = None

    def next(self):
        current_length = len(self.data["EURUSD"].Close)
        current_index = current_length - 1

        if current_length == 2 and self._entry_index is None:
            self.positions["EURUSD"].buy(quantity=1.0)
            self._entry_index = current_index
            return

        if self._entry_index is not None and current_index - self._entry_index >= self.hold_period:
            self.positions["EURUSD"].close()
            self._entry_index = None


class IndicatorResetStrategy(Strategy):
    """Strategy that recreates indicators in init to catch optimize() state leakage."""

    slow = 5
    fast = 2

    def init(self):
        self.sma_slow = self.Indicator(
            pd.Series(self.data["EURUSD"].Close).rolling(self.slow).mean().to_numpy()
        )
        self.sma_fast = self.Indicator(
            pd.Series(self.data["EURUSD"].Close).rolling(self.fast).mean().to_numpy()
        )

    def next(self):
        if len(self.data["EURUSD"].Close) < self.slow:
            return

        if self.sma_fast[-1] > self.sma_slow[-1] and self.sma_fast[-2] <= self.sma_slow[-2]:
            self.positions["EURUSD"].buy(0.9)
        elif self.sma_fast[-1] < self.sma_slow[-1] and self.sma_fast[-2] >= self.sma_slow[-2]:
            self.positions["EURUSD"].sell(0.9)


class RepeatableRunStrategy(Strategy):
    """Strategy designed to prove repeated runs stay deterministic."""

    slow = 5
    fast = 2

    def init(self):
        close = pd.Series(self.data["EURUSD"].Close)
        self.sma_slow = self.Indicator(close.rolling(self.slow).mean().to_numpy())
        self.sma_fast = self.Indicator(close.rolling(self.fast).mean().to_numpy())

    def next(self):
        if len(self.data["EURUSD"].Close) < self.slow:
            return

        if self.sma_fast[-1] > self.sma_slow[-1] and self.sma_fast[-2] <= self.sma_slow[-2]:
            self.positions["EURUSD"].buy(0.9)
        elif self.sma_fast[-1] < self.sma_slow[-1] and self.sma_fast[-2] >= self.sma_slow[-2]:
            self.positions["EURUSD"].sell(0.9)


class RiskAwareStrategy(Strategy):
    """Strategy designed to test objective selection and risk filtering."""

    fast = 2
    slow = 5

    def init(self):
        close = pd.Series(self.data["EURUSD"].Close)
        self.fast_ma = self.Indicator(close.rolling(self.fast).mean().to_numpy())
        self.slow_ma = self.Indicator(close.rolling(self.slow).mean().to_numpy())

    def next(self):
        if len(self.data["EURUSD"].Close) < self.slow:
            return

        if self.fast_ma[-1] > self.slow_ma[-1] and self.fast_ma[-2] <= self.slow_ma[-2]:
            self.positions["EURUSD"].buy(1.0)
        elif self.fast_ma[-1] < self.slow_ma[-1] and self.fast_ma[-2] >= self.slow_ma[-2]:
            self.positions["EURUSD"].close()
