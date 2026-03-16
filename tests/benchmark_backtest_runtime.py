import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantex import DataSource, SimpleBacktester, Strategy


class BenchmarkStrategy(Strategy):
    def __init__(self, stop_loss_pct: float = 0.01, take_profit_pct: float = 0.015):
        super().__init__()
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        rows = 100_000
        index = pd.date_range("2020-01-01", periods=rows, freq="min")

        base = 100 + np.linspace(0, 50, rows)
        wave = 2.5 * np.sin(np.arange(rows) / 45.0)
        pulse = 1.5 * np.sign(np.sin(np.arange(rows) / 120.0))
        close = base + wave + pulse
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) + 0.4
        low = np.minimum(open_, close) - 0.4
        volume = np.full(rows, 1_000.0)

        data = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=index,
        )
        self.add_data(DataSource(data), "TEST")

    def init(self):
        close_series = pd.Series(self.data["TEST"].Close)
        self.fast = self.Indicator(close_series.rolling(window=10).mean().to_numpy())
        self.slow = self.Indicator(close_series.rolling(window=40).mean().to_numpy())

    def next(self):
        if len(self.fast) < 41 or len(self.slow) < 41:
            return

        broker = self.positions["TEST"]
        current_open = np.float64(self.data["TEST"].COpen)

        crossed_up = self.fast[-2] <= self.slow[-2] and self.fast[-1] > self.slow[-1]
        crossed_down = self.fast[-2] >= self.slow[-2] and self.fast[-1] < self.slow[-1]

        if crossed_up and broker.is_closed():
            broker.buy(
                quantity=0.95,
                stop_loss=current_open * (1 - self.stop_loss_pct),
                take_profit=current_open * (1 + self.take_profit_pct),
            )
        elif crossed_down and broker.is_long():
            broker.close()
        elif crossed_down and broker.is_closed():
            broker.sell(
                quantity=0.95,
                stop_loss=current_open * (1 + self.stop_loss_pct),
                take_profit=current_open * (1 - self.take_profit_pct),
            )
        elif crossed_up and broker.is_short():
            broker.close()


def main():
    strategy = BenchmarkStrategy()
    backtester = SimpleBacktester(strategy, cash=100_000)

    started = time.perf_counter()
    report = backtester.run(progress_bar=False)
    elapsed = time.perf_counter() - started

    print(f"Rows: {len(report.PnlRecord):,}")
    print(f"Elapsed seconds: {elapsed:.6f}")
    print(f"Rows/sec: {len(report.PnlRecord) / elapsed:,.2f}")
    print(f"Orders: {len(report.orders):,}")
    print(f"Final cash: {report.final_cash:,.2f}")
    print(f"Total return: {report.total_return:.4%}")


if __name__ == "__main__":
    main()
