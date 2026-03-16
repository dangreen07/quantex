import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantex import DataSource, SimpleBacktester, Strategy


class StopLossTakeProfitStressStrategy(Strategy):
    def __init__(self, rows: int = 100_000):
        super().__init__()
        self.rows = rows

        index = pd.date_range("2020-01-01", periods=rows, freq="min")
        cycle = np.arange(rows) % 4
        open_ = np.where(cycle < 2, 100.0, 101.0)
        close = open_.copy()
        high = open_ + 0.25
        low = open_ - 0.25
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
        self.enter_long_next = True

    def init(self):
        pass

    def next(self):
        broker = self.positions["TEST"]
        price = np.float64(self.data["TEST"].CClose)

        if not broker.is_closed():
            return

        if self.enter_long_next:
            broker.buy(
                quantity=0.95,
                stop_loss=np.float64(price - 0.10),
                take_profit=np.float64(price + 0.10),
            )
        else:
            broker.sell(
                quantity=0.95,
                stop_loss=np.float64(price + 0.10),
                take_profit=np.float64(price - 0.10),
            )

        self.enter_long_next = not self.enter_long_next


def main():
    strategy = StopLossTakeProfitStressStrategy(rows=100_000)
    backtester = SimpleBacktester(strategy, cash=100_000)

    started = time.perf_counter()
    report = backtester.run(progress_bar=False)
    elapsed = time.perf_counter() - started

    print("Benchmark: Stop-loss / take-profit stress")
    print(f"Rows: {len(report.PnlRecord):,}")
    print(f"Elapsed seconds: {elapsed:.6f}")
    print(f"Rows/sec: {len(report.PnlRecord) / elapsed:,.2f}")
    print(f"Orders: {len(report.orders):,}")
    print(f"Trades recorded: {len(report.tradeRecord):,}")
    print(f"Final cash: {report.final_cash:,.2f}")
    print(f"Total return: {report.total_return:.4%}")


if __name__ == "__main__":
    main()
