# Quantex

A simple quant strategy creation and back-testing package written in Python.  
Quantex aims to provide a lightweight foundation for building trading
strategies, ingesting historical market data, and evaluating performance –
all without the heavy overhead of larger, more opinionated quant libraries.

---

## Table of Contents
1. [Features](#features)
2. [Project Layout](#project-layout)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Running Tests](#running-tests)
6. [Development](#development)
7. [Contributing](#contributing)
8. [License](#license)

---

## Features
* **Data Abstraction** – A generic `DataSource` interface that you can
  subclass to plug in CSVs, Parquet files, live feeds, databases, etc.
* **Back-testing Support** – A `BacktestingDataSource` base class to drive
  offline simulations.
* **Strategy Skeleton** – A minimal `Strategy` class that you can extend
  to implement trading logic and track `Trade` / `Positions` objects.
* **Python 3.13+** – Takes advantage of the latest language features.
* **Poetry-managed** – Modern dependency management, packaging, and
  virtual-environment handling.

> **Note:** The public API is still under heavy development and may change
> until v1.0. Feedback is welcome!

---

## Project Layout
```
Quant-Trading/
└── quantex/             # Python package root (this repo)
    ├── src/quantex/     # Source code
    │   ├── __init__.py
    │   ├── sources.py  # DataSource abstractions
    │   └── strategy.py # Strategy abstractions
    ├── tests/          # Pytest test-suite
    ├── poetry.lock     # Locked dependency versions
    ├── pyproject.toml  # Poetry & build metadata
    └── README.md       # ← you are here
```

---

## Installation
Quantex is managed with [Poetry](https://python-poetry.org/). Clone the
repository and install the dependencies in an isolated virtual environment:

```bash
# Clone the repo
$ git clone https://github.com/<youruser>/Quant-Trading.git
$ cd Quant-Trading/quantex

# Install dependencies
$ poetry install
```

This command will:
1. Create / activate a local virtual-env (unless Poetry is configured to use a
   global env).
2. Install package dependencies from `pyproject.toml`.
3. Install Quantex itself in *editable* mode, so changes you make in `src/` are
   reflected immediately.

---

## Quick Start
Below is a minimal example that shows how to extend `DataSource` and build a
very simple strategy skeleton:

```python
from pathlib import Path
import pandas as pd
from quantex.sources import DataSource, BacktestingDataSource
from quantex.strategy import Strategy

# 1) Implement a data source backed by a Parquet file
class ParquetDataSource(BacktestingDataSource):
    def __init__(self, path: str | Path):
        self.df = pd.read_parquet(path).reset_index(drop=True)

    def get_row(self, idx: int):
        return self.df.loc[idx]

    def get_data_before_idx(self, idx: int, lookback_period: int):
        start = max(0, idx - lookback_period)
        return self.df.iloc[start: idx + 1]

    def __len__(self):
        return len(self.df)

# 2) Implement a trivial strategy – buy on every 10th bar
class DemoStrategy(Strategy):
    def on_bar(self, bar, idx):
        if idx % 10 == 0:
            self.buy(price=bar["close"], size=1)

if __name__ == "__main__":
    source = ParquetDataSource("../data/coinbase_minute_bar_data_2022-2025-03.parquet")
    strategy = DemoStrategy(data_source=source)
    strategy.run()
    print(strategy.performance_summary())
```

---

## Running Tests
Quantex uses [pytest](https://docs.pytest.org/) for its test suite. After
installing the dev dependencies (`poetry install` above), simply run:

```bash
poetry run pytest
```

You should see tests collect and run successfully. Ensure your editor uses the
Poetry virtual-env so import paths resolve correctly.

---

## Development
1. Create a new branch: `git checkout -b feature/<name>`
2. Write your code & tests.
3. Ensure `poetry run pytest` passes and `ruff` / `mypy` (coming soon) are
   clean.
4. Open a pull request describing the change.

---

## Contributing
Contributions, bug reports, and feature requests are welcome! Please open an
issue to discuss what you'd like to work on or submit a pull request directly.
We follow the "fork → feature branch → pull request" workflow. By
contributing you agree to license your work under the same terms as Quantex.

---

## License
This project is licensed under the MIT License — see the `LICENSE` file for
full details.
