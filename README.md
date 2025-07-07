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
4. [Running Tests](#running-tests)
5. [Development](#development)
6. [Contributing](#contributing)

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
        ├── __init__.py
        ├── test_sources. # Tests for sources.py
        ├── test_strategy.py # Tests for strategy.py
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
$ git clone https://github.com/dangreen07/quantex.git
$ cd quantex

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
