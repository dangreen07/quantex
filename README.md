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
* **Strategy Skeleton** – Extendable `Strategy` base class for plug-and-play
  trading logic.
* **Core Data Models** – Immutable `Bar`, `Tick`, `Order`, `Fill`, plus
  stateful `Position` / `Portfolio` helpers for P&L accounting.
* **Black + Ruff Pre-commit** – `black` auto-formats and `ruff` lints every
  commit via *pre-commit* hooks, keeping the codebase consistent.
* **Python 3.13+** – Embraces the latest language features.
* **Poetry-managed** – Modern dependency management, packaging, and virtual
  environment handling.

> **Note:** The public API is still under heavy development and may change
> until v1.0. Feedback is welcome!

---

## Installation
Installation can be done with a single command:

```bash
pip install quantex
```

---

## Development
1. Create a new branch: `git checkout -b feature/<name>`
2. Write your code & tests.
3. Install the git hooks once per clone: `poetry run pre-commit install`.
   Hooks will run `black --check` and `ruff` automatically on every commit.
4. Ensure `poetry run pytest` passes and the pre-commit hooks are clean.

---

## Contributing
Contributions, bug reports, and feature requests are welcome! Please open an
issue to discuss what you'd like to work on or submit a pull request directly.
We follow the "fork → feature branch → pull request" workflow. By
contributing you agree to license your work under the same terms as Quantex.
