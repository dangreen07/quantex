# Quantex

A simple quant strategy creation and back-testing package written in Python.  
Quantex aims to provide a lightweight foundation for building trading
strategies, ingesting historical market data, and evaluating performance –
all without the heavy overhead of larger, more opinionated quant libraries.

---

## Documentation & Source Code

To see the documentation go [here](https://dangreen07.github.io/quantex/).

The source code for this library can be found [here](https://github.com/dangreen07/quantex)

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
