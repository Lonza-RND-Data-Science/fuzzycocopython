# FuzzyCocoPython

[![Tests](https://github.com/arthurbabey/fuzzycocopython/actions/workflows/tests.yml/badge.svg)](https://github.com/arthurbabey/fuzzycocopython/actions/workflows/tests.yml)
[![Build](https://github.com/arthurbabey/fuzzycocopython/actions/workflows/build.yml/badge.svg)](https://github.com/arthurbabey/fuzzycocopython/actions/workflows/build.yml)
[![Coverage](https://img.shields.io/badge/coverage-pytest--cov-blue)](https://github.com/arthurbabey/fuzzycocopython/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/badge/PyPI-pending-lightgrey)](https://pypi.org/project/fuzzycocopython/)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-success)](https://www.gnu.org/licenses/agpl-3.0.html)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-blue)](#installation)

[Documentation](https://arthurbabey.github.io/fuzzycocopython/)


Python bindings and scikit-learn style estimators for [fuzzycoco](https://github.com/Lonza-RND-Data-Science/fuzzycoco),
an evolutionary fuzzy rule learning engine written in C++. This package wraps the C++ core as a Python module
and exposes `FuzzyCocoClassifier` and `FuzzyCocoRegressor` with a familiar fit/predict API.

## Features
- Train fuzzy rule-based classifiers and regressors using `fit`, `predict`, and `score`
- Inspect learned linguistic variables, rules, and activation statistics from Python
- Persist trained estimators with `save`/`load` helpers based on `joblib`


## Installation

This package is not yet on PyPI; install it from source. Make sure the following prerequisites are available:

- A C++17 compiler toolchain (GCC/Clang on Linux & macOS, MSVC on Windows)
- [CMake](https://cmake.org/) ≥ 3.21 and [Ninja](https://ninja-build.org/) on your PATH
- [uv](https://github.com/astral-sh/uv) is recommended for dependency management, but `pip` works just as well

Clone the repository, initialise the `fuzzycoco` submodule, then build and install in editable mode with uv:

```bash
git clone https://github.com/arthurbabey/fuzzycocopython.git
cd fuzzycocopython
git submodule update --init --recursive

uv venv
source .venv/bin/activate
uv pip install -e .

```

If you prefer the standard tooling, create a virtual environment manually and install with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .
```

For development tasks (tests, linting, docs) install the optional toolchain:

```bash
uv pip install -e '.[dev]'
```

The build compiles the bundled C++ bindings. Refer to the
[`fuzzycoco` project](https://github.com/arthurbabey/fuzzycoco) for background on the engine itself.

## Quick start

```python
import pandas as pd
from sklearn.datasets import load_iris
from fuzzycocopython import FuzzyCocoClassifier

data = load_iris(as_frame=True)
clf = FuzzyCocoClassifier(random_state=0)
clf.fit(data.data, data.target)

preds = clf.predict(data.data)
score = clf.score(data.data, data.target)

print(f"Accuracy: {score:.3f}")
print(clf.rules_df_.head())
```

For a guided tour, open `demo.ipynb`. Additional usage examples live in `tests/test_fuzzycocopython.py`.

## Documentation

Full API documentation is available at [arthurbabey.github.io/fuzzycocopython](https://arthurbabey.github.io/fuzzycocopython/).
To build the docs locally run:

```bash
uv pip install -e '.[docs]'
uv run sphinx-build -W -b html docs docs/_build/html
```

## Pre-commit hooks

Install and activate the provided hooks (Ruff lint/format, mypy, general hygiene) once per clone:

```bash
uv run pre-commit install
```

Run them against the full tree when needed:

```bash
uv run pre-commit run --all-files
```


## License

This fuzzycoco software is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
