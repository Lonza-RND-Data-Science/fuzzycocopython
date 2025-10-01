import os
import sys
import warnings
from datetime import datetime
from types import ModuleType

# -- Path setup --------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT)

# -- Project information -----------------------------------------------------
project = "FuzzyCocoPython"
author = "FuzzyCocoPython contributors"
current_year = datetime.utcnow().year
copyright = f"{current_year}, {author}"

# Try to read version from pyproject.toml without importing the package
version = release = "0.0.0"
if sys.version_info >= (3, 11):
    import tomllib as _tomllib
else:  # pragma: no cover - Python < 3.11
    _tomllib = None

tomllib: ModuleType | None = _tomllib

if tomllib is not None:
    try:
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
            data = tomllib.load(f)
            version = release = data.get("project", {}).get("version", version)
    except OSError as exc:
        warnings.warn(f"Unable to read project version: {exc}", RuntimeWarning, stacklevel=2)

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Mock heavy/compiled deps so autodoc import works in CI
autodoc_mock_imports = [
    "fuzzycocopython._fuzzycoco_core",
    "lfa_toolbox",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
