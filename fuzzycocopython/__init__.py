from .fuzzycoco_base import (
    FuzzyCocoClassifier,
    FuzzyCocoRegressor,
    load_model,
    save_model,
)

try:  # prefer installed package metadata
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("fuzzycocopython")
except Exception:  # pragma: no cover - fallback for editable/uninstalled state
    __version__ = "0.0.0"

# expose both __version__ and version for convenience (mirrors sklearn style)
version = __version__

__all__ = [
    "FuzzyCocoClassifier",
    "FuzzyCocoRegressor",
    "load_model",
    "save_model",
    "__version__",
    "version",
]
