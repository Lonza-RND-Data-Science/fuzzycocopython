from __future__ import annotations

import copy
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import get_scorer
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._fuzzycoco_core import DataFrame, FuzzyCoco, FuzzyCocoParams, FuzzySystem, RandomGenerator
from .fuzzycoco_plot_mixin import FuzzyCocoPlotMixin
from .utils import (
    make_fuzzy_params,
    parse_fuzzy_system_from_description,
    to_linguistic_components,
    to_tables_components,
    to_views_components,
)


def save_model(model, filepath, *, compress=3):
    """Save a fitted estimator to disk with joblib.

    Parameters
    - model: the fitted estimator instance (classifier or regressor)
    - filepath: target path (str or Path-like)
    - compress: joblib compression level or bool

    Returns the path string.
    """

    path = os.fspath(filepath)
    joblib.dump(model, path, compress=compress)
    return path


def load_model(filepath):
    """Load a previously saved estimator created with save_model."""

    return joblib.load(os.fspath(filepath))


# ────────────────────────────────────────────────────────────────────────────────
# Base wrapper
# ────────────────────────────────────────────────────────────────────────────────
class _FuzzyCocoBase(BaseEstimator):
    """Shared logic for FuzzyCocoClassifier and FuzzyCocoRegressor.

    Provides scikit-learn compatible ``fit``/``predict``/``score`` plus
    utilities to inspect fuzzy rules and variables produced by the
    underlying C++ engine.
    """

    def __init__(self, params=None, random_state=None, params_overrides=None, **sk_params):
        """Initialize the estimator.

        Besides passing a full ``params`` object or nested dict, you can also
        provide scikit-learn style flat parameters. Supported forms:
          - Global fields directly, e.g. ``nb_rules=10``
          - Nested using double-underscore, e.g. ``input_vars_params__nb_sets=3``
        These are merged into ``params_overrides`` at fit time.
        """
        self.params = params
        self.random_state = random_state
        # Merge flat sk_params into params_overrides as nested dicts
        merged_overrides = dict(params_overrides or {})
        if sk_params:
            for k, v in sk_params.items():
                if "__" in k:
                    sect, key = k.split("__", 1)
                    merged_overrides.setdefault(sect, {})[key] = v
                else:
                    merged_overrides.setdefault("global_params", {})[k] = v
        self.params_overrides = merged_overrides or None

    # ──────────────────────────────────────────────────────────────────────
    # internal helpers
    # ──────────────────────────────────────────────────────────────────────
    def _resolve_seed(self):
        """Return a deterministic 32-bit seed derived from sklearn RNG."""
        rng = check_random_state(self.random_state)
        return int(rng.randint(0, 2**32 - 1, dtype=np.uint32))

    def _make_dataframe(self, arr, header):
        """Build the C++ DataFrame from a 2D numpy array and header labels."""
        rows = [list(header)] + arr.astype(str).tolist()
        return DataFrame(rows, False)

    def _prepare_dataframes(self, X_arr, y_arr=None, *, y_headers=None):
        """Create input/output DataFrame objects (output optional)."""
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        dfin = self._make_dataframe(X_arr, self.feature_names_in_)

        if y_arr is None:
            return dfin, None

        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if y_headers is not None:
            headers = list(y_headers)
        else:
            headers = [f"OUT_{i + 1}" for i in range(y_arr.shape[1])]

        dfout = self._make_dataframe(y_arr, headers)
        return dfin, dfout

    def _resolve_feature_names(self, X, provided, n_features):
        """Resolve final feature names from DataFrame, provided list, or defaults."""
        if isinstance(X, pd.DataFrame):
            names = list(X.columns)
        elif provided is not None:
            names = list(provided)
        else:
            names = [f"feature_{i + 1}" for i in range(n_features)]
        # ensure string column names for the C++ DataFrame
        names = [str(n) for n in names]

        if len(names) != n_features:
            raise ValueError(
                "feature_names length does not match number of features",
            )
        return names

    def _resolve_target_headers(self, y, y_arr, provided):
        """Return (output headers, target name) inferred from y and overrides."""
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        if isinstance(y, pd.DataFrame):
            headers = list(y.columns)
        elif isinstance(y, pd.Series):
            headers = [y.name] if y.name else []
        else:
            headers = []

        if not headers:
            if provided:
                if y_arr.shape[1] == 1:
                    headers = [provided]
                else:
                    headers = [f"{provided}_{i + 1}" for i in range(y_arr.shape[1])]
            else:
                headers = [f"OUT_{i + 1}" for i in range(y_arr.shape[1])]

        # ensure string headers for the C++ DataFrame
        headers = [str(h) for h in headers]
        target_name = provided or (headers[0] if headers else "OUT")
        return headers, target_name

    def _prepare_inference_input(self, X):
        """Align/validate prediction input and build the C++ DataFrame."""
        if isinstance(X, pd.DataFrame):
            try:
                aligned = X.loc[:, self.feature_names_in_]
            except KeyError as exc:
                missing = set(self.feature_names_in_) - set(X.columns)
                raise ValueError(
                    f"Missing features in input data: {sorted(missing)}",
                ) from exc
            raw = aligned.to_numpy(dtype=float)
        else:
            raw = np.asarray(X, dtype=float)

        arr = check_array(raw, accept_sparse=False, ensure_2d=True, dtype=float)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1]} features, but {self.__class__.__name__} \
                    is expecting {self.n_features_in_} features as input",
            )

        dfin = self._make_dataframe(arr, self.feature_names_in_)
        return dfin, arr

    def _ensure_fuzzy_system(self):
        """Rebuild and memoize the C++ FuzzySystem from the saved description."""
        if getattr(self, "_fuzzy_system_", None) is not None:
            return self._fuzzy_system_

        serialized = getattr(self, "_fuzzy_system_string_", None)
        if not serialized:
            desc = getattr(self, "_fuzzy_system_dict_", None)
            if desc is None:
                if not hasattr(self, "description_"):
                    raise RuntimeError("Estimator is missing the fuzzy system description")
                desc = self.description_.get("fuzzy_system") if self.description_ else None
                if desc is None:
                    raise RuntimeError("Estimator does not contain a fuzzy system description")
                desc = copy.deepcopy(desc)
                self._fuzzy_system_dict_ = desc
            if isinstance(desc, dict):
                from . import _fuzzycoco_core  # local import to avoid cycles

                serialized = _fuzzycoco_core._named_list_from_dict_to_string(desc)
            else:
                serialized = str(desc)
            self._fuzzy_system_string_ = serialized

        self._fuzzy_system_ = FuzzySystem.load_from_string(serialized)
        return self._fuzzy_system_

    def _predict_dataframe(self, dfin):
        """Predict using the live engine when available, else via saved description."""
        model = getattr(self, "model_", None)
        if model is not None:
            return model.predict(dfin)
        from . import _fuzzycoco_core  # local import to avoid circular deps

        if not getattr(self, "description_", None):
            raise RuntimeError("Missing model description for prediction")
        return _fuzzycoco_core.FuzzyCoco.load_and_predict_from_dict(dfin, self.description_)

    def _compute_rule_fire_levels(self, sample):
        """Compute rule activations for a single sample (1D)."""
        model = getattr(self, "model_", None)
        if model is not None:
            values = model.rules_fire_from_values(sample)
        else:
            from . import _fuzzycoco_core

            mapping = {name: float(value) for name, value in zip(self.feature_names_in_, sample, strict=False)}
            values = _fuzzycoco_core._rules_fire_from_description(self.description_, mapping)
        return np.asarray(values, dtype=float)

    # ──────────────────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────────────────
    def fit(self, X, y, **fit_params):
        """Fit a fuzzy rule-based model.

        Args:
            X: 2D array-like or pandas DataFrame of shape (n_samples, n_features).
            y: 1D or 2D array-like or pandas Series/DataFrame with targets.
            **fit_params: Optional keyword-only parameters:
                - ``feature_names``: list of column names to use when ``X`` is not a DataFrame.
                - ``target_name``: name of the output variable in the fuzzy system.

        Returns:
            The fitted estimator instance.
        """
        feature_names = fit_params.pop("feature_names", None)
        target_name = fit_params.pop("target_name", None)
        fit_params.pop("output_filename", None)  # backward compat, no-op
        if fit_params:
            unexpected = ", ".join(sorted(fit_params))
            raise TypeError(f"Unexpected fit parameters: {unexpected}")

        X_arr, y_arr = check_X_y(
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            dtype=float,
        )

        self.feature_names_in_ = self._resolve_feature_names(X, feature_names, X_arr.shape[1])
        self.n_features_in_ = len(self.feature_names_in_)

        y_arr = np.asarray(y_arr, dtype=float)
        y_2d = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr
        y_headers, resolved_target = self._resolve_target_headers(y, y_2d, target_name)
        self.target_name_in_ = resolved_target
        self.n_outputs_ = y_2d.shape[1]

        overrides = dict(self.params_overrides or {})
        if not self.params:
            # default antecedents slots per rule (kept small by default)
            gp = overrides.setdefault("global_params", {})
            gp.setdefault("nb_max_var_per_rule", 3)
            # Build params using dataset-aware defaults aligned with C++ logic
            params_obj = make_fuzzy_params(
                overrides,
                nb_input_vars=X_arr.shape[1],
                nb_output_vars=self.n_outputs_,
            )
        else:
            params_obj = (
                self.params
                if isinstance(self.params, FuzzyCocoParams)
                else make_fuzzy_params(
                    self.params,
                    nb_input_vars=X_arr.shape[1],
                    nb_output_vars=self.n_outputs_,
                )
            )

        if hasattr(params_obj, "fitness_params"):
            params_obj.fitness_params.fix_output_thresholds(self.n_outputs_)

        dfin, dfout = self._prepare_dataframes(X_arr, y_2d, y_headers=y_headers)
        rng = RandomGenerator(self._resolve_seed())
        self.model_ = FuzzyCoco(dfin, dfout, params_obj, rng)
        self.model_.run()
        self.model_.select_best()
        self.description_ = self.model_.describe()

        fuzzy_system_desc = self.description_.get("fuzzy_system")
        if fuzzy_system_desc is None:
            raise RuntimeError("Model description missing 'fuzzy_system' section")
        self._fuzzy_system_dict_ = copy.deepcopy(fuzzy_system_desc)
        self._fuzzy_system_string_ = self.model_.serialize_fuzzy_system()
        self._fuzzy_system_ = FuzzySystem.load_from_string(self._fuzzy_system_string_)

        parsed = parse_fuzzy_system_from_description(self.description_)
        self.variables_, self.rules_, self.default_rules_ = to_linguistic_components(*parsed)
        self.variables_view_, self.rules_view_, self.default_rules_view_ = to_views_components(*parsed)
        self.variables_df_, self.rules_df_ = to_tables_components(*parsed)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict outputs for ``X``.

        Implemented by subclasses; here only to document the public API.

        Args:
            X: 2D array-like or pandas DataFrame aligned with ``feature_names_in_``.

        Returns:
            ndarray of predictions; shape depends on the specific estimator.
        """
        raise NotImplementedError

    def score(self, X, y, scoring=None):
        """Compute a default metric on the given test data.

        Args:
            X: Test features.
            y: True targets.
            scoring: Optional scikit-learn scorer string or callable. If ``None``,
                uses ``"accuracy"`` for classifiers and ``"r2"`` for regressors.

        Returns:
            The score as a float.
        """
        scorer = get_scorer(scoring or self._default_scorer)
        return scorer(self, X, y)

    def rules_activations(self, X):
        """Compute rule activation levels for a single sample.

        Args:
            X: Single sample as 1D array-like, pandas Series, or single-row DataFrame.

        Returns:
            1D numpy array of length ``n_rules`` with activation strengths in [0, 1].
        """
        check_is_fitted(self, attributes=["model_"])
        sample = self._as_1d_sample(X)
        if len(sample) != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {len(sample)}",
            )
        return self._compute_rule_fire_levels(sample)

    def rules_stat_activations(self, X, threshold=1e-12, return_matrix=False, sort_by_impact=True):
        """Compute aggregate rule activations for a batch of samples.

        Args:
            X: 2D array-like or DataFrame of samples to analyze.
            threshold: Minimum activation value to count a rule as "used".
            return_matrix: If True, also return the (n_samples, n_rules) activation matrix.
            sort_by_impact: If True, sort the summary by estimated impact.

        Returns:
            If ``return_matrix`` is False, a pandas DataFrame with per-rule statistics
            (mean, std, min, max, usage rates, and impact). If True, returns a tuple
            ``(stats_df, activations_matrix)``.
        """

        check_is_fitted(self, attributes=["model_"])

        if isinstance(X, pd.DataFrame):
            try:
                arr_raw = X.loc[:, self.feature_names_in_].to_numpy(dtype=float)
            except KeyError as exc:
                missing = set(self.feature_names_in_) - set(X.columns)
                raise ValueError(
                    f"Missing features in input data: {sorted(missing)}",
                ) from exc
        else:
            arr_raw = np.asarray(X, dtype=float)

        arr = check_array(arr_raw, accept_sparse=False, ensure_2d=True, dtype=float)
        if arr.shape[0] == 0:
            raise ValueError("Empty X.")
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {arr.shape[1]}",
            )

        activations = np.vstack([self._compute_rule_fire_levels(row.astype(float).tolist()) for row in arr])

        sums = activations.sum(axis=1, keepdims=True)
        share = np.divide(activations, sums, out=np.zeros_like(activations), where=sums > 0)

        usage_rate = (activations >= threshold).mean(axis=0)
        usage_rate_pct = 100.0 * usage_rate
        importance_pct = 100.0 * share.mean(axis=0)
        impact_pct = usage_rate * importance_pct

        idx = self._rules_index(activations.shape[1])
        stats = pd.DataFrame(
            {
                "mean": activations.mean(axis=0),
                "std": activations.std(axis=0),
                "min": activations.min(axis=0),
                "max": activations.max(axis=0),
                "usage_rate": usage_rate,
                "usage_rate_pct": usage_rate_pct,
                "importance_pct": importance_pct,
                "impact_pct": impact_pct,
            },
            index=idx,
        )

        if sort_by_impact:
            stats = stats.sort_values("impact_pct", ascending=False)

        return (stats, activations) if return_matrix else stats

    # ---- helpers ----
    def _as_1d_sample(self, X):
        """Normalize various single‑row inputs (array/Series/DF) to a 1D list."""
        if isinstance(X, pd.Series):
            aligned = X.reindex(self.feature_names_in_)
            if aligned.isnull().any():
                missing = aligned[aligned.isnull()].index.tolist()
                raise ValueError(f"Missing features in sample: {missing}")
            arr = aligned.to_numpy(dtype=float)
        elif isinstance(X, pd.DataFrame):
            if len(X) != 1:
                raise ValueError("Provide a single-row DataFrame for rules_activations.")
            try:
                arr = X.loc[:, self.feature_names_in_].to_numpy(dtype=float)[0]
            except KeyError as exc:
                missing = set(self.feature_names_in_) - set(X.columns)
                raise ValueError(
                    f"Missing features in sample: {sorted(missing)}",
                ) from exc
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            elif arr.ndim != 1:
                raise ValueError(
                    "rules_activations expects a 1D sample or single-row 2D array.",
                )

        if arr.shape[0] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {arr.shape[0]}",
            )

        return arr.astype(float).tolist()

    def _rules_index(self, n_rules):
        names = getattr(self, "rules_", None)
        if isinstance(names, list | tuple) and len(names) == n_rules:
            return list(names)
        return [f"rule_{i}" for i in range(n_rules)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("model_", None)
        state.pop("_fuzzy_system_", None)
        params = state.get("params")
        if isinstance(params, FuzzyCocoParams):
            state["params"] = copy.deepcopy(params.describe())
        return state

    def __setstate__(self, state):
        params = state.get("params")
        if isinstance(params, dict):
            state["params"] = FuzzyCocoParams.from_dict(params)
        self.__dict__.update(state)
        self.model_ = None
        self._fuzzy_system_ = None
        if getattr(self, "_fuzzy_system_dict_", None) is None and getattr(self, "description_", None):
            fuzzy_desc = self.description_.get("fuzzy_system") if self.description_ else None
            if fuzzy_desc is not None:
                self._fuzzy_system_dict_ = copy.deepcopy(fuzzy_desc)
        if state.get("is_fitted_", False):
            self._ensure_fuzzy_system()

    def save(self, filepath, *, compress=3):
        """Save this fitted estimator to disk (convenience wrapper).

        Args:
            filepath: Destination path for the serialized estimator.
            compress: Joblib compression parameter.

        Returns:
            The path used to save the model.
        """
        return save_model(self, filepath, compress=compress)

    @classmethod
    def load(cls, filepath):
        """Load a previously saved estimator instance of this class.

        Args:
            filepath: Path to the serialized estimator created via :meth:`save`.

        Returns:
            An instance of the estimator loaded from disk.
        """
        model = load_model(filepath)
        if not isinstance(model, cls):
            raise TypeError(
                f"Expected instance of {cls.__name__}, got {type(model).__name__}",
            )
        return model

    def describe(self):
        """Return the full model description (variables, rules, defaults).

        Returns:
            A dictionary mirroring the native engine description, including
            the serialized fuzzy system and related metadata.
        """
        return self.description_


# ────────────────────────────────────────────────────────────────────────────────
# Classifier wrapper
# ────────────────────────────────────────────────────────────────────────────────
class FuzzyCocoClassifier(ClassifierMixin, FuzzyCocoPlotMixin, _FuzzyCocoBase):
    _default_scorer = "accuracy"

    def fit(self, X, y, **kwargs):
        """Fit the classifier on ``X`` and ``y``.

        See :meth:`_FuzzyCocoBase.fit` for details on accepted parameters.
        """
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            self.classes_ = np.unique(y_arr)
        else:
            self.classes_ = [np.unique(y_arr[:, i]) for i in range(y_arr.shape[1])]
        return super().fit(X, y, **kwargs)

    def predict(self, X):
        """Predict class labels for ``X``.

        Returns numpy array of labels matching the original label dtype.
        """
        check_is_fitted(self, attributes=["model_"])
        dfin, _ = self._prepare_inference_input(X)
        preds_df = self._predict_dataframe(dfin)
        raw = preds_df.to_list()  # list of rows

        if isinstance(self.classes_[0], np.ndarray) or isinstance(self.classes_, list):
            n_outputs = len(self.classes_)
            y_pred = np.empty((len(raw), n_outputs), dtype=self.classes_[0].dtype)
            for i, row in enumerate(raw):
                for j, val in enumerate(row[:n_outputs]):
                    idx = int(round(val))
                    idx = np.clip(idx, 0, len(self.classes_[j]) - 1)
                    y_pred[i, j] = self.classes_[j][idx]
            if n_outputs == 1:
                return y_pred.ravel()
            return y_pred
        else:
            # single output path
            y_pred_idx = np.array([int(round(v[0])) for v in raw])
            y_pred_idx = np.clip(y_pred_idx, 0, len(self.classes_) - 1)
            return self.classes_[y_pred_idx]


# ────────────────────────────────────────────────────────────────────────────────
# Regressor wrapper
# ────────────────────────────────────────────────────────────────────────────────
class FuzzyCocoRegressor(RegressorMixin, FuzzyCocoPlotMixin, _FuzzyCocoBase):
    _default_scorer = "r2"

    def predict(self, X):
        """Predict continuous targets for ``X``.

        Returns a 1D array for single-output models or 2D for multi-output.
        """
        check_is_fitted(self, attributes=["model_"])
        dfin, _ = self._prepare_inference_input(X)
        preds_df = self._predict_dataframe(dfin)
        raw = np.asarray(preds_df.to_list(), dtype=float)
        return raw.ravel() if raw.shape[1] == 1 else raw
