from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence

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
    build_fuzzycoco_params,
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

    _default_metrics_weights: dict[str, float] | None = None

    def __init__(
        self,
        nb_rules=5,
        nb_max_var_per_rule=3,
        max_generations=100,
        max_fitness=1.0,
        nb_cooperators=2,
        influence_rules_initial_population=False,
        influence_evolving_ratio=0.8,
        nb_sets_in=2,
        nb_sets_out=2,
        pop_size_rules=200,
        pop_size_mfs=200,
        elite_size_rules=5,
        elite_size_mfs=5,
        cx_prob_rules=0.6,
        cx_prob_mfs=0.9,
        mut_flip_genome_rules=0.4,
        mut_flip_genome_mfs=0.2,
        mut_flip_bit_rules=0.01,
        mut_flip_bit_mfs=0.01,
        nb_bits_pos_in=8,
        nb_bits_pos_out=8,
        nb_bits_vars_in=None,
        nb_bits_vars_out=None,
        nb_bits_sets_in=None,
        nb_bits_sets_out=None,
        threshold=0.5,
        metrics_weights=None,
        features_weights=None,
        random_state=None,
    ):
        """Initialize a FuzzyCoco estimator with explicit hyper-parameters.

        Parameters
        ----------
        nb_rules : int, default=5
            Number of fuzzy rules evolved during optimisation.
        nb_max_var_per_rule : int, default=3
            Maximum number of antecedents allowed in a rule.
        max_generations : int, default=100
            Evolution generations for both rule and membership function search.
        max_fitness : float, default=1.0
            Target fitness score that can trigger early stopping.
        nb_cooperators : int, default=2
            Number of cooperating agents in the fuzzy optimisation engine.
        influence_rules_initial_population : bool, default=False
            Whether to seed the population with rule influence heuristics.
        influence_evolving_ratio : float, default=0.8
            Ratio controlling how strongly influence is applied during evolution.
        nb_sets_in : int, default=2
            Number of linguistic sets per input variable.
        nb_sets_out : int, default=2
            Number of linguistic sets per output variable.
        pop_size_rules : int, default=200
            Population size for the rule genome evolution.
        pop_size_mfs : int, default=200
            Population size for the membership-function genome evolution.
        elite_size_rules : int, default=5
            Number of elite individuals kept each generation in the rule evolution.
        elite_size_mfs : int, default=5
            Number of elite individuals kept each generation in the membership evolution.
        cx_prob_rules : float, default=0.6
            Crossover probability for rule evolution.
        cx_prob_mfs : float, default=0.9
            Crossover probability for membership-function evolution.
        mut_flip_genome_rules : float, default=0.4
            Genome-level mutation probability for rules.
        mut_flip_genome_mfs : float, default=0.2
            Genome-level mutation probability for membership functions.
        mut_flip_bit_rules : float, default=0.01
            Bit-flip mutation probability for rules.
        mut_flip_bit_mfs : float, default=0.01
            Bit-flip mutation probability for membership functions.
        nb_bits_pos_in : int, default=8
            Bit width used to encode the positions of input membership functions.
        nb_bits_pos_out : int, default=8
            Bit width used to encode the positions of output membership functions.
        nb_bits_vars_in : int | None, optional
            Override for the automatically computed input variable bit width.
        nb_bits_vars_out : int | None, optional
            Override for the automatically computed output variable bit width.
        nb_bits_sets_in : int | None, optional
            Override for the automatically computed input set bit width.
        nb_bits_sets_out : int | None, optional
            Override for the automatically computed output set bit width.
        threshold : float, default=0.5
            Default singleton defuzzification threshold applied to each output.
        metrics_weights : dict[str, float] | None, optional
            Mapping of fitness metrics to weights. A sensible default is provided
            by the classifier/regressor subclasses when omitted.
        features_weights : dict[str, float] | None, optional
            Optional per-feature weights used by the underlying fitness function.
        random_state : int | RandomState | None, optional
            Seed or NumPy-compatible random state for reproducibility.
        """

        self.nb_rules = nb_rules
        self.nb_max_var_per_rule = nb_max_var_per_rule
        self.max_generations = max_generations
        self.max_fitness = max_fitness
        self.nb_cooperators = nb_cooperators
        self.influence_rules_initial_population = influence_rules_initial_population
        self.influence_evolving_ratio = influence_evolving_ratio
        self.nb_sets_in = nb_sets_in
        self.nb_sets_out = nb_sets_out
        self.pop_size_rules = pop_size_rules
        self.pop_size_mfs = pop_size_mfs
        self.elite_size_rules = elite_size_rules
        self.elite_size_mfs = elite_size_mfs
        self.cx_prob_rules = cx_prob_rules
        self.cx_prob_mfs = cx_prob_mfs
        self.mut_flip_genome_rules = mut_flip_genome_rules
        self.mut_flip_genome_mfs = mut_flip_genome_mfs
        self.mut_flip_bit_rules = mut_flip_bit_rules
        self.mut_flip_bit_mfs = mut_flip_bit_mfs
        self.nb_bits_pos_in = nb_bits_pos_in
        self.nb_bits_pos_out = nb_bits_pos_out
        self.nb_bits_vars_in = nb_bits_vars_in
        self.nb_bits_vars_out = nb_bits_vars_out
        self.nb_bits_sets_in = nb_bits_sets_in
        self.nb_bits_sets_out = nb_bits_sets_out
        self.threshold = threshold
        if metrics_weights is None:
            default_metrics = self._default_metrics_weights
            metrics = None if default_metrics is None else dict(default_metrics)
        else:
            metrics = metrics_weights
        self.metrics_weights = metrics

        self.features_weights = None if features_weights is None else features_weights
        self.random_state = random_state

    # ──────────────────────────────────────────────────────────────────────
    # internal helpers
    # ──────────────────────────────────────────────────────────────────────
    def _resolve_seed(self):
        """Return a deterministic 32-bit seed derived from sklearn RNG."""
        rng = check_random_state(self.random_state)
        return int(rng.randint(0, 2**32 - 1, dtype=np.uint32))

    def _extract_output_names(self):
        """Return output variable names from the stored description."""
        fuzzy_desc = getattr(self, "description_", None)
        if not fuzzy_desc:
            return []
        variables = fuzzy_desc.get("fuzzy_system", {}).get("variables", {})
        outputs = variables.get("output", {})
        return list(outputs.keys())

    @staticmethod
    def _rename_membership_label(label, old_var, new_var):
        """Rename membership labels carrying the old variable prefix."""
        if not isinstance(label, str):
            return label
        if label == old_var:
            return new_var
        for sep in (".", "_", "-", " "):
            prefix = f"{old_var}{sep}"
            if label.startswith(prefix):
                return f"{new_var}{sep}{label[len(prefix):]}"
        return label

    def _rebuild_from_description(self):
        """Refresh cached Python helpers and fuzzy system from description."""
        parsed = parse_fuzzy_system_from_description(self.description_)
        self.variables_, self.rules_, self.default_rules_ = to_linguistic_components(*parsed)
        self.variables_view_, self.rules_view_, self.default_rules_view_ = to_views_components(*parsed)
        self.variables_df_, self.rules_df_ = to_tables_components(*parsed)

        output_names = self._extract_output_names()
        self.target_names_in_ = output_names
        if output_names:
            self.target_name_in_ = output_names[0]
        self.n_outputs_ = len(output_names)

        fuzzy_desc = self.description_.get("fuzzy_system") if self.description_ else None
        self._fuzzy_system_dict_ = copy.deepcopy(fuzzy_desc) if fuzzy_desc is not None else None
        self._fuzzy_system_string_ = None
        self._fuzzy_system_ = None
        try:
            self._ensure_fuzzy_system()
        except ModuleNotFoundError:  # pragma: no cover - happens in partial installs
            pass
        except AttributeError:  # pragma: no cover - defensive for missing bindings
            pass

        # Drop the live engine; predictions fall back to the serialized description.
        self.model_ = None

    def _normalize_target_name_change(self, names):
        """Normalize provided names into a mapping old->new."""
        current = list(getattr(self, "target_names_in_", []) or self._extract_output_names())
        if not current:
            raise RuntimeError("Estimator does not expose any output variables to rename.")

        if isinstance(names, str):
            if len(current) != 1:
                raise ValueError("Provide a mapping or list when renaming multi-output models.")
            mapping = {current[0]: str(names)}
        elif isinstance(names, Mapping):
            mapping = {str(k): str(v) for k, v in names.items()}
            unknown = sorted(set(mapping) - set(current))
            if unknown:
                raise ValueError(f"Unknown output variables: {', '.join(unknown)}")
        elif isinstance(names, Sequence):
            new_names = [str(n) for n in names]
            if len(new_names) != len(current):
                raise ValueError(
                    f"Expected {len(current)} output names, got {len(new_names)}.",
                )
            mapping = {old: new for old, new in zip(current, new_names, strict=False)}
        else:
            raise TypeError("`names` must be a string, sequence, or mapping.")

        normalized = {old: new for old, new in mapping.items() if new and new != old}
        updated = [normalized.get(name, name) for name in current]
        if len(updated) != len(set(updated)):
            raise ValueError("Output names must be unique.")
        return normalized

    def set_target_names(self, names):
        """Rename the output variables and refresh cached structures.

        Args:
            names: String (single-output), sequence of strings matching the number
                of outputs, or a mapping ``{old_name: new_name}``.

        Returns:
            self
        """
        check_is_fitted(self, attributes=["description_", "is_fitted_"])
        mapping = self._normalize_target_name_change(names)
        if not mapping:
            return self

        fs = self.description_.get("fuzzy_system")
        if fs is None:
            raise RuntimeError("Estimator is missing the fuzzy system description.")

        variables = fs.get("variables", {})
        outputs = variables.get("output", {})
        if not outputs:
            raise RuntimeError("Estimator description lacks fuzzy output variables.")

        new_outputs = {}
        for var_name, sets in outputs.items():
            target_name = mapping.get(var_name, var_name)
            renamed_sets = {}
            for set_name, value in sets.items():
                renamed_sets[self._rename_membership_label(set_name, var_name, target_name)] = value
            new_outputs[target_name] = renamed_sets
        variables["output"] = new_outputs

        rules = fs.get("rules", {})
        new_rules = {}
        for rule_name, rule_def in rules.items():
            updated_rule = {}
            for key, part in rule_def.items():
                if key not in ("antecedents", "consequents") or not isinstance(part, dict):
                    updated_rule[key] = part
                    continue
                changed_part = {}
                for var, mf_dict in part.items():
                    renamed_var = mapping.get(var, var)
                    if isinstance(mf_dict, dict):
                        renamed_mf = {
                            self._rename_membership_label(label, var, renamed_var): weight
                            for label, weight in mf_dict.items()
                        }
                    else:
                        renamed_mf = mf_dict
                    changed_part[renamed_var] = renamed_mf
                updated_rule[key] = changed_part
            new_rules[rule_name] = updated_rule
        fs["rules"] = new_rules

        defaults = fs.get("default_rules", {})
        new_defaults = {}
        for var, label in defaults.items():
            renamed_var = mapping.get(var, var)
            new_defaults[renamed_var] = self._rename_membership_label(label, var, renamed_var)
        fs["default_rules"] = new_defaults

        thresholds = self.description_.get("defuzz_thresholds")
        if isinstance(thresholds, dict):
            new_thresholds = {}
            for var, value in thresholds.items():
                renamed_var = mapping.get(var, var)
                new_thresholds[renamed_var] = value
            self.description_["defuzz_thresholds"] = new_thresholds

        self._rebuild_from_description()
        return self

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
            multi_output=True,
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
        self.target_names_in_ = list(y_headers)
        self.n_outputs_ = y_2d.shape[1]

        metrics_weights = self.metrics_weights
        if metrics_weights is None:
            metrics_weights = self._default_metrics_weights

        params_obj = build_fuzzycoco_params(
            nb_features=self.n_features_in_,
            n_outputs=self.n_outputs_,
            nb_rules=self.nb_rules,
            nb_max_var_per_rule=self.nb_max_var_per_rule,
            max_generations=self.max_generations,
            max_fitness=self.max_fitness,
            nb_cooperators=self.nb_cooperators,
            influence_rules_initial_population=self.influence_rules_initial_population,
            influence_evolving_ratio=self.influence_evolving_ratio,
            nb_sets_in=self.nb_sets_in,
            nb_sets_out=self.nb_sets_out,
            nb_bits_pos_in=self.nb_bits_pos_in,
            nb_bits_pos_out=self.nb_bits_pos_out,
            nb_bits_vars_in=self.nb_bits_vars_in,
            nb_bits_vars_out=self.nb_bits_vars_out,
            nb_bits_sets_in=self.nb_bits_sets_in,
            nb_bits_sets_out=self.nb_bits_sets_out,
            pop_size_rules=self.pop_size_rules,
            elite_size_rules=self.elite_size_rules,
            cx_prob_rules=self.cx_prob_rules,
            mut_flip_genome_rules=self.mut_flip_genome_rules,
            mut_flip_bit_rules=self.mut_flip_bit_rules,
            pop_size_mfs=self.pop_size_mfs,
            elite_size_mfs=self.elite_size_mfs,
            cx_prob_mfs=self.cx_prob_mfs,
            mut_flip_genome_mfs=self.mut_flip_genome_mfs,
            mut_flip_bit_mfs=self.mut_flip_bit_mfs,
            threshold=self.threshold,
            metrics_weights=metrics_weights,
            features_weights=self.features_weights,
        )

        if hasattr(params_obj, "fitness_params"):
            params_obj.fitness_params.fix_output_thresholds(self.n_outputs_)
        self._fuzzy_params_ = params_obj

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
        params = state.get("_fuzzy_params_")
        if isinstance(params, FuzzyCocoParams):
            state["_fuzzy_params_"] = copy.deepcopy(params.describe())
        return state

    def __setstate__(self, state):
        params = state.get("_fuzzy_params_")
        if isinstance(params, dict):
            state["_fuzzy_params_"] = FuzzyCocoParams.from_dict(params)
        self.__dict__.update(state)

        output_names = []
        if getattr(self, "description_", None):
            output_names = self._extract_output_names()
        if not getattr(self, "target_names_in_", None):
            self.target_names_in_ = output_names
        if output_names and not getattr(self, "target_name_in_", None):
            self.target_name_in_ = output_names[0]

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
    _default_metrics_weights = {"accuracy": 1.0}

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
    _default_metrics_weights = {"rmse": 1.0}

    def predict(self, X):
        """Predict continuous targets for ``X``.

        Returns a 1D array for single-output models or 2D for multi-output.
        """
        check_is_fitted(self, attributes=["model_"])
        dfin, _ = self._prepare_inference_input(X)
        preds_df = self._predict_dataframe(dfin)
        raw = np.asarray(preds_df.to_list(), dtype=float)
        return raw.ravel() if raw.shape[1] == 1 else raw
