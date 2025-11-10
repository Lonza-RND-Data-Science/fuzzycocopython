import copy
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from lfa_toolbox.core.fis.singleton_fis import SingletonFIS
from lfa_toolbox.view.fis_viewer import FISViewer
from lfa_toolbox.view.mf_viewer import MembershipFunctionViewer

from .utils import parse_fuzzy_system_from_description, to_linguistic_components


class FuzzyCocoPlotMixin:
    """
    A mixin class providing plotting methods for FuzzyCoco estimators.

    Requires that the estimator using this mixin define:

      - ``self.variables_``
      - ``self.rules_``
      - ``self.model_`` (if using ``plot_rule_activations`` or ``plot_aggregated_output``)
      - ``self.default_rules_`` (if referencing default rules, e.g. in ``plot_aggregated_output``)
      - ``self._predict`` (if used in ``plot_aggregated_output``)
    """

    def plot_fuzzy_sets(self, variable=None, target_rescale=None, **kwargs):
        """Plot membership functions for variables.

        Args:
            variable: None, str, or list[str]. If None, plot all variables; if str,
                plot only that variable; if list, plot each listed variable.
            target_rescale: Optional scaling factor applied to output variables for visualization.
            **kwargs: Extra options passed to the membership function viewer.
        """
        variables, _, _ = self._get_plot_components(target_rescale)
        var_lookup = {lv.name: lv for lv in variables}
        var_list = self._to_var_list(variable)

        if var_list is None:
            names_iter = [lv.name for lv in variables]
        else:
            names_iter = var_list

        for name in names_iter:
            lv = var_lookup.get(name)
            if lv is None:
                raise ValueError(f"Variable '{name}' not found in self.variables_.")
            fig, ax = plt.subplots()
            ax.set_title(lv.name)
            for label, mf in lv.ling_values.items():
                MembershipFunctionViewer(mf, ax=ax, label=label, **kwargs)
            ax.legend()
            plt.show()

    def plot_fuzzification(self, sample, variable=None, target_rescale=None, **kwargs):
        """Plot membership functions and overlay fuzzification for a sample.

        Args:
            sample: Array-like, dict, or pandas Series holding crisp inputs.
            variable: None, str, or list[str]. If None, plot all input variables
                present in the sample; if str, only that variable; if list, each listed variable.
            target_rescale: Optional scaling factor for output variables when visualizing.
            **kwargs: Extra options forwarded to the membership function viewer.
        """
        # Normalize sample -> dict of {feature_name: value}
        try:
            # pandas Series or dict-like with keys
            if hasattr(sample, "to_dict"):
                sample_dict = dict(sample.to_dict())
            elif isinstance(sample, dict):
                sample_dict = dict(sample)
            else:
                # array-like -> map via feature_names_in_
                sample_dict = {name: value for name, value in zip(self.feature_names_in_, sample, strict=False)}
        except Exception as e:
            raise ValueError(
                "Could not interpret `sample`. Provide a dict/Series or an array aligned with `feature_names_in_`."
            ) from e

        variables, _, _ = self._get_plot_components(target_rescale)
        var_lookup = {lv.name: lv for lv in variables}
        var_list = self._to_var_list(variable)
        factor = float(target_rescale) if target_rescale else None

        output_names = self._output_variable_names()
        output_names_upper = {name.upper() for name in output_names}

        def is_output_name(name):
            if output_names:
                return name in output_names or name.upper() in output_names_upper
            return name.upper() in {"OUT", "TARGET"}

        if var_list is None:
            names_iter = [name for name in var_lookup if not is_output_name(name) and name in sample_dict]
        else:
            names_iter = []
            for name in set(var_list):
                lv = var_lookup.get(name)
                if lv is None:
                    raise ValueError(f"Variable '{name}' not found in self.variables_.")
                if name not in sample_dict:
                    continue
                names_iter.append(name)

        for name in names_iter:
            lv = var_lookup[name]
            crisp_value = sample_dict[name]
            if factor and is_output_name(name):
                crisp_value = float(crisp_value) * factor
            fig, ax = plt.subplots()
            ax.set_title(f"{lv.name} (Input: {crisp_value})")
            for label, mf in lv.ling_values.items():
                mvf = MembershipFunctionViewer(mf, ax=ax, label=label, **kwargs)
                mvf.fuzzify(crisp_value)
            ax.legend()
            plt.show()

    def plot_rule_activations(self, x, figsize=(9, 4), sort=True, top=None, annotate=True, tick_fontsize=8):
        """Bar plot of rule activations for a single sample.

        Args:
            x: Single sample as array-like compatible with ``rules_activations``.
            figsize: Matplotlib figure size.
            sort: Sort bars by activation descending.
            top: Show only the first ``top`` rules after sorting.
            annotate: Write activation values on top of bars.
            tick_fontsize: Font size for x tick labels.

        Returns:
            None. Displays a matplotlib figure.
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        a = self.rules_activations(x)  # (n_rules,)

        raw_names = getattr(self, "rules_", None)
        if isinstance(raw_names, list | tuple) and len(raw_names) == a.size:
            labels = []
            for i, r in enumerate(raw_names, 1):
                name = getattr(r, "name", None)
                labels.append(
                    str(name) if name is not None else str(r) if not isinstance(r, int | float | str) else f"Rule {i}"
                )
        else:
            labels = [f"Rule {i + 1}" for i in range(a.size)]

        df = pd.DataFrame({"rule": labels, "activation": a})
        if sort:
            df = df.sort_values("activation", ascending=False, kind="mergesort")
        if top is not None:
            df = df.head(int(top))

        x_pos = np.arange(len(df))
        fig, ax = plt.subplots(figsize=figsize)
        heights = df["activation"].to_numpy()
        bars = ax.bar(x_pos, heights)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Activation")
        ax.set_title("Rule activations (single sample)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            df["rule"].astype(str).tolist(),
            rotation=45,
            ha="right",
            fontsize=tick_fontsize,
        )

        if annotate:
            for bar, value in zip(bars, heights, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        fig.tight_layout()
        plt.show()

    def plot_aggregated_output(self, input_sample, figsize=(12, 10), target_rescale=None):
        """Visualize the aggregated fuzzy output for an input sample.

        Uses a ``SingletonFIS`` to mirror the C++ singleton-based defuzzification.

        Args:
            input_sample: Single sample of crisp input values.
            figsize: Matplotlib figure size.
            target_rescale: Optional scaling factor applied to output variables for visualization.
        """

        # Build a mapping for the input values.
        if hasattr(input_sample, "to_dict"):
            sample_dict = dict(input_sample.to_dict())
        elif isinstance(input_sample, dict):
            sample_dict = dict(input_sample)
        else:
            try:
                sample_dict = {
                    name: float(value) for name, value in zip(self.feature_names_in_, input_sample, strict=False)
                }
            except Exception as exc:
                raise ValueError(
                    "Provide as input_sample a dict/Series or an array aligned with `feature_names_in_`."
                ) from exc

        variables, rules, default_rules = self._get_plot_components(target_rescale)
        var_lookup = {lv.name: lv for lv in variables}

        # output_names = self._output_variable_names()
        input_names = list(self.feature_names_in_)
        missing = [name for name in input_names if name not in sample_dict]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing values for input variables: {missing_str}")

        input_sample = {name: float(sample_dict[name]) for name in input_names}

        target_lv = next(
            (lv for lv in var_lookup.values() if lv.name.upper() == self.target_name_in_.upper()),
            None,
        )

        if target_lv is None:
            available = ", ".join(sorted(var_lookup))
            raise ValueError(
                f"Output linguistic variable '{self.target_name_in_}' not found; available variables: {available}"
            )

        # Create a SingletonFIS instance using the (optionally) rescaled rules.
        fis = SingletonFIS(
            rules=rules,
            default_rule=(default_rules[0] if default_rules else None),
        )

        result = fis.predict(input_sample)
        # result_cpp = self._predict(input_sample)

        # if not np.isclose(float(result.get(self.target_name_in_)), float(result_cpp[0])):
        #    raise ValueError(
        #        f"Python and C++ defuzzification results do not match: {result} vs. {result_cpp}"
        #    )
        # Show the aggregated fuzzy output via FISViewer.
        fisv = FISViewer(fis, figsize=figsize)
        fisv.show()

    def _get_plot_components(self, target_rescale):
        """Return variables/rules/default rules, rescaling outputs when requested."""
        if not target_rescale or target_rescale == 1:
            return self.variables_, self.rules_, self.default_rules_

        desc = getattr(self, "description_", None)
        if not desc:
            return self.variables_, self.rules_, self.default_rules_

        factor = float(target_rescale)
        scaled_desc = copy.deepcopy(desc)

        try:
            output_vars = scaled_desc["fuzzy_system"]["variables"]["output"]
        except (TypeError, KeyError):
            return self.variables_, self.rules_, self.default_rules_

        for sets in output_vars.values():
            for label, pos in list(sets.items()):
                try:
                    sets[label] = float(pos) * factor
                except (TypeError, ValueError):
                    sets[label] = pos

        variables_dict, rules_dict, defaults_dict = parse_fuzzy_system_from_description(scaled_desc)
        scaled_vars, scaled_rules, scaled_defaults = to_linguistic_components(variables_dict, rules_dict, defaults_dict)
        return scaled_vars, scaled_rules, scaled_defaults

    def _output_variable_names(self):
        names = set()
        target_names = getattr(self, "target_names_in_", None)
        if target_names:
            names.update(target_names)
        target_name = getattr(self, "target_name_in_", None)
        if target_name:
            names.add(target_name)
        return names

    def _to_var_list(self, variable):
        """Normalize `variable` into a list of variable names or None."""
        if variable is None or variable is False:
            return None
        if isinstance(variable, str):
            return [variable]
        if isinstance(variable, Sequence):
            # accept tuples/lists of strings
            return list(variable)
        raise TypeError("`variable` must be None, str, or a sequence of str.")
