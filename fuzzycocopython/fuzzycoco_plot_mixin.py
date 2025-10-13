from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from lfa_toolbox.core.fis.singleton_fis import SingletonFIS
from lfa_toolbox.view.fis_viewer import FISViewer
from lfa_toolbox.view.mf_viewer import MembershipFunctionViewer


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

    def plot_fuzzy_sets(self, variable=None, **kwargs):
        """Plot membership functions for variables.

        Args:
            variable: None, str, or list[str]. If None, plot all variables; if str,
                plot only that variable; if list, plot each listed variable.
            **kwargs: Extra options passed to the membership function viewer.
        """
        var_list = self._to_var_list(variable)

        # Helper to fetch a LinguisticVariable by name
        def get_lv(name):
            lv = next((v for v in self.variables_ if v.name == name), None)
            if lv is None:
                raise ValueError(f"Variable '{name}' not found in self.variables_.")
            return lv

        if var_list is None:
            # all variables
            for lv in self.variables_:
                fig, ax = plt.subplots()
                ax.set_title(lv.name)
                for label, mf in lv.ling_values.items():
                    MembershipFunctionViewer(mf, ax=ax, label=label, **kwargs)
                ax.legend()
                plt.show()
        else:
            for name in var_list:
                lv = get_lv(name)
                fig, ax = plt.subplots()
                ax.set_title(lv.name)
                for label, mf in lv.ling_values.items():
                    MembershipFunctionViewer(mf, ax=ax, label=label, **kwargs)
                ax.legend()
                plt.show()

    def plot_fuzzification(self, sample, variable=None, **kwargs):
        """Plot membership functions and overlay fuzzification for a sample.

        Args:
            sample: Array-like, dict, or pandas Series holding crisp inputs.
            variable: None, str, or list[str]. If None, plot all input variables
                present in the sample; if str, only that variable; if list, each listed variable.
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

        var_list = self._to_var_list(variable)

        # Build iterable of linguistic variables to plot (input vars only)
        def is_output(lv):
            # Heuristic: skip classic output names; adapt if you store IO flags
            return lv.name.upper() in {"OUT", "TARGET"}  # adjust if needed

        if var_list is None:
            lvs = [lv for lv in self.variables_ if not is_output(lv) and lv.name in sample_dict]
        else:
            # Validate names and keep only those present in the sample
            name_set = set(var_list)
            lvs = []
            for name in name_set:
                lv = next((v for v in self.variables_ if v.name == name), None)
                if lv is None:
                    raise ValueError(f"Variable '{name}' not found in self.variables_.")
                if is_output(lv):
                    # Skip output variable in fuzzification; it has no crisp input
                    continue
                if lv.name not in sample_dict:
                    # silently skip if sample lacks this feature
                    continue
                lvs.append(lv)

        # Plot
        for lv in lvs:
            crisp_value = sample_dict[lv.name]
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

    def plot_aggregated_output(self, input_sample, figsize=(12, 10)):
        """Visualize the aggregated fuzzy output for an input sample.

        Uses a ``SingletonFIS`` to mirror the C++ singleton-based defuzzification.

        Args:
            input_sample: Single sample of crisp input values.
            figsize: Matplotlib figure size.
        """

        # Build a mapping for the input values.
        # input_dict = {name: value for name, value in zip(self.feature_names_in_, input_sample, strict=False)}

        # Retrieve the output linguistic variable
        output_lv = next(
            (lv for lv in self.variables_ if lv.name.upper() == self.target_name_in_.upper()),
            None,
        )

        if output_lv is None:
            raise ValueError("Output linguistic variable 'OUT' not found in self.variables_.")

        # Create a SingletonFIS instance using the learned rules and default rule.
        fis = SingletonFIS(
            rules=self.rules_,
            default_rule=(self.default_rules_[0] if self.default_rules_ else None),
        )

        # result = fis.predict(input_dict)
        # result_cpp = self._predict(input_sample)

        # if not np.isclose(float(result.get(self.target_name_in_)), float(result_cpp[0])):
        #    raise ValueError(
        #        f"Python and C++ defuzzification results do not match: {result} vs. {result_cpp}"
        #    )
        # Show the aggregated fuzzy output via FISViewer.
        fisv = FISViewer(fis, figsize=figsize)
        fisv.show()

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
