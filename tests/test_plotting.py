import matplotlib
import numpy as np
import pandas as pd
import pytest

# Use a non-interactive backend for headless testing environments.
matplotlib.use("Agg", force=True)

from fuzzycocopython import FuzzyCocoRegressor


@pytest.fixture(scope="module")
def trained_regressor():
    rng = np.random.default_rng(24)
    X = pd.DataFrame(rng.random((16, 3)), columns=["A", "B", "C"])
    y = pd.Series(rng.random(16), name="Target")

    model = FuzzyCocoRegressor(random_state=0)
    model.fit(X, y)
    return model, X


def test_plot_aggregated_output_runs_without_errors(monkeypatch):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((12, 3)), columns=["A", "B", "C"])
    y = pd.Series(rng.random(12), name="Target")

    model = FuzzyCocoRegressor(random_state=0)
    model.fit(X, y)

    show_calls = []

    def fake_show(self):
        show_calls.append(True)

    monkeypatch.setattr("fuzzycocopython.fuzzycoco_plot_mixin.FISViewer.show", fake_show)

    # Should run without raising and return None (implicit return).
    result = model.plot_aggregated_output(X.iloc[0])

    assert result is None
    assert show_calls, "Expected FISViewer.show to be called."


def test_plot_fuzzy_sets_uses_membership_viewer(monkeypatch, trained_regressor):
    model, _ = trained_regressor

    viewer_calls = []

    class DummyViewer:
        def __init__(self, mf, ax=None, label=None, **_kwargs):
            viewer_calls.append((label, mf))

        def fuzzify(self, _value):
            pass

    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    monkeypatch.setattr("fuzzycocopython.fuzzycoco_plot_mixin.MembershipFunctionViewer", DummyViewer)

    result = model.plot_fuzzy_sets()

    assert result is None
    assert viewer_calls, "Expected MembershipFunctionViewer to be instantiated."


def test_plot_fuzzification_calls_fuzzify(monkeypatch, trained_regressor):
    model, X = trained_regressor

    fuzzify_calls = []

    class DummyViewer:
        def __init__(self, mf, ax=None, label=None, **_kwargs):
            self.label = label

        def fuzzify(self, crisp_value):
            fuzzify_calls.append(crisp_value)

    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    monkeypatch.setattr("fuzzycocopython.fuzzycoco_plot_mixin.MembershipFunctionViewer", DummyViewer)

    result = model.plot_fuzzification(X.iloc[0])

    assert result is None
    assert fuzzify_calls, "Expected fuzzify to be invoked for at least one membership function."


def test_plot_rule_activations_runs_without_errors(monkeypatch, trained_regressor):
    model, X = trained_regressor

    show_calls = []
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: show_calls.append(True))

    sample = X.iloc[0].to_numpy(dtype=float)
    result = model.plot_rule_activations(sample)

    assert result is None
    assert show_calls, "Expected matplotlib.pyplot.show to be called."
