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
    return model, X, y


def test_plot_aggregated_output_runs_without_errors(monkeypatch):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((12, 3)), columns=["A", "B", "C"])
    y = pd.Series(rng.random(12), name="Target")

    model = FuzzyCocoRegressor(random_state=0)
    model.fit(X, y)

    show_calls = []
    captured_defuzz = []

    def fake_show(self):
        outputs = list(self._FISViewer__fis.last_defuzzified_outputs.values())
        captured_defuzz.append(outputs[0])
        show_calls.append(True)

    monkeypatch.setattr("fuzzycocopython.fuzzycoco_plot_mixin.FISViewer.show", fake_show)

    baseline = model.plot_aggregated_output(X.iloc[0])
    scaled = model.plot_aggregated_output(X.iloc[0], target_rescale=1.5)

    assert baseline is None and scaled is None
    assert len(show_calls) == 2
    assert len(captured_defuzz) == 2
    assert pytest.approx(captured_defuzz[1]) == captured_defuzz[0] * 1.5


def test_plot_fuzzy_sets_rescales_target(monkeypatch, trained_regressor):
    model, _, _ = trained_regressor

    target_lv = next(lv for lv in model.variables_ if lv.name == model.target_name_in_)
    baseline_map = {label: float(np.max(mf.in_values)) for label, mf in target_lv.ling_values.items()}

    scaled_map = {}

    class DummyViewer:
        def __init__(self, mf, ax=None, label=None, **_kwargs):
            scaled_map[label] = float(np.max(mf.in_values))

        def fuzzify(self, _value):
            pass

    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    monkeypatch.setattr("fuzzycocopython.fuzzycoco_plot_mixin.MembershipFunctionViewer", DummyViewer)

    result = model.plot_fuzzy_sets(variable=model.target_name_in_, target_rescale=2.0)

    assert result is None
    assert scaled_map, "Expected MembershipFunctionViewer for target variable to be called."
    for label, baseline_val in baseline_map.items():
        if label in scaled_map:
            assert pytest.approx(scaled_map[label]) == baseline_val * 2.0


def test_plot_fuzzification_rescales_target_sample(monkeypatch, trained_regressor):
    model, X, y = trained_regressor

    fuzzify_calls = []

    class DummyViewer:
        def __init__(self, mf, ax=None, label=None, **_kwargs):
            self._mf = mf

        def fuzzify(self, crisp_value):
            fuzzify_calls.append(crisp_value)

    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    monkeypatch.setattr("fuzzycocopython.fuzzycoco_plot_mixin.MembershipFunctionViewer", DummyViewer)

    sample = X.iloc[0].to_dict()
    sample[model.target_name_in_] = float(y.iloc[0])

    result = model.plot_fuzzification(sample, variable=model.target_name_in_, target_rescale=2.0)

    assert result is None
    assert fuzzify_calls, "Expected fuzzify to be invoked for at least one membership function."
    scaled_value = sample[model.target_name_in_] * 2.0
    assert any(pytest.approx(scaled_value) == call for call in fuzzify_calls)


def test_plot_rule_activations_runs_without_errors(monkeypatch, trained_regressor):
    model, X, _ = trained_regressor

    show_calls = []
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: show_calls.append(True))

    sample = X.iloc[0].to_numpy(dtype=float)
    result = model.plot_rule_activations(sample)

    assert result is None
    assert show_calls, "Expected matplotlib.pyplot.show to be called."
