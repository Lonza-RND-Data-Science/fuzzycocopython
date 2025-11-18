import copy

import numpy as np
import pandas as pd
import pytest

from fuzzycocopython import FuzzyCocoClassifier, FuzzyCocoRegressor, load_model, save_model
from fuzzycocopython.fuzzycoco_base import _FuzzyCocoBase


def test_classifier_with_pandas(tmp_path):
    # Generate a small classification dataset
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.random((20, 3)), columns=["A", "B", "C"])
    y = pd.Series(rng.integers(0, 2, size=20), name="Target")

    model = FuzzyCocoClassifier(random_state=123)
    model.fit(X, y)
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == len(y)
    assert 0.0 <= score <= 1.0


def test_classifier_with_numpy_no_names(tmp_path):
    # Generate a small classification dataset
    rng = np.random.default_rng(321)
    X = rng.random((20, 3))
    y = rng.integers(0, 2, size=20)
    model = FuzzyCocoClassifier(random_state=123)
    model.fit(X, y)
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == len(y)
    assert 0.0 <= score <= 1.0


def test_classifier_with_numpy_with_names(tmp_path):
    # Generate a small classification dataset
    rng = np.random.default_rng(999)
    X = rng.random((20, 3))
    y = rng.integers(0, 2, size=20)
    feature_names = ["Feat1", "Feat2", "Feat3"]
    model = FuzzyCocoClassifier(random_state=123)
    model.fit(
        X,
        y,
        feature_names=feature_names,
        target_name="Class",
    )
    preds = model.predict(X)
    score = model.score(X, y)
    # model.plot_aggregated_output(X[1])

    assert len(preds) == len(y)
    assert 0.0 <= score <= 1.0


def test_regressor_with_pandas(tmp_path):
    # Generate a small regression dataset
    rng = np.random.default_rng(456)
    X = pd.DataFrame(rng.random((20, 3)), columns=["A", "B", "C"])
    y = pd.Series(rng.random(20), name="Target")
    model = FuzzyCocoRegressor(random_state=123)
    model.fit(X, y)
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == len(y)
    assert isinstance(score, float)


def test_regressor_with_numpy_no_names(tmp_path):
    rng = np.random.default_rng(654)
    X = rng.random((20, 3))
    y = rng.random(20)
    model = FuzzyCocoRegressor(random_state=123)
    model.fit(X, y)
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == 20
    assert isinstance(score, float)


def test_regressor_with_numpy_with_names(tmp_path):
    rng = np.random.default_rng(777)
    X = rng.random((20, 3))
    y = rng.random(20)
    feature_names = ["Var1", "Var2", "Var3"]
    model = FuzzyCocoRegressor(random_state=123)
    model.fit(
        X,
        y,
        feature_names=feature_names,
        target_name="Y",
    )
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == 20
    assert isinstance(score, float)


def _generate_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.random((30, 4))
    y_class = rng.integers(0, 3, size=30)
    y_reg = rng.random(30)
    return X, y_class, y_reg


def _generate_multi_output(seed: int = 0, outputs: int = 2):
    rng = np.random.default_rng(seed)
    X = rng.random((30, 4))
    y_class = rng.integers(0, 3, size=(30, outputs))
    y_reg = rng.random((30, outputs))
    return X, y_class, y_reg


def test_classifier_save_and_load(tmp_path):
    X, y_class, _ = _generate_dataset()
    model = FuzzyCocoClassifier(random_state=42)
    model.fit(X, y_class)

    path = tmp_path / "classifier.pkl"
    model.save(path)

    loaded = FuzzyCocoClassifier.load(path)
    np.testing.assert_allclose(model.predict(X), loaded.predict(X))


def test_module_level_save_load(tmp_path):
    X, _, y_reg = _generate_dataset(seed=1)
    reg = FuzzyCocoRegressor(random_state=7)
    reg.fit(X, y_reg)

    path = tmp_path / "regressor.pkl"
    save_model(reg, path)
    loaded = load_model(path)

    assert isinstance(loaded, FuzzyCocoRegressor)
    original = reg.predict(X)
    loaded_pred = loaded.predict(X)
    np.testing.assert_allclose(original, loaded_pred)


def test_classifier_multi_output():
    X, y_multi, _ = _generate_multi_output(seed=23, outputs=2)
    clf = FuzzyCocoClassifier(random_state=5)
    clf.fit(X, y_multi)
    preds = clf.predict(X)
    assert preds.shape == y_multi.shape
    assert clf.n_outputs_ == y_multi.shape[1]
    thresholds = clf._fuzzy_params_.fitness_params.output_vars_defuzz_thresholds
    assert len(thresholds) == y_multi.shape[1]
    assert isinstance(clf.classes_, list)
    assert len(clf.classes_) == y_multi.shape[1]
    for idx, column_classes in enumerate(clf.classes_):
        np.testing.assert_array_equal(
            np.sort(np.unique(y_multi[:, idx])),
            np.sort(column_classes),
        )


def test_regressor_multi_output():
    X, _, y_multi = _generate_multi_output(seed=29, outputs=3)
    reg = FuzzyCocoRegressor(random_state=11)
    reg.fit(X, y_multi)
    preds = reg.predict(X)
    assert preds.shape == y_multi.shape
    assert reg.n_outputs_ == y_multi.shape[1]
    thresholds = reg._fuzzy_params_.fitness_params.output_vars_defuzz_thresholds
    assert len(thresholds) == y_multi.shape[1]


def test_rules_activations_and_stats():
    X, y_class, _ = _generate_dataset(seed=5)
    model = FuzzyCocoClassifier(random_state=1)
    model.fit(X, y_class)

    stats, matrix = model.rules_stat_activations(X, return_matrix=True, sort_by_impact=False)

    # Expected reporting columns provided by the estimator
    expected_columns = {
        "mean",
        "std",
        "min",
        "max",
        "usage_rate",
        "usage_rate_pct",
        "importance_pct",
        "impact_pct",
    }
    assert expected_columns.issubset(stats.columns)

    # rule activation matrix aligns with samples and reported rules
    assert matrix.shape[0] == X.shape[0]
    assert matrix.shape[1] == stats.shape[0]


def test_rules_stats_after_loading(tmp_path):
    X, y_class, _ = _generate_dataset(seed=11)
    model = FuzzyCocoClassifier(random_state=123)
    model.fit(X, y_class)

    path = tmp_path / "model.joblib"
    model.save(path)

    loaded = FuzzyCocoClassifier.load(path)
    stats, matrix = loaded.rules_stat_activations(X, return_matrix=True)

    assert stats.shape[0] == matrix.shape[1]
    assert matrix.shape[0] == X.shape[0]
    assert not stats.empty

    # sampling rules_activations for a single sample is consistent with the matrix
    single = model.rules_activations(X[0])
    assert single.shape == (matrix.shape[1],)
    np.testing.assert_allclose(single, matrix[0], rtol=1e-6, atol=1e-6)


def test_describe_contains_fuzzy_system():
    X, y_class, _ = _generate_dataset(seed=3)
    clf = FuzzyCocoClassifier(random_state=42)
    clf.fit(X, y_class)

    description = clf.describe()
    assert isinstance(description, dict)
    assert "fuzzy_system" in description
    assert description["fuzzy_system"]


def test_fit_rejects_mismatched_feature_names():
    X, y_class, _ = _generate_dataset(seed=7)
    clf = FuzzyCocoClassifier(random_state=123)

    with pytest.raises(ValueError, match="feature_names length"):
        clf.fit(X, y_class, feature_names=["only_one_name"])


def test_rules_activations_dataframe_missing_columns():
    X, y_class, _ = _generate_dataset(seed=11)
    columns = ["c1", "c2", "c3", "c4"]
    clf = FuzzyCocoClassifier(random_state=0)
    clf.fit(X, y_class, feature_names=columns)

    sample = pd.DataFrame([X[0]], columns=["c1", "c2", "c3", "extra"])
    with pytest.raises(ValueError, match="Missing features"):
        clf.rules_activations(sample)


def test_rules_stat_activations_empty_input_raises():
    X, y_class, _ = _generate_dataset(seed=13)
    clf = FuzzyCocoClassifier(random_state=1)
    clf.fit(X, y_class)

    empty = np.empty((0, clf.n_features_in_))
    with pytest.raises(ValueError, match="0 sample"):
        clf.rules_stat_activations(empty)


def test_rules_activations_exposes_default_rule_levels():
    class DummyModel:
        def __init__(self, values):
            self._values = values

        def rules_fire_from_values(self, sample):
            return list(self._values)

    description = {
        "fuzzy_system": {
            "variables": {
                "input": {"feature": {"feature.low": 0.0}},
                "output": {
                    "target": {"target.low": 0.0, "target.high": 1.0},
                    "other": {"other.low": -1.0, "other.high": 1.0},
                },
            },
            "rules": {
                "rule_1": {
                    "antecedents": {"feature": {"feature.low": 1.0}},
                    "consequents": {"target": {"target.high": 1.0}},
                },
                "rule_2": {
                    "antecedents": {"feature": {"feature.low": 1.0}},
                    "consequents": {"target": {"target.low": 0.0}},
                },
                "rule_3": {
                    "antecedents": {"feature": {"feature.low": 1.0}},
                    "consequents": {"other": {"other.high": 1.0}},
                },
            },
            "default_rules": {"target": "target.low", "other": "other.low"},
        }
    }

    clf = FuzzyCocoClassifier()
    clf.description_ = copy.deepcopy(description)
    clf._fuzzy_system_dict_ = copy.deepcopy(description["fuzzy_system"])
    clf.feature_names_in_ = ["feature"]
    clf.n_features_in_ = 1
    clf.is_fitted_ = True
    sentinel = np.finfo(np.float64).min
    clf.model_ = DummyModel([0.25, 0.8, sentinel])

    activations = clf.rules_activations([0.0])
    assert isinstance(activations, np.ndarray)
    assert activations.shape == (3,)
    assert activations.default_rules == {
        "target": pytest.approx(0.2),
        "other": pytest.approx(0.0),
    }


def test_rules_activations_uses_description_mapping_for_defaults():
    class DummyModel:
        def __init__(self, values):
            self._values = values

        def rules_fire_from_values(self, sample):
            return list(self._values)

    clf = FuzzyCocoClassifier()
    description = _sample_description_single_output()
    _prime_model_with_description(clf, description)
    clf.feature_names_in_ = ["feature1"]
    clf.n_features_in_ = 1
    clf.is_fitted_ = True
    clf.model_ = DummyModel([0.4])

    activations = clf.rules_activations([0.1])
    assert activations.default_rules == {"target": pytest.approx(0.6)}


def test_rules_stat_activations_matrix_carries_default_rules():
    class DummyModel:
        def rules_fire_from_values(self, sample):
            value = float(sample[0])
            return [value, 1.0 - value]

    description = {
        "fuzzy_system": {
            "variables": {
                "input": {"feature": {"feature.low": 0.0, "feature.high": 1.0}},
                "output": {"target": {"target.low": 0.0, "target.high": 1.0}},
            },
            "rules": {
                "rule_1": {
                    "antecedents": {"feature": {"feature.low": 1.0}},
                    "consequents": {"target": {"target.high": 1.0}},
                },
                "rule_2": {
                    "antecedents": {"feature": {"feature.high": 1.0}},
                    "consequents": {"target": {"target.low": 0.0}},
                },
            },
            "default_rules": {"target": "target.low"},
        }
    }

    clf = FuzzyCocoClassifier()
    clf.description_ = copy.deepcopy(description)
    clf._fuzzy_system_dict_ = copy.deepcopy(description["fuzzy_system"])
    clf.is_fitted_ = True
    clf.model_ = DummyModel()
    clf.feature_names_in_ = ["feature"]
    clf.n_features_in_ = 1
    clf.rules_ = ["rule_1", "rule_2"]

    X = np.array([[0.2], [0.8]])
    stats, matrix = clf.rules_stat_activations(X, return_matrix=True, sort_by_impact=False)

    assert stats.shape[0] == matrix.shape[1]
    assert isinstance(matrix, np.ndarray)
    assert hasattr(matrix, "default_rules")
    assert matrix.default_rules is not None
    assert len(matrix.default_rules) == 2
    assert matrix.default_rules[0]["target"] == pytest.approx(0.2)
    assert matrix.default_rules[1]["target"] == pytest.approx(0.2)
    assert "ELSE target is Low" in stats.index
    default_row = stats.loc["ELSE target is Low"]
    assert pytest.approx(default_row["mean"], rel=1e-6) == 0.2


def test_classifier_load_type_guard(tmp_path):
    X, _, y_reg = _generate_dataset(seed=17)
    reg = FuzzyCocoRegressor(random_state=2)
    reg.fit(X, y_reg)

    path = tmp_path / "reg.joblib"
    reg.save(path)

    with pytest.raises(TypeError, match="Expected instance of FuzzyCocoClassifier"):
        FuzzyCocoClassifier.load(path)


def test_regressor_custom_scoring():
    X, _, y_reg = _generate_dataset(seed=19)
    reg = FuzzyCocoRegressor(random_state=3)
    reg.fit(X, y_reg)

    score = _FuzzyCocoBase.score(reg, X, y_reg, scoring="neg_mean_squared_error")
    assert isinstance(score, float)


def _sample_description_single_output():
    return {
        "fuzzy_system": {
            "variables": {
                "input": {"feature1": {"feature1.low": 0.0, "feature1.high": 1.0}},
                "output": {"target": {"target.low": 0.0, "target.high": 1.0}},
            },
            "rules": {
                "rule_1": {
                    "antecedents": {"feature1": {"feature1.low": 1.0}},
                    "consequents": {"target": {"target.high": 1.0}},
                }
            },
            "default_rules": {"target": "target.low"},
        },
        "defuzz_thresholds": {"target": 0.4},
    }


def _sample_description_multi_output():
    return {
        "fuzzy_system": {
            "variables": {
                "input": {
                    "feature1": {"feature1.low": 0.0, "feature1.high": 1.0},
                },
                "output": {
                    "target": {"target.low": 0.0, "target.high": 1.0},
                    "other": {"other.low": -1.0, "other.high": 2.0},
                },
            },
            "rules": {
                "rule_1": {
                    "antecedents": {"feature1": {"feature1.low": 1.0}},
                    "consequents": {
                        "target": {"target.high": 1.0},
                        "other": {"other.low": 0.5},
                    },
                }
            },
            "default_rules": {"target": "target.low", "other": "other.high"},
        },
        "defuzz_thresholds": {"target": 0.5, "other": 0.7},
    }


def _prime_model_with_description(model, description):
    model.description_ = copy.deepcopy(description)
    model.is_fitted_ = True
    outputs = list(description["fuzzy_system"]["variables"]["output"].keys())
    model.target_name_in_ = outputs[0]
    model.target_names_in_ = outputs
    model.n_outputs_ = len(outputs)
    model._fuzzy_system_dict_ = copy.deepcopy(description["fuzzy_system"])


def test_set_target_names_single_output():
    model = FuzzyCocoRegressor()
    description = _sample_description_single_output()
    _prime_model_with_description(model, description)

    model.set_target_names("Outcome")

    assert model.target_name_in_ == "Outcome"
    assert model.target_names_in_ == ["Outcome"]
    outputs = model.description_["fuzzy_system"]["variables"]["output"]
    assert list(outputs.keys()) == ["Outcome"]
    assert set(outputs["Outcome"].keys()) == {"Outcome.low", "Outcome.high"}
    consequents = model.description_["fuzzy_system"]["rules"]["rule_1"]["consequents"]
    assert list(consequents.keys()) == ["Outcome"]
    assert set(consequents["Outcome"].keys()) == {"Outcome.high"}
    defaults = model.description_["fuzzy_system"]["default_rules"]
    assert defaults == {"Outcome": "Outcome.low"}
    thresholds = model.description_["defuzz_thresholds"]
    assert thresholds == {"Outcome": 0.4}


def test_set_target_names_multi_sequence():
    model = FuzzyCocoRegressor()
    description = _sample_description_multi_output()
    _prime_model_with_description(model, description)

    model.set_target_names(["T", "O"])

    assert model.target_name_in_ == "T"
    assert model.target_names_in_ == ["T", "O"]
    outputs = model.description_["fuzzy_system"]["variables"]["output"]
    assert list(outputs.keys()) == ["T", "O"]
    assert "T.high" in outputs["T"]
    assert "O.low" in outputs["O"]
    rule = model.description_["fuzzy_system"]["rules"]["rule_1"]["consequents"]
    assert set(rule.keys()) == {"T", "O"}
    assert set(rule["T"].keys()) == {"T.high"}
    assert set(rule["O"].keys()) == {"O.low"}
    defaults = model.description_["fuzzy_system"]["default_rules"]
    assert defaults == {"T": "T.low", "O": "O.high"}
    thresholds = model.description_["defuzz_thresholds"]
    assert thresholds == {"T": 0.5, "O": 0.7}


def test_set_target_names_duplicate_raises():
    model = FuzzyCocoRegressor()
    description = _sample_description_multi_output()
    _prime_model_with_description(model, description)

    with pytest.raises(ValueError, match="unique"):
        model.set_target_names({"target": "other"})


def test_set_target_names_whitespace_raises():
    model = FuzzyCocoRegressor()
    description = _sample_description_single_output()
    _prime_model_with_description(model, description)

    with pytest.raises(ValueError, match="must not contain"):
        model.set_target_names("Bad Name")
