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
