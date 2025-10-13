import numpy as np
from sklearn.model_selection import GridSearchCV

from fuzzycocopython import FuzzyCocoClassifier, FuzzyCocoRegressor


def _toy_classification(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((20, 3))
    y = rng.integers(0, 2, size=20)
    return X, y


def _toy_regression(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((20, 2))
    y = rng.random(20)
    return X, y


def test_get_set_params_roundtrip():
    clf = FuzzyCocoClassifier()
    params = clf.get_params()

    assert params["nb_rules"] == 5
    assert params["mut_flip_bit_rules"] == 0.01

    clf.set_params(nb_rules=8, mut_flip_bit_rules=0.05)
    assert clf.nb_rules == 8
    assert clf.mut_flip_bit_rules == 0.05


def test_default_metrics_weights():
    clf = FuzzyCocoClassifier()
    reg = FuzzyCocoRegressor()

    assert clf.metrics_weights == {"accuracy": 1.0}
    assert reg.metrics_weights == {"rmse": 1.0}

    clf.set_params(metrics_weights=None)
    assert clf.metrics_weights is None


def test_refit_after_param_update_rebuilds_params():
    X, y = _toy_classification(seed=3)
    clf = FuzzyCocoClassifier(random_state=0, nb_rules=6)
    clf.fit(X, y)
    assert clf._fuzzy_params_.global_params.nb_rules == 6

    clf.set_params(nb_rules=9)
    clf.fit(X, y)
    assert clf._fuzzy_params_.global_params.nb_rules == 9


def test_grid_search_cv_compatibility():
    X, y = _toy_regression(seed=4)
    reg = FuzzyCocoRegressor(random_state=0)

    grid = GridSearchCV(
        reg,
        param_grid={
            "nb_rules": [4, 6],
            "nb_sets_in": [2, 3],
        },
        cv=2,
        n_jobs=1,
    )

    grid.fit(X, y)
    assert isinstance(grid.best_estimator_, FuzzyCocoRegressor)
