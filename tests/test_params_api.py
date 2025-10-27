import numpy as np
import pytest
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


def test_metrics_weights_zero_fill():
    X, y = _toy_regression(seed=8)
    custom = {"accuracy": 0.7, "rmse": 0.3}
    reg = FuzzyCocoRegressor(nb_rules=4, metrics_weights=custom, random_state=5)
    reg.fit(X, y)

    params_desc = reg._fuzzy_params_.describe()
    weights = params_desc["fitness_params"]["metrics_weights"]
    assert weights["accuracy"] == pytest.approx(0.7)
    assert weights["rmse"] == pytest.approx(0.3)

    zero_expected = [
        "sensitivity",
        "specificity",
        "ppv",
        "rrse",
        "rae",
        "mse",
        "distanceThreshold",
        "distanceMinThreshold",
        "nb_vars",
        "overLearn",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    ]
    for key in zero_expected:
        assert weights[key] == 0.0


def test_metrics_weights_accepts_integer_values():
    X, y = _toy_regression(seed=12)
    reg = FuzzyCocoRegressor(
        nb_rules=4,
        metrics_weights={"rmse": 2},
        random_state=3,
    )
    reg.fit(X, y)

    params_desc = reg._fuzzy_params_.describe()
    weights = params_desc["fitness_params"]["metrics_weights"]
    assert weights["rmse"] == pytest.approx(2.0)
    assert weights["accuracy"] == 0.0


def test_invalid_metrics_key_raises():
    X, y = _toy_regression(seed=9)
    reg = FuzzyCocoRegressor(metrics_weights={"accuracy": 0.6, "unknown_metric": 0.4})
    with pytest.raises(ValueError, match="Unknown metrics_weights keys: unknown_metric"):
        reg.fit(X, y)


def test_invalid_estimator_param_raises():
    reg = FuzzyCocoRegressor()
    with pytest.raises(ValueError, match="Invalid parameter"):
        reg.set_params(not_a_real_param=123)


def test_python_cpp_params_alignment():
    X, y = _toy_regression(seed=11)
    reg = FuzzyCocoRegressor(
        nb_rules=9,
        nb_max_var_per_rule=4,
        max_generations=75,
        max_fitness=0.95,
        nb_cooperators=3,
        influence_rules_initial_population=True,
        influence_evolving_ratio=0.55,
        nb_sets_in=3,
        nb_sets_out=4,
        pop_size_rules=110,
        pop_size_mfs=90,
        elite_size_rules=6,
        elite_size_mfs=7,
        cx_prob_rules=0.52,
        cx_prob_mfs=0.88,
        mut_flip_genome_rules=0.34,
        mut_flip_genome_mfs=0.22,
        mut_flip_bit_rules=0.015,
        mut_flip_bit_mfs=0.025,
        nb_bits_pos_in=6,
        nb_bits_pos_out=7,
        nb_bits_vars_in=5,
        nb_bits_vars_out=4,
        nb_bits_sets_in=3,
        nb_bits_sets_out=2,
        threshold=0.42,
        metrics_weights={"rmse": 1.0},
        random_state=13,
    )
    reg.fit(X, y)

    desc = reg._fuzzy_params_.describe()

    gp = desc["global_params"]
    assert gp["nb_rules"] == reg.nb_rules
    assert gp["nb_max_var_per_rule"] == reg.nb_max_var_per_rule
    assert gp["max_generations"] == reg.max_generations
    assert gp["max_fitness"] == pytest.approx(reg.max_fitness)
    assert gp["nb_cooperators"] == reg.nb_cooperators
    assert bool(gp["influence_rules_initial_population"]) == reg.influence_rules_initial_population
    assert gp["influence_evolving_ratio"] == pytest.approx(reg.influence_evolving_ratio)

    in_vars = desc["input_vars_params"]
    assert in_vars["nb_sets"] == reg.nb_sets_in
    assert in_vars["nb_bits_pos"] == reg.nb_bits_pos_in
    assert in_vars["nb_bits_vars"] == reg.nb_bits_vars_in
    assert in_vars["nb_bits_sets"] == reg.nb_bits_sets_in

    out_vars = desc["output_vars_params"]
    assert out_vars["nb_sets"] == reg.nb_sets_out
    assert out_vars["nb_bits_pos"] == reg.nb_bits_pos_out
    assert out_vars["nb_bits_vars"] == reg.nb_bits_vars_out
    assert out_vars["nb_bits_sets"] == reg.nb_bits_sets_out

    rules_params = desc["rules_params"]
    assert rules_params["pop_size"] == reg.pop_size_rules
    assert rules_params["elite_size"] == reg.elite_size_rules
    assert rules_params["cx_prob"] == pytest.approx(reg.cx_prob_rules)
    assert rules_params["mut_flip_genome"] == pytest.approx(reg.mut_flip_genome_rules)
    assert rules_params["mut_flip_bit"] == pytest.approx(reg.mut_flip_bit_rules)

    mfs_params = desc["mfs_params"]
    assert mfs_params["pop_size"] == reg.pop_size_mfs
    assert mfs_params["elite_size"] == reg.elite_size_mfs
    assert mfs_params["cx_prob"] == pytest.approx(reg.cx_prob_mfs)
    assert mfs_params["mut_flip_genome"] == pytest.approx(reg.mut_flip_genome_mfs)
    assert mfs_params["mut_flip_bit"] == pytest.approx(reg.mut_flip_bit_mfs)

    fitness_params = desc["fitness_params"]
    thresholds = fitness_params["output_vars_defuzz_thresholds"]
    assert len(thresholds) == 1
    assert pytest.approx(reg.threshold) == thresholds["1"]
    assert fitness_params["metrics_weights"]["rmse"] == pytest.approx(1.0)
    for key, value in fitness_params["metrics_weights"].items():
        if key != "rmse":
            assert value == 0.0
