import pandas as pd
import pytest

from fuzzycocopython.utils import (
    generate_generic_labels,
    parse_fuzzy_system_from_description,
    to_linguistic_components,
    to_tables_components,
    to_views_components,
)


@pytest.fixture(autouse=True)
def configure_matplotlib_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "mpl-cache"
    cache_dir.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(cache_dir))


@pytest.fixture
def sample_description():
    return {
        "fuzzy_system": {
            "variables": {
                "input": {
                    "feature1": {"feature1.low": 0.0, "feature1.high": 1.0},
                },
                "output": {
                    "target": {"target.low": 0.0, "target.high": 1.0},
                },
            },
            "rules": {
                "rule_1": {
                    "antecedents": {"feature1": {"feature1.low": 1.0}},
                    "consequents": {"target": {"target.high": 1.0}},
                }
            },
            "default_rules": {"target": "target.low"},
        }
    }


def test_generate_generic_labels_progression():
    assert generate_generic_labels(1) == ["Medium"]
    assert generate_generic_labels(3) == ["Low", "Medium", "High"]
    six_labels = generate_generic_labels(6)
    assert six_labels[2] == "Slightly Low"
    assert six_labels[-1] == "Very High"
    assert generate_generic_labels(8)[-1] == "Set 8"


def test_to_linguistic_components_assigns_generic_labels(sample_description):
    variables, rules, defaults = parse_fuzzy_system_from_description(sample_description)
    linguistic_variables, fuzzy_rules, default_rules = to_linguistic_components(variables, rules, defaults)

    names = sorted(lv.name for lv in linguistic_variables)
    assert names == ["feature1", "target"]

    # Antecedent/Consequent labels should use the generated names
    antecedent = fuzzy_rules[0].antecedents[0]
    consequent = fuzzy_rules[0].consequents[0]
    assert antecedent.lv_value == "Low"
    assert consequent.lv_value == "High"
    assert default_rules[0].consequents[0].lv_value == "Low"


def test_views_components_render_human_strings(sample_description):
    variables, rules, defaults = parse_fuzzy_system_from_description(sample_description)
    variables_view, rules_view, defaults_view = to_views_components(variables, rules, defaults)

    assert variables_view["feature1"] == [
        "Low = 0 (from feature1.low)",
        "High = 1 (from feature1.high)",
    ]
    assert rules_view == ["RULE_1: IF feature1 is Low (0) THEN target is High (1)"]
    assert defaults_view == ["DEFAULT: target is Low (0)"]


def test_tables_components_builds_expected_dataframes(sample_description):
    variables, rules, defaults = parse_fuzzy_system_from_description(sample_description)
    vars_df, rules_df = to_tables_components(variables, rules, defaults)

    # Variables table should list every label with its origin and role
    expected_vars = pd.DataFrame(
        [
            {"io": "input", "var": "feature1", "label": "Low", "position": 0.0, "orig_set": "feature1.low"},
            {"io": "input", "var": "feature1", "label": "High", "position": 1.0, "orig_set": "feature1.high"},
            {"io": "output", "var": "target", "label": "Low", "position": 0.0, "orig_set": "target.low"},
            {"io": "output", "var": "target", "label": "High", "position": 1.0, "orig_set": "target.high"},
        ]
    )
    pd.testing.assert_frame_equal(vars_df, expected_vars)

    # Rules table must include antecedent, consequent, and default entries with the mapped labels
    expected_rules = pd.DataFrame(
        [
            {
                "rule": "default",
                "role": "default",
                "io": "output",
                "var": "target",
                "label": "Low",
                "position": 0.0,
                "orig_set": "target.low",
            },
            {
                "rule": "rule_1",
                "role": "antecedent",
                "io": "input",
                "var": "feature1",
                "label": "Low",
                "position": 0.0,
                "orig_set": "feature1.low",
            },
            {
                "rule": "rule_1",
                "role": "consequent",
                "io": "output",
                "var": "target",
                "label": "High",
                "position": 1.0,
                "orig_set": "target.high",
            },
        ]
    )
    pd.testing.assert_frame_equal(rules_df, expected_rules)
