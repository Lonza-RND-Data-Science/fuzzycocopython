import numpy as np
import pandas as pd
from lfa_toolbox.core.lv.linguistic_variable import LinguisticVariable
from lfa_toolbox.core.mf.lin_piece_wise_mf import LeftShoulderMF, RightShoulderMF, TriangularMF
from lfa_toolbox.core.mf.singleton_mf import SingletonMF
from lfa_toolbox.core.rules.default_fuzzy_rule import DefaultFuzzyRule
from lfa_toolbox.core.rules.fuzzy_rule import FuzzyRule
from lfa_toolbox.core.rules.fuzzy_rule_element import Antecedent, Consequent

from ._fuzzycoco_core import FuzzyCocoParams

_METRICS_KEYS = (
    "sensitivity",
    "specificity",
    "accuracy",
    "ppv",
    "rmse",
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
)


def create_input_linguistic_variable(var_name, set_items):
    set_items = sorted(set_items, key=lambda s: s.get("position", 0))
    n = len(set_items)
    ling_values_dict = {}
    for i, s in enumerate(set_items):
        pos = s.get("position")
        # We ignore the original "name" and use the provided generic label.
        set_name = s.get("name")
        if n == 1:
            mf = LeftShoulderMF(pos, pos + 1)
        elif n == 2:
            if i == 0:
                mf = LeftShoulderMF(pos, set_items[i + 1]["position"])
            else:
                mf = RightShoulderMF(set_items[i - 1]["position"], pos)
        else:
            if i == 0:
                mf = LeftShoulderMF(pos, set_items[i + 1]["position"])
            elif i == n - 1:
                mf = RightShoulderMF(set_items[i - 1]["position"], pos)
            else:
                mf = TriangularMF(set_items[i - 1]["position"], pos, set_items[i + 1]["position"])
        ling_values_dict[set_name] = mf
    return LinguisticVariable(var_name, ling_values_dict)


def create_output_linguistic_variable(var_name, set_items):
    set_items = sorted(set_items, key=lambda s: s.get("position", 0))
    ling_values_dict = {}
    for s in set_items:
        pos = s.get("position")
        set_name = s.get("name")
        mf = SingletonMF(pos)
        ling_values_dict[set_name] = mf
    return LinguisticVariable(var_name, ling_values_dict)


def generate_generic_labels(n):
    match n:
        case 1:
            return ["Medium"]
        case 2:
            return ["Low", "High"]
        case 3:
            return ["Low", "Medium", "High"]
        case 4:
            return ["Very Low", "Low", "High", "Very High"]
        case 5:
            return ["Very Low", "Low", "Medium", "High", "Very High"]
        case 6:
            return [
                "Very Low",
                "Low",
                "Slightly Low",
                "Slightly High",
                "High",
                "Very High",
            ]
        case 7:
            return [
                "Very Low",
                "Low",
                "Slightly Low",
                "Medium",
                "Slightly High",
                "High",
                "Very High",
            ]
        case _:
            return [f"Set {i + 1}" for i in range(n)]


def _auto_bits(n):
    if n is None or n < 2:
        return 0
    return int(np.ceil(np.log2(n)))


def build_fuzzycoco_params(
    *,
    nb_features,
    n_outputs,
    nb_rules,
    nb_max_var_per_rule,
    max_generations,
    max_fitness,
    nb_cooperators,
    influence_rules_initial_population,
    influence_evolving_ratio,
    nb_sets_in,
    nb_sets_out,
    nb_bits_pos_in,
    nb_bits_pos_out,
    nb_bits_vars_in=None,
    nb_bits_sets_in=None,
    nb_bits_vars_out=None,
    nb_bits_sets_out=None,
    pop_size_rules,
    elite_size_rules,
    cx_prob_rules,
    mut_flip_genome_rules,
    mut_flip_bit_rules,
    pop_size_mfs,
    elite_size_mfs,
    cx_prob_mfs,
    mut_flip_genome_mfs,
    mut_flip_bit_mfs,
    threshold,
    metrics_weights=None,
    features_weights=None,
):
    """Build a ``FuzzyCocoParams`` instance using explicit estimator parameters."""

    def _bits_vars(value, override):
        if override is not None:
            return override
        if value is None:
            return None
        return _auto_bits(value) + 1

    def _bits_sets(sets, override):
        if override is not None:
            return override
        return _auto_bits(sets)

    bits_vars_in = _bits_vars(nb_features, nb_bits_vars_in)
    bits_sets_in = _bits_sets(nb_sets_in, nb_bits_sets_in)
    bits_vars_out = _bits_vars(n_outputs, nb_bits_vars_out)
    bits_sets_out = _bits_sets(nb_sets_out, nb_bits_sets_out)

    input_params = {
        "nb_sets": nb_sets_in,
        "nb_bits_pos": nb_bits_pos_in,
    }
    if bits_vars_in is not None:
        input_params["nb_bits_vars"] = bits_vars_in
    if bits_sets_in is not None:
        input_params["nb_bits_sets"] = bits_sets_in

    output_params = {
        "nb_sets": nb_sets_out,
        "nb_bits_pos": nb_bits_pos_out,
    }
    if bits_vars_out is not None:
        output_params["nb_bits_vars"] = bits_vars_out
    if bits_sets_out is not None:
        output_params["nb_bits_sets"] = bits_sets_out

    rules_params = {
        "pop_size": pop_size_rules,
        "elite_size": elite_size_rules,
        "cx_prob": cx_prob_rules,
        "mut_flip_genome": mut_flip_genome_rules,
        "mut_flip_bit": mut_flip_bit_rules,
    }

    mfs_params = {
        "pop_size": pop_size_mfs,
        "elite_size": elite_size_mfs,
        "cx_prob": cx_prob_mfs,
        "mut_flip_genome": mut_flip_genome_mfs,
        "mut_flip_bit": mut_flip_bit_mfs,
    }

    fitness_params = {
        "output_vars_defuzz_thresholds": [float(threshold)],
    }
    if metrics_weights is not None:
        unknown_keys = sorted(set(metrics_weights) - set(_METRICS_KEYS))
        if unknown_keys:
            raise ValueError(
                f"Unknown metrics_weights keys: {', '.join(unknown_keys)}",
            )
        normalized = {key: 0.0 for key in _METRICS_KEYS}
        for key, value in metrics_weights.items():
            normalized[key] = value
        fitness_params["metrics_weights"] = normalized
    if features_weights:
        fitness_params["features_weights"] = dict(features_weights)

    desc = {
        "global_params": {
            "nb_rules": nb_rules,
            "nb_max_var_per_rule": nb_max_var_per_rule,
            "max_generations": max_generations,
            "max_fitness": max_fitness,
            "nb_cooperators": nb_cooperators,
            "influence_rules_initial_population": influence_rules_initial_population,
            "influence_evolving_ratio": influence_evolving_ratio,
        },
        "input_vars_params": input_params,
        "output_vars_params": output_params,
        "rules_params": rules_params,
        "mfs_params": mfs_params,
        "fitness_params": fitness_params,
    }

    return FuzzyCocoParams.from_dict(desc)


def _parse_variables(desc):
    fs = desc.get("fuzzy_system", {})
    return fs.get("variables", {})


def _parse_rules(desc):
    fs = desc.get("fuzzy_system", {})
    return fs.get("rules", {})


def _parse_default_rules(desc):
    fs = desc.get("fuzzy_system", {})
    return fs.get("default_rules", {})


def parse_fuzzy_system_from_description(desc):
    """High‑level aggregator used by fit() to expose .variables_, .rules_, etc."""
    return (
        _parse_variables(desc),
        _parse_rules(desc),
        _parse_default_rules(desc),
    )


def to_linguistic_components(variables_dict, rules_dict, default_rules_dict):
    """
    Takes variables_/rules_/default_rules_ (parsed from description)
    and returns:
      linguistic_variables, fuzzy_rules, default_rules
    using the user-provided helper factories:
      - create_input_linguistic_variable
      - create_output_linguistic_variable
    """
    # ---- build LinguisticVariable objects ------------------------
    linguistic_variables = []
    lv_by_name = {}
    label_map = {}  # var -> {orig_set: generic_label}

    # input vars
    for var_name, sets in variables_dict.get("input", {}).items():
        # sets: {'sepal_length.1': pos, 'sepal_length.2': pos, ...}
        set_items = [{"name": k, "position": v} for k, v in sets.items()]
        generic = generate_generic_labels(len(set_items))
        for s, lbl in zip(set_items, generic, strict=False):
            label_map.setdefault(var_name, {})[s["name"]] = lbl
            s["name"] = lbl
        lv = create_input_linguistic_variable(var_name, set_items)
        linguistic_variables.append(lv)
        lv_by_name[var_name] = lv

    # output vars
    for var_name, sets in variables_dict.get("output", {}).items():
        set_items = [{"name": k, "position": v} for k, v in sets.items()]
        generic = generate_generic_labels(len(set_items))
        for s, lbl in zip(set_items, generic, strict=False):
            label_map.setdefault(var_name, {})[s["name"]] = lbl
            s["name"] = lbl
        lv = create_output_linguistic_variable(var_name, set_items)
        linguistic_variables.append(lv)
        lv_by_name[var_name] = lv

    # ---- build FuzzyRule / DefaultFuzzyRule ----------------------
    fixed_act = (np.min, "AND_min")
    fixed_imp = (np.min, "AND_min")
    fuzzy_rules = []
    default_rules = []

    for _, rule in rules_dict.items():
        ants = []
        for var, mf_dict in rule["antecedents"].items():
            orig_set = next(iter(mf_dict.keys()))
            label = label_map[var][orig_set]
            ants.append(Antecedent(lv_by_name[var], label, is_not=False))

        cons = []
        for var, mf_dict in rule["consequents"].items():
            orig_set = next(iter(mf_dict.keys()))
            label = label_map[var][orig_set]
            cons.append(Consequent(lv_by_name[var], label))

        fuzzy_rules.append(FuzzyRule(ants, fixed_act, cons, fixed_imp))

    # default consequents
    for var, orig_set in default_rules_dict.items():
        label = label_map[var][orig_set]
        cons = [Consequent(lv_by_name[var], label)]
        default_rules.append(DefaultFuzzyRule(cons, fixed_imp))

    return linguistic_variables, fuzzy_rules, default_rules


# ---- same helpers as before ----------------------------------------------


def _build_pos_and_label_maps(variables_dict):
    pos_index = {"input": {}, "output": {}}
    label_map = {"input": {}, "output": {}}
    for io in ("input", "output"):
        for var, sets in variables_dict.get(io, {}).items():
            items = sorted(sets.items(), key=lambda kv: kv[1])  # by position
            generic = generate_generic_labels(len(items))
            pos_index[io][var] = {}
            label_map[io][var] = {}
            for (orig_set, pos), gen in zip(items, generic, strict=False):
                pos_index[io][var][gen] = {"position": float(pos), "orig_set": orig_set}
                label_map[io][var][orig_set] = gen
    return pos_index, label_map


def _format_var_lines_for_io(pos_index, io, digits=6):
    out = {}
    for var, label_info in pos_index.get(io, {}).items():
        label_items = sorted(label_info.items(), key=lambda kv: kv[1]["position"])
        out[var] = [f"{lbl} = {info['position']:.{digits}g} (from {info['orig_set']})" for lbl, info in label_items]
    return out


def _lookup_label_and_pos(pos_index, label_map, var, orig_set):
    io = "input" if var in pos_index["input"] else "output"
    lbl = label_map[io][var][orig_set]
    pos = pos_index[io][var][lbl]["position"]
    return lbl, pos, io


def _rule_key(name):
    num = "".join(ch for ch in name if ch.isdigit())
    return (int(num) if num else 10**9, name)


# ---- VIEWS (human-readable text) -----------------------------------------


def to_views_components(variables_dict, rules_dict, default_rules_dict, *, digits=6):
    pos_index, label_map = _build_pos_and_label_maps(variables_dict)

    # flat variable map
    vars_in = _format_var_lines_for_io(pos_index, "input", digits)
    vars_out = _format_var_lines_for_io(pos_index, "output", digits)
    variables_view = dict(vars_in)
    for var, lines in vars_out.items():
        if var not in variables_view:
            variables_view[var] = lines
        else:
            variables_view[f"{var} (output)"] = lines  # collision guard

    def ants_to_text(ant_dict):
        parts = []
        for var, mf_dict in ant_dict.items():
            orig_set = next(iter(mf_dict.keys()))
            lbl, pos, _ = _lookup_label_and_pos(pos_index, label_map, var, orig_set)
            parts.append(f"{var} is {lbl} ({pos:.{digits}g})")
        return " AND ".join(parts) if parts else "TRUE"

    def cons_to_text(cons_dict):
        parts = []
        for var, mf_dict in cons_dict.items():
            orig_set = next(iter(mf_dict.keys()))
            lbl, pos, _ = _lookup_label_and_pos(pos_index, label_map, var, orig_set)
            parts.append(f"{var} is {lbl} ({pos:.{digits}g})")
        return " AND ".join(parts) if parts else "(no consequent)"

    rules_view = []
    for rname, rdef in sorted(rules_dict.items(), key=lambda kv: _rule_key(kv[0])):
        ants = ants_to_text(rdef.get("antecedents", {}))
        cons = cons_to_text(rdef.get("consequents", {}))
        rules_view.append(f"{rname.upper()}: IF {ants} THEN {cons}")

    default_rules_view = []
    for var, orig_set in default_rules_dict.items():
        lbl, pos, _ = _lookup_label_and_pos(pos_index, label_map, var, orig_set)
        default_rules_view.append(f"DEFAULT: {var} is {lbl} ({pos:.{digits}g})")

    return variables_view, rules_view, default_rules_view


# ---- TABLES (machine-friendly; great in notebooks) -----------------------


def to_tables_components(variables_dict, rules_dict, default_rules_dict):
    pos_index, label_map = _build_pos_and_label_maps(variables_dict)

    # variables_df
    rows_vars = []
    for io in ("input", "output"):
        for var, label_info in pos_index[io].items():
            for lbl, info in label_info.items():
                rows_vars.append(
                    {
                        "io": io,
                        "var": var,
                        "label": lbl,
                        "position": info["position"],
                        "orig_set": info["orig_set"],
                    }
                )
    variables_df = pd.DataFrame(rows_vars).sort_values(["io", "var", "position"]).reset_index(drop=True)

    # rules_df
    rows_rules = []
    for rname, rdef in sorted(rules_dict.items(), key=lambda kv: _rule_key(kv[0])):
        for role, part in (
            ("antecedent", rdef.get("antecedents", {})),
            ("consequent", rdef.get("consequents", {})),
        ):
            for var, mf_dict in part.items():
                orig_set = next(iter(mf_dict.keys()))
                lbl, pos, io = _lookup_label_and_pos(pos_index, label_map, var, orig_set)
                rows_rules.append(
                    {
                        "rule": rname,
                        "role": role,
                        "io": io,
                        "var": var,
                        "label": lbl,
                        "position": pos,
                        "orig_set": orig_set,
                    }
                )
    # defaults as rule rows too (role='default')
    for var, orig_set in default_rules_dict.items():
        lbl, pos, io = _lookup_label_and_pos(pos_index, label_map, var, orig_set)
        rows_rules.append(
            {
                "rule": "default",
                "role": "default",
                "io": io,
                "var": var,
                "label": lbl,
                "position": pos,
                "orig_set": orig_set,
            }
        )

    rules_df = pd.DataFrame(rows_rules).sort_values(["rule", "role", "var"]).reset_index(drop=True)
    return variables_df, rules_df
