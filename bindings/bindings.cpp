#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <limits>
#include <sstream>

#include "dataframe.h"
#include "fuzzy_coco.h"
#include "fuzzy_coco_params.h"
#include "fuzzy_system.h"
#include "named_list.h"
#include "random_generator.h"

namespace py = pybind11;
using fuzzy_coco::NamedList;
using fuzzy_coco::Scalar;
using fuzzy_coco::DataFrame;
using fuzzy_coco::GlobalParams;
using fuzzy_coco::VarsParams;
using fuzzy_coco::EvolutionParams;
using fuzzy_coco::FitnessParams;
using fuzzy_coco::FuzzyCocoParams;
using fuzzy_coco::FuzzyCoco;
using fuzzy_coco::FuzzySystem;
using fuzzy_coco::RandomGenerator;

// -----------------------------------------------------------------------------
// Helper to convert NamedList recursively to Python objects
// -----------------------------------------------------------------------------
py::object named_list_to_python(const NamedList &nl) {
    if (nl.is_scalar()) {
        const Scalar &v = nl.scalar();
        if (v.is_bool())    return py::bool_(v.get_bool());
        if (v.is_int())     return py::int_(v.get_int());
        if (v.is_double())  return py::float_(v.get_double());
        if (v.is_string())  return py::str(v.get_string());
        return py::none();
    }
    py::dict d;
    for (const auto &child : nl) d[py::str(child.name())] = named_list_to_python(child);
    return d;
}

// -----------------------------------------------------------------------------
// Helper to convert Python objects back to NamedList
// -----------------------------------------------------------------------------
NamedList python_to_named_list(const std::string& name, py::handle obj) {
    if (obj.is_none()) {
        return NamedList(name);
    }
    if (py::isinstance<py::bool_>(obj)) {
        return NamedList(name, obj.cast<bool>());
    }
    if (py::isinstance<py::int_>(obj)) {
        long long value = obj.cast<long long>();
        if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
            return NamedList(name, static_cast<double>(value));
        }
        return NamedList(name, static_cast<int>(value));
    }
    if (py::isinstance<py::float_>(obj)) {
        return NamedList(name, obj.cast<double>());
    }
    if (py::isinstance<py::str>(obj)) {
        return NamedList(name, obj.cast<std::string>());
    }
    if (py::isinstance<py::dict>(obj)) {
        NamedList list(name);
        py::dict d = py::reinterpret_borrow<py::dict>(obj);
        for (auto item : d) {
            std::string key = py::cast<std::string>(item.first);
            auto child = python_to_named_list(key, item.second);
            list.add(key, child);
        }
        return list;
    }
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        NamedList list(name);
        py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
        const py::ssize_t n = seq.size();
        for (py::ssize_t i = 0; i < n; ++i) {
            std::string key = std::to_string(i);
            auto child = python_to_named_list(key, seq[i]);
            list.add(key, child);
        }
        return list;
    }
    throw std::runtime_error("Unsupported Python type when converting to NamedList");
}

// -----------------------------------------------------------------------------
PYBIND11_MODULE(_fuzzycoco_core, m) {
    m.doc() = "Python bindings for the FuzzyCoco library";

    m.def(
        "_named_list_from_dict_to_string",
        [](py::dict desc) {
            return python_to_named_list("root", desc).to_string();
        },
        py::arg("desc"));

    m.def(
        "_round_trip_named_list",
        [](py::dict desc) {
            auto nl = python_to_named_list("root", desc);
            return named_list_to_python(nl);
        },
        py::arg("desc"));

    m.def(
        "_rules_fire_from_description",
        [](py::dict saved_desc, py::dict sample_map) {
            auto saved = python_to_named_list("saved", saved_desc);
            auto fs = FuzzySystem::load(saved["fuzzy_system"]);
            const auto& db = fs.getDB();
            const int nb_inputs = db.getNbInputVars();
            std::vector<double> values(nb_inputs, 0.0);
            for (int i = 0; i < nb_inputs; ++i) {
                const auto& var = db.getInputVariable(i);
                py::handle key = py::str(var.getName());
                if (!sample_map.contains(key)) {
                    throw std::runtime_error("Missing value for input variable '" + var.getName() + "'");
                }
                values[i] = sample_map[key].cast<double>();
            }
            return fs.computeRulesFireLevels(values);
        },
        py::arg("description"),
        py::arg("sample"));

    // ------------------------ DataFrame ----------------------------------
    py::class_<DataFrame>(m, "DataFrame")
        .def(py::init<const std::vector<std::vector<std::string>>&, bool>())
        .def("nbcols", &DataFrame::nbcols)
        .def("nbrows", &DataFrame::nbrows)
        .def("to_list", [](const DataFrame &df) {
            std::vector<std::vector<double>> rows;
            for (int r = 0; r < df.nbrows(); ++r) rows.push_back(df.fetchRow(r));
            return rows;
        });

    // ------------------------ Parameter structs --------------------------
    py::class_<GlobalParams>(m, "GlobalParams")
        .def(py::init<>())
        .def_readwrite("nb_rules", &GlobalParams::nb_rules)
        .def_readwrite("nb_max_var_per_rule", &GlobalParams::nb_max_var_per_rule)
        .def_readwrite("max_generations", &GlobalParams::max_generations)
        .def_readwrite("max_fitness", &GlobalParams::max_fitness)
        .def_readwrite("nb_cooperators", &GlobalParams::nb_cooperators)
        .def_readwrite("influence_rules_initial_population", &GlobalParams::influence_rules_initial_population)
        .def_readwrite("influence_evolving_ratio", &GlobalParams::influence_evolving_ratio);

    py::class_<VarsParams>(m, "VarsParams")
        .def(py::init<>())
        .def_readwrite("nb_sets", &VarsParams::nb_sets)
        .def_readwrite("nb_bits_vars", &VarsParams::nb_bits_vars)
        .def_readwrite("nb_bits_sets", &VarsParams::nb_bits_sets)
        .def_readwrite("nb_bits_pos", &VarsParams::nb_bits_pos);

    py::class_<EvolutionParams>(m, "EvolutionParams")
        .def(py::init<>())
        .def_readwrite("pop_size", &EvolutionParams::pop_size)
        .def_readwrite("elite_size", &EvolutionParams::elite_size)
        .def_readwrite("cx_prob", &EvolutionParams::cx_prob)
        .def_readwrite("mut_flip_genome", &EvolutionParams::mut_flip_genome)
        .def_readwrite("mut_flip_bit", &EvolutionParams::mut_flip_bit);

    py::class_<FitnessParams>(m, "FitnessParams")
        .def(py::init<>())
        .def("fix_output_thresholds",
            &FitnessParams::fix_output_thresholds,
            py::arg("nb_out_vars"))
        .def_readwrite("output_vars_defuzz_thresholds",
                    &FitnessParams::output_vars_defuzz_thresholds)
        .def_readwrite("metrics_weights", &FitnessParams::metrics_weights)
        .def_readwrite("features_weights", &FitnessParams::features_weights);

    py::class_<FuzzyCocoParams>(m, "FuzzyCocoParams")
        .def(py::init<>())
        .def(py::init<>([](py::dict desc) {
            return FuzzyCocoParams(python_to_named_list("params", desc));
        }))
        .def_readwrite("global_params", &FuzzyCocoParams::global_params)
        .def_readwrite("input_vars_params", &FuzzyCocoParams::input_vars_params)
        .def_readwrite("output_vars_params", &FuzzyCocoParams::output_vars_params)
        .def_readwrite("rules_params", &FuzzyCocoParams::rules_params)
        .def_readwrite("mfs_params", &FuzzyCocoParams::mfs_params)
        .def_readwrite("fitness_params", &FuzzyCocoParams::fitness_params)
        .def(
            "describe",
            [](const FuzzyCocoParams& self) {
                return named_list_to_python(self.describe());
            })
        .def_static(
            "from_dict",
            [](py::dict desc) {
                return FuzzyCocoParams(python_to_named_list("params", desc));
            },
            py::arg("desc"));

    // ------------------------ RandomGenerator ----------------------------
    py::class_<RandomGenerator>(m, "RandomGenerator")
        .def(py::init<uint32_t>(), py::arg("seed"));

    // ------------------------ FuzzySystem ------------------------------
    py::class_<FuzzySystem>(m, "FuzzySystem")
        .def_static(
            "load_from_string",
            [](const std::string& content) {
                return FuzzySystem::load(content);
            },
            py::arg("content"))
        .def(
            "predict",
            &FuzzySystem::smartPredict,
            py::arg("df"))
        .def(
            "smart_predict",
            &FuzzySystem::smartPredict,
            py::arg("df"))
        .def(
            "compute_rules_fire_levels",
            [](const FuzzySystem& self, py::sequence seq) {
                std::vector<double> values = seq.cast<std::vector<double>>();
                auto expected = static_cast<size_t>(self.getDB().getNbInputVars());
                if (values.size() != expected) {
                    throw std::runtime_error(
                        "Expected " + std::to_string(expected) +
                        " input values, got " + std::to_string(values.size()));
                }
                return self.computeRulesFireLevels(values);
            },
            py::arg("input_values"))
        .def(
            "compute_rules_fire_levels_from_dataframe",
            [](const FuzzySystem& self, const DataFrame& df, int row) {
                std::vector<double> fire_levels;
                self.computeRulesFireLevels(row, df, fire_levels);
                return fire_levels;
            },
            py::arg("dataframe"),
            py::arg("row") = 0)
        .def(
            "describe",
            [](const FuzzySystem& self) {
                return named_list_to_python(self.describe());
            })
        .def(
            "nb_input_vars",
            [](const FuzzySystem& self) {
                return self.getDB().getNbInputVars();
            })
        .def(
            "default_rule_indices",
            [](const FuzzySystem& self) {
                return self.getDefaultRulesOutputSets();
            })
        .def(
            "input_variable_names",
            [](const FuzzySystem& self) {
                std::vector<std::string> names;
                const auto& db = self.getDB();
                const int nb = db.getNbInputVars();
                names.reserve(nb);
                for (int i = 0; i < nb; ++i) {
                    names.push_back(db.getInputVariable(i).getName());
                }
                return names;
            })
        .def_static(
            "load_from_dict",
            [](py::dict desc) {
                return FuzzySystem::load(python_to_named_list("fuzzy_system", desc));
            },
            py::arg("desc"));

    // ------------------------ FuzzyCoco core -----------------------------
    py::class_<FuzzyCoco>(m, "FuzzyCoco")
        .def(py::init<const DataFrame&, const DataFrame&, const FuzzyCocoParams&, RandomGenerator&>(),
             py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())
        .def(py::init<const DataFrame&, const DataFrame&, const FuzzyCocoParams&, const FuzzySystem&, RandomGenerator&>(),
             py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())
        .def("run", [](FuzzyCoco &self) {
            self.run();
        })
        .def("init",
             static_cast<void (FuzzyCoco::*)(bool, double)>(&FuzzyCoco::init),
             py::arg("influence") = false,
             py::arg("evolving_ratio") = 0.8)
        .def("step", &FuzzyCoco::step)
        .def("current_generation", &FuzzyCoco::currentGenerationNumber)
        .def("current_fitness", &FuzzyCoco::currentFitness)
        .def("has_active_generation", &FuzzyCoco::hasActiveGeneration)
        .def("select_best", &FuzzyCoco::selectBestFuzzySystem)
        .def("describe", [](FuzzyCoco &self) {
            return named_list_to_python(self.describeBestFuzzySystem());
        })
        .def("serialize_fuzzy_system",
             [](FuzzyCoco &self) {
                 auto desc = self.describeBestFuzzySystem();
                 return desc["fuzzy_system"].to_string();
             })
        .def("predict", &FuzzyCoco::predict)
        .def("rules_fire_from_values",
             [](FuzzyCoco &self, const std::vector<double>& x) {
                 return self.getFuzzySystem().computeRulesFireLevels(x);
             },
             py::arg("input_values"))
        .def_static(
            "load_and_predict_from_dict",
            [](const DataFrame& df, py::dict desc) {
                return FuzzyCoco::loadAndPredict(df, python_to_named_list("saved", desc));
            },
            py::arg("df"),
            py::arg("description"));
}
