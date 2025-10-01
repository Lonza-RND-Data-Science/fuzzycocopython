#include <cmath>
#include "fuzzy_coco_params.h"
#include "dataframe.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace logging;

NamedList VarsParams::describe() const {
  NamedList desc;
  desc.add("nb_sets", nb_sets);
  desc.add("nb_bits_vars", nb_bits_vars);
  desc.add("nb_bits_sets", nb_bits_sets);
  desc.add("nb_bits_pos", nb_bits_pos);
  return desc;
} 

VarsParams::VarsParams(const NamedList& desc) {
  nb_sets = desc.get_as_int("nb_sets", nb_sets);
  nb_bits_vars = desc.get_as_int("nb_bits_vars", nb_bits_vars);
  nb_bits_sets = desc.get_as_int("nb_bits_sets", nb_bits_sets);
  nb_bits_pos = desc.get_as_int("nb_bits_pos", nb_bits_pos);
}



bool VarsParams::operator==(const VarsParams& p) const {
    return 
        nb_sets == p.nb_sets && 
        nb_bits_vars == p.nb_bits_vars && 
        nb_bits_sets == p.nb_bits_sets &&
        nb_bits_pos == p.nb_bits_pos;
}

void VarsParams::evaluate_missing(int nb_vars) {
    if (is_na(nb_bits_vars))
        nb_bits_vars = evaluate_nb_bits_vars(nb_vars);
    if (is_na(nb_bits_sets))
        nb_bits_sets = evaluate_nb_bits_for_nb(nb_sets);
}

NamedList GlobalParams::describe() const {
  NamedList desc;
  desc.add("nb_rules", nb_rules);
  desc.add("nb_max_var_per_rule", nb_max_var_per_rule);
  desc.add("max_generations", max_generations);
  desc.add("max_fitness", max_fitness);
  desc.add("nb_cooperators", nb_cooperators);
  desc.add("influence_rules_initial_population", influence_rules_initial_population);
  desc.add("influence_evolving_ratio", influence_evolving_ratio);
  return desc;
} 

GlobalParams::GlobalParams(const NamedList& desc) {
  nb_rules = desc.get_as_int("nb_rules", nb_rules);
  nb_max_var_per_rule = desc.get_as_int("nb_max_var_per_rule", nb_max_var_per_rule);
  max_generations = desc.get_as_int("max_generations", max_generations);
  max_fitness = desc.get_double("max_fitness", max_fitness);
  nb_cooperators = desc.get_as_int("nb_cooperators", nb_cooperators);
  influence_rules_initial_population = desc.get_bool("influence_rules_initial_population", influence_rules_initial_population);
  influence_evolving_ratio = desc.get_double("influence_evolving_ratio", influence_evolving_ratio);
}

bool GlobalParams::operator==(const GlobalParams& p) const {
    return 
        nb_rules == p.nb_rules && 
        nb_max_var_per_rule == p.nb_max_var_per_rule && 
        max_generations == p.max_generations &&
        max_fitness == p.max_fitness&&
        nb_cooperators == p.nb_cooperators &&
        influence_rules_initial_population == p.influence_rules_initial_population;
}




NamedList FitnessParams::describe() const {
  NamedList desc;
  desc.add("output_vars_defuzz_thresholds", output_vars_defuzz_thresholds);
  // desc.add("defuzz_threshold_activated", defuzz_threshold_activated);
  desc.add("metrics_weights", metrics_weights.describe());
  desc.add("features_weights", features_weights);
  return desc;
} 

bool FitnessParams::has_missing() const {
  return output_vars_defuzz_thresholds.empty();
}

FitnessParams::FitnessParams(const NamedList& desc) 
  : FitnessParams() 
{
  if (desc.has("metrics_weights")) 
    metrics_weights.setValues(desc.get_list("metrics_weights"));
 
  if (desc.has("output_vars_defuzz_thresholds")) 
    output_vars_defuzz_thresholds = desc["output_vars_defuzz_thresholds"].as_numeric_vector();
    
  // defuzz_threshold_activated = bool(desc["defuzz_threshold_activated"].get_int());
  if (desc.has("features_weights"))
    features_weights = desc["features_weights"].as_string_numeric_map();
}

void FitnessParams::fix_output_thresholds(int nb_output_vars) 
{
  assert(nb_output_vars > 0);
  if (output_vars_defuzz_thresholds.size() != 1) return;
  int missing = nb_output_vars - output_vars_defuzz_thresholds.size();
  double scalar = output_vars_defuzz_thresholds.front();
  for (int i = 0; i < missing; i++)
    output_vars_defuzz_thresholds.push_back(scalar);
}

bool FitnessParams::operator==(const FitnessParams& p) const {
  return 
    output_vars_defuzz_thresholds == p.output_vars_defuzz_thresholds &&
    metrics_weights == p.metrics_weights &&
    features_weights == p.features_weights;
}




vector<double> 
fuzzy_coco::convertFeaturesWeights(const vector<string>& input_vars, const map<string, double>& features_weights_by_name)
{
  vector<double> weights(input_vars.size(), 0);

  map<string, int> var_to_idx;
  for (size_t i = 0; i < input_vars.size(); i++) 
    var_to_idx[input_vars[i]] = i;

  for (const auto& [key, value] : features_weights_by_name) {
    auto search = var_to_idx.find(key);
    if (search == var_to_idx.end())
      throw runtime_error("bad input variable name '" + key + "'");
    if (value < 0 || value > 1)
      throw runtime_error("bad features weight value, mut be in [0,1]: '" + std::to_string(value) + "'");
    weights[search->second] = value;
  }

  return weights;
}


FuzzyCocoParams::FuzzyCocoParams(const NamedList& desc)
{
  global_params = GlobalParams(desc.get_list("global_params", NamedList()));
  input_vars_params = VarsParams(desc.get_list("input_vars_params", NamedList()));
  output_vars_params = VarsParams(desc.get_list("output_vars_params", NamedList()));
  // output_vars_defuzz_thresholds = desc["output_vars_defuzz_thresholds"].as_numeric_vector();
  // defuzz_threshold_activated = bool(desc["defuzz_threshold_activated"].get_int());
  rules_params = EvolutionParams(desc.get_list("rules_params", NamedList()));
  mfs_params = EvolutionParams(desc.get_list("mfs_params", NamedList()));
  fitness_params = FitnessParams(desc.get_list("fitness_params", NamedList()));
}

NamedList FuzzyCocoParams::describe() const {
  NamedList desc;
  desc.add("global_params", global_params.describe());
  desc.add("input_vars_params", input_vars_params.describe());
  desc.add("output_vars_params", output_vars_params.describe());
  desc.add("rules_params", rules_params.describe());
  desc.add("mfs_params", mfs_params.describe());
  desc.add("fitness_params", fitness_params.describe());
  return desc;
} 

bool FuzzyCocoParams::has_missing() const {
  return 
    global_params.has_missing() ||
    input_vars_params.has_missing() ||
    output_vars_params.has_missing() ||
    rules_params.has_missing() ||
    fitness_params.has_missing() ||
    mfs_params.has_missing();
}

bool FuzzyCocoParams::operator==(const FuzzyCocoParams& p) const {
  return 
    global_params == p.global_params &&
    input_vars_params == p.input_vars_params &&
    output_vars_params == p.output_vars_params &&
    fitness_params == p.fitness_params &&
    rules_params == p.rules_params &&
    mfs_params == p.mfs_params;
}

namespace fuzzy_coco {
  ostream& operator<<(ostream& out, const VarsParams& p) {
      DataFrame df(1, 4);
      df.colnames({"nb_sets", "nb_bits_vars", "nb_bits_sets", "nb_bit_pos"});
      auto D = [](int i) { return is_na(i) ? MISSING_DATA_DOUBLE : double(i); };
      vector<double> row = {D(p.nb_sets), D(p.nb_bits_vars), D(p.nb_bits_sets), D(p.nb_bits_pos)};
      df.fillRow(0, row);
      out << df;

      return out;
  }

  ostream& operator<<(ostream& out, const GlobalParams& p) {
      DataFrame df(1, 6);
      df.colnames({"nb_rules", "nb_max_var_per_rule", "max_generations", "max_fitness", "nb_cooperators", "influence_rules_initial_population"});
      auto D = [](int i) { return is_na(i) ? MISSING_DATA_DOUBLE : double(i); };
      vector<double> row = {D(p.nb_rules), D(p.nb_max_var_per_rule), D(p.max_generations), D(p.max_fitness), D(p.nb_cooperators),
        static_cast<double>(p.influence_rules_initial_population)};
      df.fillRow(0, row);
      out << df;

      return out;
  }

  ostream& operator<<(ostream& out, const FitnessParams& p) {

    out << "output_vars_defuzz_thresholds:" << p.output_vars_defuzz_thresholds << endl
        << "Fitness Metric Weights:" << endl << '(' << p.metrics_weights << ')' << endl;

    out << "Features Weights:";
    if (p.features_weights.empty()) {
      out << "None\n";
    } else {
      const int nb = p.features_weights.size();
      DataFrame fw(nb, 1);
      vector<string> rownames(nb);
      fw.colnames({"weight"});
      int row = 0;
      for (const auto& [key, value] : p.features_weights) {
          rownames[row] = key;
          fw.fillRow(row,  { value });
          row++;
      }
      fw.rownames(rownames);
      out << endl << fw;
    }

    return out;
  }

  ostream& operator<<(ostream& out, const FuzzyCocoParams& p) {
    out << "Fuzzy Coco Params:" << endl
        << "===================" << endl
        << "Global params:" << endl << p.global_params
        << "Input Variables:" << endl << p.input_vars_params
        << "Output Variables:" << endl << p.output_vars_params
        << "Fitness params:" << endl  << p.fitness_params
        << "Rules params"  << endl << p.rules_params
        << "MFs (Membership Functions) params"  << endl << p.mfs_params;
  
      return out;
  }
}