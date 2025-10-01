#ifndef FUZZY_COCO_PARAMS_H
#define FUZZY_COCO_PARAMS_H

#include <string>
#include <iostream>
#include <cmath>
#include <map>

#include "fuzzy_system_metrics.h"
#include "evolution_params.h"
#include "named_list.h"

namespace fuzzy_coco {
using namespace std;

struct VarsParams {
  VarsParams() = default;
  VarsParams(const NamedList& desc);
  // the number of fuzzy sets to use for the Membership function associated with the variables
  int nb_sets = 3;
  // the nb of bits to encode the variable index. N.B: for input var, at least one extra bit must be added to account for the DontCare
  int nb_bits_vars = MISSING_DATA_INT;
  // the nb of bits to encode the set index. Directy related to nb_sets
  int nb_bits_sets = MISSING_DATA_INT;
  // the nb of bits to encode the variable position/value. a low value means that the variable range is heavily discretized (can take only few values)
  int nb_bits_pos = MISSING_DATA_INT;

  static int evaluate_nb_bits_for_nb(int nb) {  return nb < 2 ? 0 : ceil(log2(nb)); }
  static int evaluate_nb_bits_vars(int nb_vars) {  return evaluate_nb_bits_for_nb(nb_vars) + 1; }

  // N.B: nb_input_vars comes from the dataset
  void evaluate_missing(int nb_vars);

  bool has_missing() const { return is_na(nb_sets) || is_na(nb_bits_vars) || is_na(nb_bits_vars) || is_na(nb_bits_pos); }

  bool operator!=(const VarsParams& p) const { return ! (*this == p); }
  bool operator==(const VarsParams& p) const;

  NamedList describe() const;
  friend ostream& operator<<(ostream& out, const VarsParams& p);
};

struct GlobalParams {
  GlobalParams() = default;
  GlobalParams(const NamedList& desc);

  // N.B: default values, either NA --> must be evaluated from other values or given, or a value
  // ================ global =======================
  // the number of rules for inferring the fuzzy system. Note that some rules may be discarded because of the "DontCare" mechanism
  int nb_rules = MISSING_DATA_INT;
  // the number of input variables "slots" using by the antecedents of rules
  // note that some vars may be discarded because of the "DontCare" mechanism
  int nb_max_var_per_rule = MISSING_DATA_INT;
  // the maximum number of coevolution generations to compute. also cf max_fit
  int max_generations = 100;
  // the fitness theshold to stop the evolution. N.B: > 1 means that it will never early-stop
  double max_fitness = 1;
  // the number of cooperators to use to evaluate the fitness in the coevolution algorithm
  int nb_cooperators = 2;

  // whether to influence the initial genome rules population with the features weights (if any)
  bool influence_rules_initial_population = false;
  // the evolving ratio to use, cf influence_rules_initial_population
  double influence_evolving_ratio = 0.8;

  bool has_missing() const { 
      return is_na(nb_rules) || is_na(nb_max_var_per_rule) || is_na(max_generations) || is_na(max_fitness) || is_na(nb_cooperators); 
  }

  bool operator!=(const GlobalParams& p) const { return ! (*this == p); }
  bool operator==(const GlobalParams& p) const;

  NamedList describe() const;
  friend ostream& operator<<(ostream& out, const GlobalParams& p);
};

// build the weights by input variable index. variables not present in features_weigths_by_name will have a weight of 0
// also check that the names do exist in input_vars
// N.B: throws a runtime exception in case of bad var names in features_weights_by_name
vector<double> convertFeaturesWeights(const vector<string>& input_vars, const map<string, double>& features_weights_by_name);


// fitness related params
struct FitnessParams {
  FitnessParams(const NamedList& desc);
  FitnessParams() : metrics_weights() {
    metrics_weights.sensitivity = 1.0;
    metrics_weights.specificity = 0.8;
  }

  vector<double> output_vars_defuzz_thresholds;
  // bool defuzz_threshold_activated = true; // N.B: removed. We absolutely need the thresholds

  // fitness weights: weights to apply to the FuzzySystem Metrics, cf FuzzySystemWeightedFitness
  // N.B: we (ab)use a FuzzySystemMetrics struct to store the weights instead of the values
  FuzzySystemMetrics metrics_weights;

  // features weights: features can be encouraged/discouraged using weights between 0 to 1.
  // N.B: we use a map since a feature should have at max one weight
  // N.B: this is an optional param
  map<string, double> features_weights;
  vector<double> asFeaturesWeightsByIdx(const vector<string>& input_vars) const { 
    return convertFeaturesWeights(input_vars, features_weights); 
  }
  // recycle the output thresholds if it contains only one value
  void fix_output_thresholds(int nb_output_vars);

  bool has_missing() const;

  NamedList describe() const;
  friend ostream& operator<<(ostream& out, const FitnessParams& p);
  bool operator==(const FitnessParams& p) const;
};


struct FuzzyCocoParams
{
  GlobalParams global_params;
  // variables
  VarsParams input_vars_params;
  VarsParams output_vars_params;

  // =========== rules pop ===============
  EvolutionParams rules_params;
  // =========== Membership function positions pop
  EvolutionParams mfs_params;

  FitnessParams fitness_params;

  // // N.B: not used any longer
  // bool fixedVars = false;
  // ====================================================================================
  FuzzyCocoParams() {}
  FuzzyCocoParams(const NamedList& desc);

  void evaluate_missing(int nb_input_vars, int nb_output_vars) {
    input_vars_params.evaluate_missing(nb_input_vars);
    output_vars_params.evaluate_missing(nb_output_vars);
  }

  bool has_missing() const;
  bool operator!=(const FuzzyCocoParams& p) const { return ! (*this == p); }
  bool operator==(const FuzzyCocoParams& p) const;

  NamedList describe() const;
  friend ostream& operator<<(ostream& out, const FuzzyCocoParams& p);
};

}
#endif // FUZZY_COCO_PARAMS_H
