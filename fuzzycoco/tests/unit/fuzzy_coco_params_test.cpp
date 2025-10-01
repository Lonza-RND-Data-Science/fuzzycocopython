#include "tests.h"
#include <algorithm>
#include "logging_logger.h"

#include "fuzzy_coco_params.h"
#include "string_utils.h"

using namespace fuzzy_coco;
using namespace logging;


TEST(VarsParams, basic) {
  VarsParams p;
  cerr << p;

  VarsParams p2;
  EXPECT_EQ(p, p2);

  p2.nb_bits_vars = 100;
  EXPECT_NE(p, p2);

  // evaluate_nb_bits_vars
  EXPECT_EQ(VarsParams::evaluate_nb_bits_for_nb(0), 0);
  EXPECT_EQ(VarsParams::evaluate_nb_bits_for_nb(1), 0);
  EXPECT_EQ(VarsParams::evaluate_nb_bits_for_nb(2), 1);
  EXPECT_EQ(VarsParams::evaluate_nb_bits_for_nb(7), 3);

  // // evaluate_nb_bits_vars
  EXPECT_EQ(VarsParams::evaluate_nb_bits_vars(0), 1);
  EXPECT_EQ(VarsParams::evaluate_nb_bits_vars(1), 1);
  EXPECT_EQ(VarsParams::evaluate_nb_bits_vars(2), 2);
  EXPECT_EQ(VarsParams::evaluate_nb_bits_vars(7), 4);

  // evaluate_missing
  p.evaluate_missing(4);
  VarsParams expected;
  expected.nb_bits_vars = 3;
  expected.nb_bits_sets = 2;

  EXPECT_EQ(p, expected);
  // reentrant
  p.evaluate_missing(4);
  EXPECT_EQ(p, expected);

  // N.B: p2 has already a value in .nb_bits_vars
  p2.evaluate_missing(4);
  EXPECT_NE(p2, expected);

  // === has_missing //
  EXPECT_TRUE(p.has_missing());
  p.nb_bits_pos = 2;
  EXPECT_FALSE(p.has_missing());
}

TEST(VarsParams, IO) {
  VarsParams p;
  p.evaluate_missing(4);

  auto desc = p.describe();
  VarsParams p2(desc);

  EXPECT_EQ(p2, p);

  // defaults
  NamedList quasi_empty;
  quasi_empty.add("nb_bits_sets", -1);
  VarsParams p3(quasi_empty);
  VarsParams expected;
  expected.nb_bits_sets = -1;
  EXPECT_EQ(p3, expected);
}

TEST(EvolutionParams, basic) {
  EvolutionParams p, p2;

  EXPECT_TRUE(p.has_missing());

  p2.pop_size = 100;
  EXPECT_FALSE(p2.has_missing());

  EXPECT_NE(p, p2);

  p.pop_size = 100;
  EXPECT_FALSE(p.has_missing());

  EXPECT_EQ(p, p2);

}

TEST(EvolutionParams, IO) {
  EvolutionParams p;
  p.pop_size = 1000;
  p.mut_flip_genome = 0.000001;
  p.mut_flip_bit = MISSING_DATA_DOUBLE;

  auto desc = p.describe();
  EvolutionParams p2(desc);
  cerr << p2;
  EXPECT_EQ(p2, p);

  // ==== load with defauls =====
  NamedList quasi_empty;
  quasi_empty.add("cx_prob", -0.1);
  EvolutionParams p3(quasi_empty);
  EvolutionParams ref;
  ref.cx_prob = -0.1;
  EXPECT_EQ(p3, ref);
}

TEST(GlobalParams, basic) {
  GlobalParams p, p2;
  cerr << p;
  EXPECT_EQ(p, p2);
  EXPECT_TRUE(p.has_missing());

  p.nb_rules = 3;
  EXPECT_TRUE(p.has_missing());
  EXPECT_NE(p, p2);

  p.nb_max_var_per_rule = 3;
  EXPECT_FALSE(p.has_missing());

  // checking default
  EXPECT_FALSE(p.influence_rules_initial_population); 
}

TEST(GlobalParams, IO) {
  GlobalParams p;
  // N.B != defaults to check save/load
  p.nb_rules = 1000;
  p.nb_max_var_per_rule = -1;
  p.influence_rules_initial_population = true; 
  p.influence_evolving_ratio = -1.5;

  auto desc = p.describe();
  cerr << desc;

  GlobalParams p2(desc);

  EXPECT_EQ(p2, p);

  // using defaults
  NamedList quasi_empty;
  quasi_empty.add("nb_max_var_per_rule", 2);
  GlobalParams p3(quasi_empty);
  GlobalParams expected;
  expected.nb_max_var_per_rule = 2;
  EXPECT_EQ(p3, expected);
}

TEST(FitnessParams, convertFeaturesWeights) {
  vector<string> input_vars = { "toto", "titi", "tata" };
  
  // no weights
  map<string, double> weights_by_name = {};

  auto weights = convertFeaturesWeights(input_vars, weights_by_name);
  EXPECT_TRUE(all(weights, [](auto x) { return x == 0; }));
  
  // some weights
  weights_by_name = { {"tata", 1}, {"toto", 0.1}};
  weights = convertFeaturesWeights(input_vars, weights_by_name);
  EXPECT_EQ(weights, vector<double>({0.1, 0, 1}));

  // bad name
  weights_by_name = { {"tata", 1}, {"bad", 0.1}};
  EXPECT_THROW(convertFeaturesWeights(input_vars, weights_by_name), runtime_error);

  // bad value
  weights_by_name = { {"tata", -1}};
  EXPECT_THROW(convertFeaturesWeights(input_vars, weights_by_name), runtime_error); 
}


TEST(FitnessParams, basic) {
  FitnessParams p;
  EXPECT_TRUE(p.has_missing());

  p.output_vars_defuzz_thresholds.push_back(0.5);
  EXPECT_FALSE(p.has_missing());

  // fix_output_thresholds
  p.fix_output_thresholds(5);
  EXPECT_EQ(p.output_vars_defuzz_thresholds, vector<double>({0.5, 0.5, 0.5, 0.5, 0.5}));

  p.features_weights["toto"] = 1;
  p.features_weights["titi"] = 0.1;

  EXPECT_EQ(p.asFeaturesWeightsByIdx({"toto", "titi"}), vector<double>({1, 0.1}));

  auto desc = p.describe();
  FitnessParams p2(desc);
  EXPECT_EQ(p2, p);

  // with defaults
  NamedList quasi_empty;
  quasi_empty.add("output_vars_defuzz_thresholds", 0.01);

  FitnessParams p3(quasi_empty);
  FitnessParams ref;
  ref.output_vars_defuzz_thresholds.push_back(0.01);
  EXPECT_EQ(p3, ref);
}


TEST(FuzzyCocoParams, basic) {
  FuzzyCocoParams p, p2;
  EXPECT_EQ(p, p2);

  // ============= has_missing ==============

  EXPECT_TRUE(p.has_missing());

  p.global_params.nb_rules = 5;
  p.global_params.nb_max_var_per_rule = 3;

  EXPECT_TRUE(p.has_missing());

  p.evaluate_missing(10, 1);
  EXPECT_TRUE(p.has_missing());

  p.input_vars_params.nb_bits_pos = 8;
  p.output_vars_params.nb_bits_pos = 2;
  EXPECT_TRUE(p.has_missing());

  p.rules_params.pop_size = 100;
  p.mfs_params.pop_size = 50;

  EXPECT_TRUE(p.has_missing());

  p.fitness_params.output_vars_defuzz_thresholds.push_back(0.5);
  EXPECT_FALSE(p.has_missing());
  cerr << p;

  // copy construction
  FuzzyCocoParams pbis = p;
  EXPECT_TRUE(p == pbis);
}

TEST(FuzzyCocoParams, IO) {
  FuzzyCocoParams p;
  p.global_params.nb_rules = 5;
  p.global_params.nb_max_var_per_rule = 3;
  p.evaluate_missing(10, 1);
  p.input_vars_params.nb_bits_pos = 8;
  p.output_vars_params.nb_bits_pos = 2;
  p.rules_params.pop_size = 100;
  p.mfs_params.pop_size = 50;
  p.fitness_params.output_vars_defuzz_thresholds.push_back(0.5);

  auto desc = p.describe();
  cerr << desc;

  FuzzyCocoParams p2(desc);

  EXPECT_EQ(p2, p);
  cerr << p2.describe();

  // 
  p.fitness_params.output_vars_defuzz_thresholds.push_back(1000);
  p.fitness_params.features_weights["var1"] = 0.11;
  p.fitness_params.features_weights["var2"] = 1;

  auto desc3 = p.describe();
  FuzzyCocoParams p3(desc3);

  EXPECT_EQ(p3, p);

  // ==== load with defaults ====
  NamedList quasi_empty;
  NamedList rules_params;
  rules_params.add("pop_size", -1);
  quasi_empty.add("rules_params", rules_params);

  FuzzyCocoParams p4(quasi_empty);
  FuzzyCocoParams ref;
  ref.rules_params.pop_size = -1;
  EXPECT_EQ(p4, ref);
}

TEST(FuzzyCocoParams, features_weights) {
  FuzzyCocoParams p;

  // make it non-missing
  p.global_params.nb_rules =  p.global_params.nb_max_var_per_rule = 1;
  p.rules_params.pop_size = 100;
  p.mfs_params.pop_size = 50;
  p.input_vars_params.nb_bits_pos = 8;
  p.output_vars_params.nb_bits_pos = 2;
  p.evaluate_missing(10, 1);
  p.fitness_params.output_vars_defuzz_thresholds.push_back(0.5);
  EXPECT_FALSE(p.has_missing());

  // ===
  FuzzyCocoParams p2 = p;

  EXPECT_TRUE(p.fitness_params.features_weights.empty());
  EXPECT_FALSE(p.has_missing()); // optional params

  p.fitness_params.features_weights["feat1"] = 0;
  p.fitness_params.features_weights["feat2"] = 0.5;
  p.fitness_params.features_weights["feat3"] = 1;

  EXPECT_EQ(p.fitness_params.features_weights.size(), 3);

  // ################ == #################
  EXPECT_FALSE(p == p2);
  EXPECT_TRUE(p != p2);

  // same values but in different order (map is sorted)
  p2.fitness_params.features_weights["feat3"] = 1;
  p2.fitness_params.features_weights["feat1"] = 0;
  p2.fitness_params.features_weights["feat2"] = 0.5;

  EXPECT_TRUE(p == p2);
  EXPECT_FALSE(p != p2);

  // ################ << #################
  cerr << p;
}

TEST(FuzzyCocoParams, params_json) {
  string PARAMS = R"(
{
  "global_params":{
    "nb_rules":3
    # (max) number of antecedents per rule
    ,"nb_max_var_per_rule":3
  },
  "fitness_params":{
    "output_vars_defuzz_thresholds": 0.5,
    "metrics_weights":{
      #"sensitivity":1.0
      "rmse":0.2
    }
  }
}
)";
  logger().activate();
  FuzzyCocoParams p(NamedList::parse(StringUtils::stripComments(PARAMS)));
  
  FuzzyCocoParams ref;
  ref.global_params.nb_rules = 3;
  ref.global_params.nb_max_var_per_rule = 3;
  ref.fitness_params.output_vars_defuzz_thresholds = {0.5};
  ref.fitness_params.metrics_weights.rmse = 0.2;

  EXPECT_EQ(p, ref);
}