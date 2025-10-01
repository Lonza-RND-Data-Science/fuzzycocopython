#include "tests.h"
#include <algorithm>
#include "fuzzy_coco.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace logging;

string CSV = 
R"(Days;Temperature;Sunshine;Tourists
day1;19;25;55
day2;40;99;95
day3;24;NA;70
day4;5;3;2
)";

DataFrame DF(CSV, true);
DataFrame DFIN = DF.subsetColumns(0, DF.nbcols() - 2);
DataFrame DFOUT = DF.subsetColumns(DF.nbcols() - 1, DF.nbcols() - 1);

string FUZZY_SYSTEM_112_POS = R"(
  {
    "Temperature":{
      "Cold":17.0,
      "Warm":20.0,
      "Hot":29.0
    },
    "Sunshine":{
      "Cloudy":30.0,
      "PartSunny":50.0,
      "Sunny":100.0
    }
    "Tourists":{
      "Low":0.0,
      "Medium":50.0,
      "High":100.0
    }
  }
  )";

TEST(FuzzyCocoFitnessMethod, createDiscretizersForData) {
  DataFrame df(CSV, true);

  auto ds = FuzzyCocoCodec::createDiscretizersForData(df, 2);

  EXPECT_EQ(ds.size(), df.nbcols());
}

FuzzyCocoParams GET_SAMPLE_PARAMS(int nb_max_var_per_rule) {
  FuzzyCocoParams params;

  params.global_params.nb_rules = 3;
  params.global_params.nb_max_var_per_rule = nb_max_var_per_rule;
  params.global_params.nb_cooperators = 2;

  auto& ip = params.input_vars_params;
  ip.nb_sets = 3;
  ip.nb_bits_vars = 3;
  ip.nb_bits_sets = 2;
  ip.nb_bits_pos = 8;

  auto& op = params.output_vars_params;
  op.nb_sets = 3;
  op.nb_bits_vars = 2;
  op.nb_bits_sets = 2;
  op.nb_bits_pos = 2;

  params.fitness_params.metrics_weights.sensitivity = 1;
  params.fitness_params.metrics_weights.specificity = 1;
  params.fitness_params.metrics_weights.nb_vars = 0.35;

  // !!!
  params.fitness_params.output_vars_defuzz_thresholds.push_back(50);
  
  params.rules_params.pop_size = 10;
  params.mfs_params.pop_size = 20;

  return params;
}

FuzzySystem GET_FUZZY_SYSTEM(const FuzzyCocoParams& params) {
  return FuzzySystem(DFIN.colnames(), DFOUT.colnames(), params.input_vars_params.nb_sets, params.output_vars_params.nb_sets);
}

TEST(FuzzyCocoFitnessMethod, fitnessImpl) {
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());

  // N.B: we set metrics_weights.nb_vars to 0 otherwise the fitness can not be equal to 1
  params.fitness_params.metrics_weights.nb_vars = 0;



  // !!!
  RandomGenerator rng(666);
  FuzzyCoco coco(DFIN, DFOUT, params, rng);

  coco.getFuzzySystem().getDB().setPositions(FUZZY_SYSTEM_112_POS);
  const auto& db = coco.getFuzzySystem().getDB();

  // a hand-made optimal pair to test the fitness
  auto& fs = coco.getFuzzySystem();

  string GOOD_RULES = R"(
  {
  "rule1":{
    "antecedents":{
      "Temperature":"Cold",
      "Sunshine":"Cloudy"
    },
    "consequents":{
      "Tourists":"Low"
    }
  },
  "rule2":{
    "antecedents":{
      "Temperature":"Warm",
      "Sunshine":"PartSunny"
    },
    "consequents":{
      "Tourists":"Medium"
    }
  },
  "rule3":{
    "antecedents":{
      "Temperature":"Hot",
      "Sunshine":"Sunny"
    },
    "consequents":{
      "Tourists":"High"
    }
  }
}
)";

  auto rules = FuzzyRule::loadRules(GOOD_RULES, db);
  fs.setRules(rules);
  FuzzyDefaultRule defrule = FuzzyDefaultRule::load(R"("Tourists":"High")", db);
  fs.setDefaultRule(defrule);

  string POS = R"(
  {
    "Temperature":{
      "Cold":10,
      "Warm":20,
      "Hot":30
    },
    "Sunshine":{
      "Cloudy":25,
      "PartSunny":50,
      "Sunny":75
    }
    "Tourists":{
      "Low":5,
      "Medium":50,
      "High":100
    }
  }
  )";
  cerr << db;
  fs.setVariablesSetPositions(POS);

  // testing fitness
  EXPECT_TRUE(coco.getFitnessMethod().getBestFitness() < 0); // no fitness computed yet

  double fit = coco.getFitnessMethod().fitnessImpl();
  // double fit = coco.getFitnessMethod().fitnessImpl(rules_geno, vars_geno);
  EXPECT_DOUBLE_EQ(fit, 1);


  // the same to build the worst rules and pos
  string BAD_RULES = R"(
    {
    "rule1":{
      "antecedents":{
        "Temperature":"Cold",
        "Sunshine":"Cloudy"
      },
      "consequents":{
        "Tourists":"High"
      }
    },
    "rule2":{
      "antecedents":{
        "Temperature":"Warm",
        "Sunshine":"PartSunny"
      },
      "consequents":{
        "Tourists":"Low"
      }
    },
    "rule3":{
      "antecedents":{
        "Temperature":"Hot",
        "Sunshine":"Sunny"
      },
      "consequents":{
        "Tourists":"Low"
      }
    }
  }
  )";


  fs.setRules(FuzzyRule::loadRules(BAD_RULES, db));
  fs.setDefaultRule(FuzzyDefaultRule::load(R"("Tourists":"High")", db));

  fit = coco.getFitnessMethod().fitnessImpl();

  cerr << "fit=" << fit << endl;

  EXPECT_TRUE(fit < 0.2);
}


TEST(FuzzyCocoFeaturesWeightsFitnessMethod, sumOfUsedFeaturesWeights) {
  auto sumof = FuzzyCocoFeaturesWeightsFitnessMethod::sumOfUsedFeaturesWeights;
  EXPECT_EQ(sumof({false, false}, {0.5, 0.9}), 0);
  EXPECT_EQ(sumof({true, true}, {0.5, 0.9}), 0.5 + 0.9);
  EXPECT_EQ(sumof({true, false}, {0.5, 0.9}), 0.5);
  EXPECT_EQ(sumof({true, true, true}, {0, 0, 0}), 0);
}

TEST(FuzzyCocoFeaturesWeightsFitnessMethod, mandatoryFeatureNotUsed) {
  auto mandatory = FuzzyCocoFeaturesWeightsFitnessMethod::mandatoryFeatureNotUsed;
  EXPECT_FALSE(mandatory({true, true}, {1, 0.9}));
  EXPECT_TRUE(mandatory({false, false}, {1, 0.9}));
  EXPECT_TRUE(mandatory({true, false}, {0.5, 2}));
  EXPECT_FALSE(mandatory({false, false, true}, {0.1, 0, 1.1}));
}


TEST(FuzzyCocoFeaturesWeightsFitnessMethod, getFeaturesWeights) {
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());
  
  // N.B: we set metrics_weights.nb_vars to 0 otherwise the fitness can not be equal to 1
  params.fitness_params.metrics_weights.nb_vars = 0;

  auto fs = GET_FUZZY_SYSTEM(params);

  /// ================ getFeaturesWeights =======================

  // N.B: no features weights yet
  FuzzySystemWeightedFitness fs_fitter(params.fitness_params.metrics_weights);
  FuzzyCocoFeaturesWeightsFitnessMethod fitter0(fs, fs_fitter, DFIN, DFOUT, params);
  vector<bool> used_features;

  EXPECT_TRUE(all(fitter0.getFeaturesWeights(), [](auto x) { return x == 0; }));
  EXPECT_EQ(fitter0.getSumOfFeaturesWeights(), 0);
  used_features = {false, false};

  params.fitness_params.features_weights["Sunshine"] = 0.2;
  params.fitness_params.features_weights["Temperature"] = 0.9;


  FuzzyCocoFitnessMethod fitter_ref(fs, fs_fitter, DFIN, DFOUT, params);
  FuzzyCocoFeaturesWeightsFitnessMethod fitter(fs, fs_fitter, DFIN, DFOUT, params);
  EXPECT_EQ(fitter.getFeaturesWeights(), vector<double>({0.9, 0.2}));
  EXPECT_EQ(fitter.getSumOfFeaturesWeights(), 1.1);
}

// Testing the features weight impact on fitness on rules using a single input variable
TEST(FuzzyCocoFeaturesWeightsFitnessMethod, test1) {
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());

  // N.B: we set metrics_weights.nb_vars to 0 otherwise the fitness can not be equal to 1
  params.fitness_params.metrics_weights.nb_vars = 0;
  auto fs = GET_FUZZY_SYSTEM(params);
  FuzzySystemWeightedFitness fs_fitter(params.fitness_params.metrics_weights);

  FuzzyCocoFeaturesWeightsFitnessMethod fitter0(fs, fs_fitter, DFIN, DFOUT, params);
  FuzzyCocoFitnessMethod fitter_ref(fs, fs_fitter, DFIN, DFOUT, params);

  params.fitness_params.features_weights["Sunshine"] = 0.2;
  params.fitness_params.features_weights["Temperature"] = 0.9;
  FuzzyCocoFeaturesWeightsFitnessMethod fitter(fs, fs_fitter, DFIN, DFOUT, params);

  auto init_fitter = [](FuzzyCocoFitnessMethod& fitmeth) {
    auto& fcs = fitmeth.getFuzzySystem();
    fcs.getDB().setPositions(FUZZY_SYSTEM_112_POS);
    string RULES = R"(
      {
      "rule1":{
        "antecedents":{
          "Temperature":"Cold"
        },
        "consequents":{
          "Tourists":"High"
        }
      },
      "rule2":{
        "antecedents":{
          "Temperature":"Warm"
        },
        "consequents":{
          "Tourists":"Medium"
        }
      },
      "rule3":{
        "antecedents":{
          "Temperature":"Hot"
        },
        "consequents":{
          "Tourists":"High"
        }
      }
    }
    )";
    fcs.setRules(FuzzyRule::loadRules(RULES, fcs.getDB()));
    fcs.setDefaultRule(FuzzyDefaultRule::load(R"("Tourists":"Medium")", fcs.getDB()));
  };

  // ====== manual build of rules and mfs genomes ======
  init_fitter(fitter0);
  init_fitter(fitter_ref);
  init_fitter(fitter);

  // fitness!
  // logger().activate();
  double fit_ref = fitter_ref.fitnessImpl();
  double fit = fitter.fitnessImpl();
  double fit0 = fitter0.fitnessImpl();
  // logger().activate(false);
  EXPECT_EQ(fit_ref, 0.5);
  // N.B: no weights, so theoretically exactly the same as fit_ref
  EXPECT_EQ(fit0, fit_ref);

  // there are some weights, so the fitness is different
  EXPECT_GT(fit, fit0);
  // computed with extra num and denum: the weight of Temperature (0.9), and the sum of weights 0.9+0.2
  EXPECT_EQ(fit, fitter_ref.computeFuzzySystemFitness(fitter_ref.getFuzzySystem(), fitter_ref.getFuzzySystemFitness(), 
    fitter_ref.fitMetrics(), 0.9, 1.1));

  // ==== with mandatory variables missing --> fitness == 0 ==============
  params.fitness_params.features_weights["Sunshine"] = 1;
  params.fitness_params.features_weights["Temperature"] = 0.9;

  FuzzyCocoFeaturesWeightsFitnessMethod fitter_missing(fs, fs_fitter, DFIN, DFOUT, params);
  init_fitter(fitter_missing);
  EXPECT_EQ(fitter_missing.fitnessImpl(), 0);

  // with used input variable used weight == 0 --> still a change because of an extra term in the denum
  params.fitness_params.features_weights["Sunshine"] = 0.9;
  params.fitness_params.features_weights["Temperature"] = 0;
  FuzzyCocoFeaturesWeightsFitnessMethod fitter_w0(fs, fs_fitter, DFIN, DFOUT, params);
  init_fitter(fitter_w0);

  EXPECT_EQ(fitter_w0.fitnessImpl(), fitter_ref.computeFuzzySystemFitness(fitter_ref.getFuzzySystem(), 
    fitter_ref.getFuzzySystemFitness(), fitter_ref.fitMetrics(), 0, 0.9));

  // ============ edge case : negative weights
  {
    params.fitness_params.features_weights["Sunshine"] = 0;
    params.fitness_params.features_weights["Temperature"] = -1000;
    FuzzySystemWeightedFitness fs_fitter(params.fitness_params.metrics_weights);
    EXPECT_THROW(FuzzyCocoFeaturesWeightsFitnessMethod fitter(fs, fs_fitter, DFIN, DFOUT, params), runtime_error);
  }
}


// Testing the features weight impact on fitness on rules using two input variables
TEST(FuzzyCocoFeaturesWeightsFitnessMethod, test2) {
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());
  // N.B: we set metrics_weights.nb_vars to 0 otherwise the fitness can not be equal to 1
  params.fitness_params.metrics_weights.nb_vars = 0;
  FuzzySystem fs(DFIN.colnames(), DFOUT.colnames(), params.input_vars_params.nb_sets, params.output_vars_params.nb_sets);
  FuzzySystemWeightedFitness fs_fitter(params.fitness_params.metrics_weights);

  FuzzyCocoFeaturesWeightsFitnessMethod fitter0(fs, fs_fitter, DFIN, DFOUT, params);
  FuzzyCocoFitnessMethod fitter_ref(fs, fs_fitter, DFIN, DFOUT, params);

  params.fitness_params.features_weights["Sunshine"] = 0.5;
  params.fitness_params.features_weights["Temperature"] = 0.5;
  FuzzyCocoFeaturesWeightsFitnessMethod fitter(fs, fs_fitter, DFIN, DFOUT, params);

  auto init_fitter = [](FuzzyCocoFitnessMethod& fitmeth) {
    auto& fcs = fitmeth.getFuzzySystem();
    fcs.getDB().setPositions(FUZZY_SYSTEM_112_POS);
    string RULES = R"(
      {
      "rule1":{
        "antecedents":{
          "Temperature":"Cold",
          "Sunshine": "Cloudy"
        },
        "consequents":{
          "Tourists":"High"
        }
      },
      "rule2":{
        "antecedents":{
          "Temperature":"Warm"
        },
        "consequents":{
          "Tourists":"Medium"
        }
      },
      "rule3":{
        "antecedents":{
          "Temperature":"Hot",
          "Sunshine": "Sunny"
        },
        "consequents":{
          "Tourists":"High"
        }
      }
    }
    )";
    fcs.setRules(FuzzyRule::loadRules(RULES, fcs.getDB()));
    fcs.setDefaultRule(FuzzyDefaultRule::load(R"("Tourists":"Low")", fcs.getDB()));
  };

  // ====== manual build of rules and mfs genomes ======
  init_fitter(fitter0);
  init_fitter(fitter_ref);
  init_fitter(fitter);

  // fitness!
  // logger().activate();
  double fit_ref = fitter_ref.fitnessImpl();
  double fit = fitter.fitnessImpl();
  double fit0 = fitter0.fitnessImpl();
  // logger().activate(false);

  // N.B: no weights, so theoretically exactly the same as fit_ref
  EXPECT_EQ(fit0, fit_ref);
  // there are some weights, so the fitness is different
  EXPECT_GT(fit, fit0);
  // computed with extra num and denum: the sum of weights 0.5+0.5==1
  EXPECT_EQ(fit, fitter_ref.computeFuzzySystemFitness(fitter_ref.getFuzzySystem(), fitter_ref.getFuzzySystemFitness(), 
    fitter_ref.fitMetrics(), 1, 1));


  // with one input variable used weight == 0 -->
  params.fitness_params.features_weights["Sunshine"] = 0.5;
  params.fitness_params.features_weights["Temperature"] = 0;

  FuzzyCocoFeaturesWeightsFitnessMethod fitter_w0(fs, fs_fitter, DFIN, DFOUT, params);
  init_fitter(fitter_w0);

  EXPECT_EQ(fitter_w0.fitnessImpl(), fitter_ref.computeFuzzySystemFitness(fitter_ref.getFuzzySystem(), 
    fitter_ref.getFuzzySystemFitness(), fitter_ref.fitMetrics(), 0.5, 0.5));
}