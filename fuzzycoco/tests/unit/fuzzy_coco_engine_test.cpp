#include "tests.h"
#include "fuzzy_coco_engine.h"
#include "file_utils.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace FileUtils;
using namespace logging;

// NB: the main feature are tested via FuzzyCoco. Here we only test the custom fitness methods

string CSV = 
R"(Days;Temperature;Sunshine;Tourists
day1;19;25;55
day2;40;99;95
day3;24;NA;70
day4;5;3;2
)";

FuzzyCocoParams GET_SAMPLE_PARAMS(int nb_max_var_per_rule) {
  FuzzyCocoParams params;

  params.global_params.nb_rules = 3;
  params.global_params.nb_max_var_per_rule = nb_max_var_per_rule;
  params.global_params.nb_cooperators = 2;

  auto& ip = params.input_vars_params;
  ip.nb_sets = 3;
  // ip.nb_bits_vars = 2;
  // ip.nb_bits_sets = 2;
  ip.nb_bits_pos = 8;

  auto& op = params.output_vars_params;
  op.nb_sets = 3;
  // op.nb_bits_vars = 2;
  // op.nb_bits_sets = 2;
  op.nb_bits_pos = 2;

  params.fitness_params.metrics_weights.sensitivity = 1;
  params.fitness_params.metrics_weights.specificity = 1;
  params.fitness_params.metrics_weights.nb_vars = 0.35;

  // !!!
  params.fitness_params.output_vars_defuzz_thresholds.push_back(50);
  
  params.rules_params.pop_size = 10;
  params.mfs_params.pop_size = 20;

  params.evaluate_missing(nb_max_var_per_rule, 1);

  return params;
}

class FuzzyCocoTest : public testing::Test {
protected:
  FuzzyCocoTest() : DF(CSV, true) {}

  void SetUp() override {
    DFIN = DF.subsetColumns(0, DF.nbcols() - 2);
    DFOUT = DF.subsetColumns(DF.nbcols() - 1, DF.nbcols() - 1);
  }
 
    DataFrame DF, DFIN, DFOUT;
 };

// testing the use of a custom FuzzySystemFitness class with FuzzyCoco
// this class just adds a constant to the fitness computed on the FuzzySystemMetrics
class CustomFuzzySystemFitness : public FuzzySystemWeightedFitness {
public:
  CustomFuzzySystemFitness(const FuzzySystemMetrics& weights, double constant) : 
    FuzzySystemWeightedFitness(weights), _constant(constant) {}
  double fitness(const FuzzySystemMetrics& metrics, double extra_num = 0, double extra_denum = 0) override {
    // cerr << "CustomFuzzySystemFitness::fitness()\n";
    return FuzzySystemWeightedFitness::fitness(metrics) + _constant;
  }
private:
  double _constant;
};

TEST_F(FuzzyCocoTest, custom_fitness) {
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());
  FuzzySystem fs(DFIN.colnames(), DFOUT.colnames(), params.input_vars_params.nb_sets, params.output_vars_params.nb_sets);

  CustomFuzzySystemFitness custom_fs_fitter(params.fitness_params.metrics_weights, 1000);
  FuzzyCocoFitnessMethod custom_coco_fitter(fs, custom_fs_fitter, DFIN, DFOUT, params);
  RandomGenerator rng(6423), rng2(6423);
  FuzzyCocoEngine custom_cocoe(DFIN, DFOUT, custom_coco_fitter, params, rng2);

  FuzzySystemWeightedFitness fs_fitter(params.fitness_params.metrics_weights);
  FuzzyCocoFitnessMethod coco_fitter(fs, fs_fitter, DFIN, DFOUT, params);
  FuzzyCocoEngine cocoe(DFIN, DFOUT, coco_fitter, params, rng);

 
  auto gen = cocoe.run();
  auto [best_rule, best_mf] = cocoe.getBest();
  EXPECT_LT(gen.fitness, 1.1);

  cerr << cocoe; // for coverage, to exercise <<

  auto gen2 = custom_cocoe.run();
  EXPECT_GT(gen2.fitness, 1000);
}



// testing the use of a custom FuzzySystemMethod class with FuzzyCoco
// this class adds some terms related to the actual input variables used in the rules of the current fuzzy system

class CustomFuzzyCocoFitnessMethod : public FuzzyCocoFitnessMethod {
public:
  CustomFuzzyCocoFitnessMethod(FuzzySystem& fs, FuzzySystemFitness& fitter, const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params) 
    : FuzzyCocoFitnessMethod(fs, fitter, dfin, dfout, params) {}
  
    double fitnessImpl(const Genome& rules_genome, const Genome& vars_genome) override {
      // double fit0 = FuzzyCocoFitnessMethod::fitnessImpl(rules_genome, vars_genome);
      auto used_invars = getFuzzySystem().getUsedInputVariables();
      // cerr << "used_invars=" << used_invars << endl;

      setRulesGenome(rules_genome);
      setMFsGenome(vars_genome);
      if (!getFuzzySystem().ok()) return 0;

      double extra_num = sum(used_invars);
      double extra_denum = sum(used_invars) + 1;

      return getFuzzySystemFitness().fitness(fitMetrics(), extra_num,  extra_denum);
    }
  
};

TEST_F(FuzzyCocoTest, custom_fitness_method) {
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());
  FuzzySystem fs(DFIN.colnames(), DFOUT.colnames(), params.input_vars_params.nb_sets, params.output_vars_params.nb_sets);

  FuzzySystemWeightedFitness fs_fitter(params.fitness_params.metrics_weights);

  CustomFuzzyCocoFitnessMethod custom_fitter(fs, fs_fitter, DFIN, DFOUT, params);
  FuzzyCocoFitnessMethod ref_fitter(fs, fs_fitter, DFIN, DFOUT, params);

  RandomGenerator rng(6423), rng2(6423);

  FuzzyCocoEngine cocoe_ref(DFIN, DFOUT, ref_fitter, params, rng);
  FuzzyCocoEngine cocoe_custom(DFIN, DFOUT, custom_fitter, params, rng2);

  auto ref_gen = cocoe_ref.run();
  auto custom_gen = cocoe_custom.run();
  EXPECT_LT(custom_gen.fitness, ref_gen.fitness);
}

