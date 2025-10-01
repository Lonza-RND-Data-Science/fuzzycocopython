#include <fstream>
#include <sstream>
#include "tests.h"
#include "fuzzy_coco.h"
#include "file_utils.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace FileUtils;
using namespace logging;



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

 // a synthetic dataset with 2 independent variables, one constant, one binary outcome, and one highly correlated var
 string CSV2 = 
R"(obs;ind1;ind2;constant;cause;effect
obs1;1;10;3;1;1
obs2;2;9;3;10;0
obs3;3;8;3;2;1
obs4;4;7;3;9;0
obs5;5;6;3;3;1
obs6;6;5;3;8;0
obs7;7;4;3;4;1
obs8;8;3;3;7;0
obs9;9;2;3;5;1
obs10;10;1;3;6;0
)";

class FuzzyCocoTest2 : public testing::Test {
protected:
  FuzzyCocoTest2() : DF(CSV2, true) {}

  void SetUp() override {
    int nb = DF.nbcols();
    DFIN = DF.subsetColumns(0, nb - 2);
    DFOUT = DF.subsetColumns(nb - 1, nb - 1);
  }

  DataFrame DF, DFIN, DFOUT;
};
 


TEST_F(FuzzyCocoTest, run) {
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());
  RandomGenerator rng(6423);
  FuzzyCoco coco(DFIN, DFOUT, params, rng);
  // cerr << coco;

  auto gen = coco.run();

  auto [best_rule, best_mf] = coco.getEngine().getBest();
  cerr << endl << "best pair: " << endl;
  cerr << best_rule << endl << best_mf << endl;

  coco.getFitnessMethod().setRulesGenome(best_rule);
  coco.getFitnessMethod().setMFsGenome(best_mf);

  double fit = coco.getFitnessMethod().fitnessImpl(best_rule, best_mf);

  // describeBestFuzzySystem
  auto desc = coco.getEngine().describeBestFuzzySystem();
  // cerr << desc;
  // cerr << params;
  EXPECT_EQ(desc.names(), vector<string>({"fitness", "fitness_metrics_weights", "fuzzy_system", "defuzz_thresholds"}));
  EXPECT_DOUBLE_EQ(fit, desc.get_double("fitness"));
}


TEST_F(FuzzyCocoTest, searchBestFuzzySystem) {
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());

  params.global_params.max_fitness = 0.6;

  auto desc = FuzzyCoco::searchBestFuzzySystem(DF, DFOUT.nbcols(), params, 6423);
  // cerr << desc;
  // cerr << params;
  EXPECT_EQ(desc.names(), vector<string>({"fit",  "fuzzy_system", "params"}));
  EXPECT_GT(desc["fit"].get_double("fitness"), 0.6);
  EXPECT_LT(desc["fit"].get_int("generations"), 15);

  // reproducible using FuzzyCoco::eval
  FuzzySystem fs2 = FuzzySystem::load(desc["fuzzy_system"]);
  auto params2 = FuzzyCocoParams(desc["params"]);

  auto res = FuzzyCoco::eval(DF, fs2, params2); 

  EXPECT_EQ(res["fitness"], desc["fit"]["fitness"]);
  EXPECT_EQ(res["metrics"], desc["fit"]["metrics"]);

  // predict
  auto predicted = FuzzyCoco::loadAndPredict(DF, desc);
  EXPECT_EQ(predicted, fs2.smartPredict(DF));


  // simulating best fitness == 0
  {
    params.global_params.max_generations = 0;
    params.rules_params.pop_size = 1;
    params.rules_params.elite_size = 1;
    params.mfs_params.pop_size = 1;
    params.mfs_params.elite_size = 1;
    // set weigths so that fitness == 0
    params.fitness_params.metrics_weights.sensitivity = 0;
    params.fitness_params.metrics_weights.specificity = 0;
    params.fitness_params.metrics_weights.nb_vars = 0;
    params.fitness_params.metrics_weights.ppv = 1;


    auto desc = FuzzyCoco::searchBestFuzzySystem(DF, DFOUT.nbcols(), params, 6423);
    // cerr << desc;
    EXPECT_TRUE(desc.empty());
    // abort();
  }

}

TEST_F(FuzzyCocoTest, predict_save_load) {
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());
  RandomGenerator rng(6423);
  FuzzyCoco coco(DFIN, DFOUT, params, rng);

  auto gen = coco.run(1000, 0.7);
// cerr << gen;
  coco.selectBestFuzzySystem();
  EXPECT_GT(coco.getFitness(), 0.7);
  double fitness = coco.getFitness() ;
  // cerr << fitness << endl;

  cerr << coco; // for coverage

  auto df1 = coco.predict(DFIN);

  auto metrics = coco.getFitnessMethod().computeMetrics(df1, DFOUT);
  cerr << metrics << endl;

  double fitness2 = coco.getFitnessMethod().computeFuzzySystemFitness(coco.getFuzzySystem(), coco.getFuzzySystemFitness(), metrics, 0, 0);
  EXPECT_EQ(fitness2, fitness);

  {
    auto fs = coco.getFuzzySystem();

    string temp_fuzzy_system = poor_man_tmpnam("predict_save_load");
cerr << temp_fuzzy_system << endl;
    ofstream fs_out(temp_fuzzy_system);
    fs_out << coco.describe(-1);
    fs_out.close();

    // ================== loadFuzzyFile ===========================

    auto loaded = FuzzyCoco::loadFuzzyFile(temp_fuzzy_system);
    EXPECT_EQ(loaded.names(), vector<string>({"fit", "fuzzy_system", "params"}));
  
    // ================== loadAndPredict ===========================

    auto df2 = FuzzyCoco::loadAndPredict(DFIN, temp_fuzzy_system);
    EXPECT_EQ(df2, FuzzyCoco::loadAndPredict(DFIN, loaded));

    // ================== evalAndSave ===========================
    ostringstream oss;
    FuzzyCoco::evalAndSave(DF, temp_fuzzy_system, oss);
    auto eval1 = NamedList::parse(oss.str());
    auto eval2 = FuzzyCoco::eval(DF, FuzzySystem::load(loaded["fuzzy_system"]), FuzzyCocoParams(loaded["params"]));

    EXPECT_TRUE(cmp(eval1, eval2));


    remove(temp_fuzzy_system);
  }
}


