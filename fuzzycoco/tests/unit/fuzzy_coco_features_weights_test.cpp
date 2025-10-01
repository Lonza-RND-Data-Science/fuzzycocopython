#include "tests.h"
#include "fuzzy_coco.h"
#include "file_utils.h"
#include "logging_logger.h"
#include "digest.h"

using namespace fuzzy_coco;
using namespace FileUtils;
using namespace logging;
using namespace Digest;


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


string CSV = 
R"(Days;Temperature;Sunshine;Tourists
day1;19;25;55
day2;40;99;95
day3;24;NA;70
day4;5;3;2
)";

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
 

  // ================== using features_weights =============================
TEST_F(FuzzyCocoTest, features_weights) {

  { // first try without weights --> one rule with Temperature
    FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());
    // force to one rule
    params.global_params.nb_rules = 1;

    // without special params, the best rules are using the Temperature
    RandomGenerator rng(123);
    // params.global_params.max_generations = 20;
    FuzzyCoco coco(DFIN, DFOUT, params, rng);

    EXPECT_EQ(coco.getFitnessMethod().description(), "FuzzyCocoFitnessMethod");

    auto gen0 = coco.start(rng, false, 0.8);

    EXPECT_EQ(digest(gen0.left_gen), "c685520f7f126040");
    EXPECT_EQ(digest(gen0.left_gen.individuals), "d21393f22b177dc1");
    EXPECT_EQ(digest(gen0.left_gen.elite), "20f062c1dd2e9559");
    EXPECT_EQ(double_to_hex(gen0.left_gen.fitness), "3fd29c244fe2f34a");
    EXPECT_EQ(digest(gen0.left_gen.fitnesses), "6ef2cfbc3e5bad3b");
    // cerr << "gen0.left_gen.fitnesses:" << gen0.left_gen.fitnesses << endl;

    EXPECT_EQ(digest(gen0.right_gen), "49c1a33a4bc9b4e5");
    EXPECT_EQ(digest(gen0.right_gen.individuals), "2a26eb4d928138c8");
    EXPECT_EQ(digest(gen0.right_gen.elite), "0e52b12cca2f1902");
    EXPECT_EQ(double_to_hex(gen0.right_gen.fitness), "3fe2620ae4c415ca");
    EXPECT_EQ(digest(gen0.right_gen.fitnesses), "9eeec0a7b6f6c70f");
    // cerr << "gen0.right_gen.fitnesses:" << gen0.right_gen.fitnesses << endl;

    // // investigate right_gen
    // cerr << gen0.right_gen << endl;

    auto gen = gen0;
    vector<string> digests;
    digests.push_back(digest(gen0));
    // cerr << digests.back() << endl;


    for (int i = 0; i < 100; i++) {
      gen = coco.getEngine().run(gen, 1, 1);
      digests.push_back(digest(gen));
      // cerr << digests.back() << endl;
    }

    auto digest_all = digest(digests);
    // cerr << "digest_all=" << digest_all << endl;
    EXPECT_EQ(digests[0], "45bc4ea6e8bd95f6");
    EXPECT_EQ(digests[1], "502ccbadb3465dde");
    EXPECT_EQ(digests[2], "33aa7a6c05de5e0a");

    EXPECT_EQ(digest_all, "c63fce0ca6775740");

    // auto gen = coco.run(100, 1);

    auto rules = coco.getEngine().describeBestFuzzySystem()["fuzzy_system"]["rules"];
    EXPECT_EQ(rules.size(), 1);
    auto antecedents = rules[0]["antecedents"];
    // only one var in rule
    EXPECT_EQ(antecedents.size(), 1);
    // !! The unique input var in rules is Temperature
    EXPECT_EQ(antecedents[0].name(), "Temperature");
    // with perfect fitness
    EXPECT_EQ(gen.fitness, 1);

  }

  { 
    // now let's put some weight on Sunshine
    // N.B: that's the only difference with previous run
    FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols()); 
    // force to one rule
    params.global_params.nb_rules = 1;
    params.fitness_params.features_weights["Sunshine"] = 0.99;
    // params.fitness_params.features_weights["Temperature"] = 1;

    RandomGenerator rng(123);
    FuzzyCoco coco(DFIN, DFOUT, params, rng);

    EXPECT_EQ(coco.getFitnessMethod().description(), "FuzzyCocoFeaturesWeightsFitnessMethod");

    auto gen0 = coco.start(rng, false, 0.8);

    auto gen = gen0;
    vector<string> digests;
    digests.push_back(digest(gen0));
    // cerr << digests.back() << endl;
    for (int i = 0; i < 100; i++) {
      gen = coco.getEngine().run(gen, 1, 1);
      digests.push_back(digest(gen));
      // cerr << digests.back() << endl;
    }

    auto digest_all = digest(digests);
    // cerr << "digest_all=" << digest_all << endl;
    EXPECT_EQ(digests[0], "24eeaf0102401862");
    EXPECT_EQ(digests[1], "c8d75f442b7772fd");
    EXPECT_EQ(digests[2], "d2e02330950ddd1c");

    EXPECT_EQ(digest_all, "71bd32a3175eb06d");

    // auto gen = coco.run(100, 1);
// cerr << gen;
    auto rules = coco.describeBestFuzzySystem()["fuzzy_system"]["rules"];
    // cerr << rules;

    EXPECT_EQ(rules.size(), 1);

    auto antecedents = rules[0].get_list("antecedents");
    EXPECT_EQ(antecedents.size(), 1);  
    EXPECT_EQ(antecedents[0].name(), "Sunshine"); // it's sunshine!
    // all 2 input variables are now used! --> Sunshine is used too
    EXPECT_LT(coco.getFitnessMethod().getBestFitness(), 1); // lower fitness tho
  }

  { 
    // now with all weights set to 0. should be exactly the same as without weights
    FuzzyCocoParams params1 = GET_SAMPLE_PARAMS(DFIN.nbcols());
    RandomGenerator rng1(123), rng2(123);
    auto params2 = params1;
    params2.fitness_params.features_weights["Sunshine"] = 0;
    params2.fitness_params.features_weights["Temperature"] = 0;
    FuzzyCoco coco1(DFIN, DFOUT, params1, rng1);
    FuzzyCoco coco2(DFIN, DFOUT, params2, rng2);

    coco1.run(5, 1);
    coco2.run(5, 1);

    EXPECT_LT(coco1.getFitnessMethod().getBestFitness(), 1);
    EXPECT_EQ(coco1.getFitnessMethod().getBestFitness(), coco2.getFitnessMethod().getBestFitness());
  }

  {  // now let's force all vars using weights == 1
   
    FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());
    // force to one rule
    params.global_params.nb_rules = 1;

    params.fitness_params.features_weights["Sunshine"] = 1;
    params.fitness_params.features_weights["Temperature"] = 1;
    RandomGenerator rng(456);
    FuzzyCoco coco(DFIN, DFOUT, params, rng);

    coco.run(20, 1);
    auto desc = coco.describeBestFuzzySystem();
    // cerr << desc;

    auto rules = desc["fuzzy_system"]["rules"];
    EXPECT_EQ(rules.size(), 1);
    EXPECT_EQ(rules[0].size(), 2); // both vars are used
  }

  { 
    // now let's use 3 input variables (including the ouput) to predict the output
    auto dfin = DF.subsetColumns(0, DF.nbcols() - 1);
    auto dfout = DF.subsetColumns(DF.nbcols() - 1, DF.nbcols() - 1);

    FuzzyCocoParams params = GET_SAMPLE_PARAMS(dfin.nbcols());
    // params.global_params.influence_rules_initial_population = false;
    // force to one rule
    params.global_params.nb_rules = 1;
    params.fitness_params.features_weights["Sunshine"] = 1;
    params.fitness_params.features_weights["Temperature"] = 1;
    params.fitness_params.features_weights["Tourists"] = 1;

    RandomGenerator rng(345);
    FuzzyCoco coco(dfin, dfout, params, rng);

    auto gen = coco.run(1000, 0.1);
    cerr << gen.generation_number << endl;
    cerr << gen.fitness << endl;

    auto desc = coco.describeBestFuzzySystem();
    
    auto rules = desc["fuzzy_system"]["rules"];
    EXPECT_EQ(rules.size(), 1);
    EXPECT_EQ(rules[0]["antecedents"].size(), 3); // both vars are used
  }

  { // influence_rules_initial_population and speed of convergence

    auto dfin = DF.subsetColumns(0, DF.nbcols() - 1);
    auto dfout = DF.subsetColumns(DF.nbcols() - 1, DF.nbcols() - 1);

    FuzzyCocoParams params = GET_SAMPLE_PARAMS(dfin.nbcols());
    // params.global_params.influence_rules_initial_population = false;
    // force to one rule
    params.global_params.nb_rules = 10;
    params.fitness_params.features_weights["Temperature"] = 0.8;
 
    {
      RandomGenerator rng(345);
      FuzzyCoco coco(dfin, dfout, params, rng);
      auto gen = coco.run(100, 0.956, false);
      cerr << gen.generation_number << endl;
      cerr << gen.fitness << endl;
    }
    {
      RandomGenerator rng(345);
      FuzzyCoco coco(dfin, dfout, params, rng);
      auto gen = coco.run(100, 0.956, true, 1);
      cerr << gen.generation_number << endl;
      cerr << gen.fitness << endl;
    }

  }

}


TEST_F(FuzzyCocoTest2, features_weights_one_rule_one_var) {
  {
    // hand-made solution
    // cerr << DFIN << DFOUT;
    FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols()); 
    params.global_params.nb_rules = 1;
    params.fitness_params.output_vars_defuzz_thresholds = {0.5};

    params.input_vars_params.nb_bits_vars = 3;
    params.input_vars_params.nb_sets = 2;
    params.output_vars_params.nb_sets = 2;
    params.fitness_params.metrics_weights.nb_vars = 0;
    params.global_params.nb_max_var_per_rule = 1;

    {
      RandomGenerator rng(456);
      FuzzyCoco coco(DFIN, DFOUT, params, rng);

      // ====== manual build of optimal rules and mfs genomes ======
      // easy to reach max fitness with one rule of one var: "cause"
      auto& fs =  coco.getFuzzySystem();
      string POS = R"(
        {
          "ind1":{ "Low":1, "High":10 },
          "ind2":{ "Low":1, "High":10 },
          "constant":{ "Low":0, "High":3 },
          "cause":{ "Low":1, "High":10 },
          "effect":{ "Low":0, "High":1 },
        }
        )";
        fs.setVariablesSetPositions(POS);
      // cerr << fs.getDB();
      string optimal_rules_str = R"( { "rule1":{ "antecedents":{"cause":"Low" },"consequents":{ "effect":"High" } } } )";
      auto optimal_rules = FuzzyRule::loadRules(optimal_rules_str, fs.getDB());
      fs.setRules(optimal_rules);
      fs.setDefaultRule(FuzzyDefaultRule::load(R"("effect":"Low")", fs.getDB()));
    
      double fitness = coco.getFitnessMethod().fitnessImpl();
      EXPECT_EQ(fitness, 1);

      // now starting form scratch
      auto gen = coco.run(10000, 1);
      cerr << gen.generation_number << " " << gen.fitness << endl;

      EXPECT_EQ(gen.fitness, 1);
      EXPECT_LT(gen.generation_number, 350);

      // cerr << coco.describeBestFuzzySystem();
      coco.selectBestFuzzySystem();
      auto rules = coco.getFuzzySystem().getRules();
      EXPECT_EQ(rules, optimal_rules);
    }

    { // using weight==1 for constant, it is not possible to find an optimal rule
      RandomGenerator rng(456);
      auto params2 = params;
      params2.fitness_params.features_weights["constant"] = 1;
      FuzzyCoco coco(DFIN, DFOUT, params2, rng);
      auto gen = coco.run(1000, 1);
      EXPECT_EQ(gen.generation_number, 1000);
      EXPECT_LT(gen.fitness, 1);
    }

  }
}


TEST_F(FuzzyCocoTest2, features_weights_multiple_rules_one_var) {
  {

    FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols()); 
    params.global_params.nb_rules = 3;
    params.fitness_params.output_vars_defuzz_thresholds = {0.5};

    params.input_vars_params.nb_bits_vars = 3;
    params.input_vars_params.nb_sets = 2;
    params.output_vars_params.nb_sets = 2;
    params.fitness_params.metrics_weights.nb_vars = 0;
    params.global_params.nb_max_var_per_rule = 1;

    { 
      RandomGenerator rng(123);
      FuzzyCoco coco(DFIN, DFOUT, params, rng);

      auto gen = coco.run(300, 1);
      // cerr << gen;

      EXPECT_EQ(gen.fitness, 1);
      EXPECT_LE(gen.generation_number, 201);

      // cerr << coco.describeBestFuzzySystem();
      coco.selectBestFuzzySystem();
      auto rules = coco.getFuzzySystem().getRules();
      auto desc = FuzzyRule::describeRules(rules);
      EXPECT_EQ(rules.size(), 1); // only 1 rule (max 3)
      EXPECT_EQ(desc[0]["antecedents"][0].name(), "cause");
    }

    { // using high weights for useless variables, it slows down the convergence. The vars with heavy weights are indeed used
      RandomGenerator rng(654);
      auto params2 = params;
      params2.fitness_params.features_weights["ind1"] = 0.99;
      // params2.fitness_params.features_weights["ind2"] = 0.99;
      params2.fitness_params.features_weights["cause"] = 0.1;

      params2.fitness_params.metrics_weights.nb_vars = 0; // do not penalize vars

      FuzzyCoco coco(DFIN, DFOUT, params2, rng);
      auto gen = coco.run(300, 0.5);
      // cerr << gen;
  
      EXPECT_LT(gen.generation_number, 50);

      auto rules = coco.describeBestFuzzySystem()["fuzzy_system"]["rules"];
      // cerr << rules;

      EXPECT_EQ(rules.size(), 1); // 1 rule
      EXPECT_EQ(rules[0]["antecedents"][0].name(), "ind1");
      // set<string> vars; 
      // vars.insert(rules[0]["antecedents"][0].name());
      // vars.insert(rules[1]["antecedents"][0].name());
      // EXPECT_TRUE(vars.count("ind1") > 0);
      // EXPECT_TRUE(vars.count("ind2") > 0);
    }

  }
}

int count_genome_rules_with_antecedent(const Genome& rules_geno, int var_idx, FuzzyCocoCodec& codec) {
  vector<ConditionIndexes> rules_in, rules_out;
  vector<int> default_rules;
  codec.decode(rules_geno, rules_in, rules_out, default_rules);
  int nb = 0;
  for (const auto& cis : rules_in) {
    for (const auto& ci : cis) {
      if (ci.var_idx == var_idx) {
        nb++;
        break;
      }
    }
  }
  return nb;
}

int count_genomes_rules_with_antecedent(const Genomes& rules_genomes, int var_idx, FuzzyCocoCodec& codec) {
  int nb = 0;
  for (const auto& gen : rules_genomes)
    nb += (count_genome_rules_with_antecedent(gen, var_idx, codec) > 0);
  return nb;
}

TEST_F(FuzzyCocoTest, count_genomes_rules_with_antecedent) {
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());
  RandomGenerator rng(6423);
  FuzzyCoco coco(DFIN, DFOUT, params, rng);
  auto& codec = coco.getEngine().getFuzzyCocoCodec();

  Genome gen = codec.buildRulesGenome();
  int nb = count_genome_rules_with_antecedent(gen, 0, codec);
  EXPECT_EQ(nb, codec.getNbRules());

  nb = count_genome_rules_with_antecedent(gen, 1, codec);
  EXPECT_EQ(nb, 0);

  Genome gen2 = gen;
  codec.modifyRuleAntecedent(gen2, 0, 0, 1);
  EXPECT_EQ(count_genome_rules_with_antecedent(gen2, 1, codec), 1);

  Genome gen3 = gen2;
  codec.modifyRuleAntecedent(gen3, 1, 1, 1);
  EXPECT_EQ(count_genome_rules_with_antecedent(gen3, 1, codec),2);

  Genomes genos = {gen, gen2, gen3 };
  EXPECT_EQ(count_genomes_rules_with_antecedent(genos, 1, codec), 2);
  EXPECT_EQ(count_genomes_rules_with_antecedent(genos, 0, codec), 3);
}

TEST_F(FuzzyCocoTest, influence_rules_genomes) {
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());

  { // no weights
    RandomGenerator rng(6423);
    FuzzyCoco coco(DFIN, DFOUT, params, rng);

    // Genomes
    auto gen = coco.getEngine().start();
    Genomes before = gen.left_gen.individuals;
    Genomes after = before;
    FuzzyCoco::influence_rules_genomes(after, params.fitness_params.asFeaturesWeightsByIdx(DFIN.colnames()), 
      coco.getEngine().getFuzzyCocoCodec(), coco.getEngine().getRng());

    EXPECT_EQ(after, before);
  }

  { // one weight to 1 with evolving_ratio=1 --> all rules should use the variable
    RandomGenerator rng(6423);
    auto params2 = params;
    params2.fitness_params.features_weights["Sunshine"] = 0;
    params2.fitness_params.features_weights["Temperature"] = 1;

    FuzzyCoco coco(DFIN, DFOUT, params2, rng);
    auto& codec = coco.getEngine().getFuzzyCocoCodec();
    auto gen = coco.getEngine().start();
    Genomes before = gen.left_gen.individuals;
    Genomes after = before;

    FuzzyCoco::influence_rules_genomes(after, params2.fitness_params.asFeaturesWeightsByIdx(DFIN.colnames()), 
      coco.getEngine().getFuzzyCocoCodec(), coco.getEngine().getRng(), 1);


    int nb_genomes = before.size();
    // before, not all genomes were using var 0
    EXPECT_LT(count_genomes_rules_with_antecedent(before, 0, codec), nb_genomes); 

    // after: all genomes are using var 0
    EXPECT_EQ(count_genomes_rules_with_antecedent(after, 0, codec), nb_genomes); 

    // for var 1, should not have increased, but may have decreased by accident
    EXPECT_LE(count_genomes_rules_with_antecedent(after, 1, codec), count_genomes_rules_with_antecedent(before, 1, codec));

    // ====== now with very zero evolving_ratio ===========
    after = before;
    FuzzyCoco::influence_rules_genomes(after, params2.fitness_params.asFeaturesWeightsByIdx(DFIN.colnames()), 
    coco.getEngine().getFuzzyCocoCodec(), coco.getEngine().getRng(), 0);
    EXPECT_EQ(after, before);

    // ============= now with low evolving_ratio ====
    after = before;
    FuzzyCoco::influence_rules_genomes(after, params2.fitness_params.asFeaturesWeightsByIdx(DFIN.colnames()), 
    coco.getEngine().getFuzzyCocoCodec(), coco.getEngine().getRng(), 0.1);
    
    EXPECT_LT(count_genomes_rules_with_antecedent(before, 0, codec), count_genomes_rules_with_antecedent(after, 0, codec));
    EXPECT_LT(count_genomes_rules_with_antecedent(after, 0, codec), nb_genomes);
  }
}


TEST_F(FuzzyCocoTest2, influence_rules_genomes) {
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());
  cerr << DFIN;
  RandomGenerator rng(123);
  params.fitness_params.features_weights["ind1"] = 0.9;
  params.fitness_params.features_weights["ind2"] = 0.8;
  params.fitness_params.features_weights["cause"] = 1;

  params.rules_params.pop_size = 100;
  params.global_params.nb_max_var_per_rule = 1;
  // cerr << params;

  FuzzyCoco coco(DFIN, DFOUT, params, rng);
  auto& codec = coco.getEngine().getFuzzyCocoCodec();
  auto gen = coco.getEngine().start();
  Genomes before = gen.left_gen.individuals;
  Genomes after = before;

  FuzzyCoco::influence_rules_genomes(after, params.fitness_params.asFeaturesWeightsByIdx(DFIN.colnames()), 
    codec, coco.getEngine().getRng(), 1);

  // N.B: more rules for vars 0, 1, 3 because of weights
  EXPECT_LT(count_genomes_rules_with_antecedent(before, 0, codec), 
  count_genomes_rules_with_antecedent(after, 0, codec));
  EXPECT_LT(count_genomes_rules_with_antecedent(before, 1, codec), 
  count_genomes_rules_with_antecedent(after, 1, codec));
  EXPECT_LT(count_genomes_rules_with_antecedent(before, 3, codec), 
  count_genomes_rules_with_antecedent(after, 3, codec));

  // but var 2 has no weight, so has probably lost some rules
  EXPECT_GT(count_genomes_rules_with_antecedent(before, 2, codec), 
  count_genomes_rules_with_antecedent(after, 2, codec));

  // int nb_genomes = before.size();
  // for (int i = 0; i < 4; i++) {
  //   cerr << i << endl;
  //   cerr << count_genomes_rules_with_antecedent(before, i, codec) << endl;
  //   cerr << count_genomes_rules_with_antecedent(after, i, codec) << endl;
  //   cerr << endl;
  // }
}

TEST_F(FuzzyCocoTest2, features_weights_and_convergence) {

  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols()); 
  params.global_params.nb_rules = 1;
  params.rules_params.pop_size = 10;
  params.global_params.nb_max_var_per_rule = 1;

  params.fitness_params.output_vars_defuzz_thresholds = {0.5};

  params.input_vars_params.nb_bits_vars = 3;
  params.input_vars_params.nb_sets = 2;
  params.output_vars_params.nb_sets = 2;
  params.fitness_params.metrics_weights.nb_vars = 0;

  // params.rules_params.elite_size = params.rules_params.pop_size / 2;

  // example when encouraging a bad variable: the influence of initial pop greatly speeds up the convergence
  // =========================================================================================================
  {
    double fit1, fit2;
    auto params2 = params;
    params2.fitness_params.features_weights["ind1"] = 0.8;
    // params2.fitness_params.features_weights["ind2"] = 0;
    // params2.fitness_params.features_weights["cause"] = 0;
    // params2.fitness_params.features_weights["constant"] = 0;

    {
      RandomGenerator rng(123);
      FuzzyCoco coco(DFIN, DFOUT, params2, rng);
      auto gen = coco.run(100, .6);
      // cerr << gen;
      fit1 = gen.fitness;
      // cerr << coco.describeBestFuzzySystem()["fuzzy_system"]["rules"];
    }

    {     
      RandomGenerator rng(123);
      FuzzyCoco coco(DFIN, DFOUT, params2, rng);
      // N.B: use another dedicated rng to not perturb the algorithm
      RandomGenerator rng2(123);
      auto gen = coco.start(rng2, true, 0.6);
      auto gen2 = coco.run(100, .6, gen);
      fit2 = gen2.fitness;
    }

    // nothing really to test....
    EXPECT_GT(fit1, 0.6);
    EXPECT_GT(fit2, 0.6);
  }

    // example when encouraging a good variable
    // actually not that obvious
  // =========================================================================================================
  {
    double nb1, nb2;
    auto params2 = params;

    params2.global_params.nb_rules = 1;
    params2.global_params.nb_max_var_per_rule = 2;


    params2.fitness_params.features_weights["cause"] = 1;
    params2.fitness_params.features_weights["constant"] = 0.1;

    {
      RandomGenerator rng(234);
      FuzzyCoco coco(DFIN, DFOUT, params2, rng);
      auto gen = coco.run(100, .95);
      cerr << gen;
  
      nb1 = gen.generation_number;
      cerr << coco.describeBestFuzzySystem()["fuzzy_system"]["rules"];
    }

    {     
      RandomGenerator rng(234);
      FuzzyCoco coco(DFIN, DFOUT, params2, rng);
      // N.B: use another dedicated rng to not perturb the algorithm
      RandomGenerator rng2(234);
      auto gen0 = coco.start(rng2, true, 0.8);
      auto gen = coco.run(100, .95, gen0);
      cerr << coco.describeBestFuzzySystem()["fuzzy_system"]["rules"];
      cerr << gen;
      nb2 = gen.generation_number;
    }

    EXPECT_GT(nb1, nb2);
  }


}