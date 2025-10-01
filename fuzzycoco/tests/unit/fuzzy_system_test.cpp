#include "tests.h"
#include <algorithm>
#include "file_utils.h"
#include "fuzzy_system.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace FileUtils;
using namespace logging;

string CSV = 
R"(Days;Temperature;Sunshine;Tourists
day1;19;20;17
day2;40;99;95
day3;24;NA;70
day4;5;3;2
)";


// cf coco book pp26 fig 1.12
string DB_BOOK_112 = R"(
    {
  "input":{
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
  },
  "output":{
    "Tourists":{
      "Low":0.0,
      "Medium":50.0,
      "High":100.0
    }
  }
}
)";

class FuzzySystemTest112 : public testing::Test {
 protected:
  FuzzySystemTest112() : FS_NO_THRESHOLD(FuzzyVariablesDB::load(DB_BOOK_112)) {}

  void SetUp() override {

    vector<vector<string>> tokens;
    parseCSV(CSV, tokens);
    DF.assign(tokens, true);

    DFIN = DF.subsetColumns(0, DF.nbcols() - 2);
    DFOUT = DF.subsetColumns(DF.nbcols() - 1, DF.nbcols() - 1);

  }

  FuzzySystem FS_NO_THRESHOLD;
  DataFrame DF, DFIN, DFOUT;
};

class FuzzySystemTestNoThreshold : public FuzzySystemTest112 {
  protected:
    // inherit constructors
    using FuzzySystemTest112::FuzzySystemTest112;

    void SetUp() override {
      FuzzySystemTest112::SetUp();
      auto& fs = FS_NO_THRESHOLD;
      fs.resetRules(3);
      fs.addRule( FuzzyRule::load(R"( {
      "antecedents":{ "Temperature":"Cold", "Sunshine":"Cloudy"  },
      "consequents":{  "Tourists":"Low" }
      })", fs.getDB()) );

      fs.addRule( FuzzyRule::load(R"( {
      "antecedents":{ "Temperature":"Warm", "Sunshine":"PartSunny"  },
      "consequents":{  "Tourists":"Medium"  }
      })", fs.getDB()) );
      
      fs.addRule( FuzzyRule::load(R"( {
        "antecedents":{ "Temperature":"Hot", "Sunshine":"Sunny"  },
        "consequents":{  "Tourists":"High"  }
        })", fs.getDB()) );
  
    
      fs.getDB().setPositions(R"(
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
      },
      "Tourists":{
        "Low":0.0,
        "Medium":50.0,
        "High":100.0
      }
      })");
  }
};

TEST_F(FuzzySystemTest112, basic) {

  FuzzySystem fs = FS_NO_THRESHOLD;
  const auto& db = fs.getDB();
  EXPECT_EQ(db.getNbInputVars(), DFIN.nbcols());
  EXPECT_EQ(db.getNbOutputVars(), DFOUT.nbcols());
  EXPECT_EQ(fs.getNbRules(), 0);
  // EXPECT_FALSE(fs.areDefuzzThresholdsEnabled());

  // === default rules ===
  const auto& v = fs.getDefaultRulesOutputSets();
  EXPECT_TRUE(all_zero(v));

  vector<int> defs = {2};
  fs.setDefaultRulesConditions(defs);
  EXPECT_EQ(fs.getDefaultRulesOutputSets(), defs);

  // ======== addRule()
  vector<ConditionIndexes> in_cis = { {{0, 0}, {1, 0}}, {{0, 1}, {1, 1}}, {{0, 2}, {1, 2}} };
  vector<ConditionIndexes> out_cis = { {{0, 0}}, {{0, 1}},  {{0, 2}} };
  fs.resetRules(3);
  FuzzyRule rule1{db, {{0, 0}, {1, 0}}, {{0, 0}}};
  fs.addRule(rule1);
  EXPECT_EQ(fs.getRule(0).getInputConditionIndexes(), ConditionIndexes({{0, 0}, {1, 0}}));
  EXPECT_EQ(fs.getRule(0).getOutputConditionIndexes(), ConditionIndexes({{0, 0}}));

  // ======== setVariablesSetPositions()
  Matrix<double> insets_pos_mat = { {17, 20, 29}, {30, 50, 100} };
  Matrix<double> outsets_pos_mat = { {0, 50, 100} };
  fs.setVariablesSetPositions(insets_pos_mat, outsets_pos_mat);

  EXPECT_EQ(db.fetchInputPositions(),  insets_pos_mat);
  EXPECT_EQ(db.fetchOutputPositions(),  outsets_pos_mat);


  // describe and <<
  auto vars = fs.getUsedInputVariables();
  EXPECT_EQ(vars.size(), 2);

  fs.resetRules();
  fs.addRule({db, {{0,0}} , { {0,0}} });

  // fs.setRulesConditions({ {{0,0}} }, { {{0,0}} });
  vars = fs.getUsedInputVariables();
  EXPECT_EQ(vars, vector<int>({0}));

  {
    auto vars = fs.getUsedOutputVariables();
    EXPECT_EQ(vars, vector<int>({0}));
  }

  auto desc = fs.describe();
  cerr << desc;
  EXPECT_EQ(desc.names(), vector<string>({"variables", "rules", "default_rules"}));

  // cerr << fs;
  FuzzySystem fs2 = FuzzySystem::load(desc);
  EXPECT_EQ(fs2, fs);
}

TEST(utils, reset) {
  // just a work-around for a test coverage bug
  vector<int> vec;
  reset(vec, 0);
}

TEST_F(FuzzySystemTestNoThreshold, computeRulesFireLevels) {
  logger().activate(false);

  FuzzySystem fs = FS_NO_THRESHOLD;

  // ======== computeRulesFireLevels ==========
  // NA are ignored if there are other rules with non-missing values
  vector<double> expected{1, 0, 0};
  EXPECT_EQ(fs.computeRulesFireLevels({0, 0}), expected);
  EXPECT_EQ(fs.computeRulesFireLevels({MISSING_DATA_DOUBLE, 0}), expected);
  EXPECT_EQ(fs.computeRulesFireLevels({0, MISSING_DATA_DOUBLE}), expected);
  expected = {0, 0, 1};
  EXPECT_EQ(fs.computeRulesFireLevels({100, 100}), expected);
  EXPECT_EQ(fs.computeRulesFireLevels({MISSING_DATA_DOUBLE, 100}), expected);
  EXPECT_EQ(fs.computeRulesFireLevels({100, MISSING_DATA_DOUBLE}), expected);
  expected = {0, 1, 0};
  EXPECT_EQ(fs.computeRulesFireLevels({20, 50}), expected);
  EXPECT_EQ(fs.computeRulesFireLevels({MISSING_DATA_DOUBLE, 50}), expected);
  EXPECT_EQ(fs.computeRulesFireLevels({20, MISSING_DATA_DOUBLE}), expected);

  // adapted from book pp26
  expected = {0, 2.0/3, 0};
  EXPECT_EQ(fs.computeRulesFireLevels({19, 60}), expected);

  // === missingness
  // all NA ?
  expected = {MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE};
  EXPECT_EQ(fs.computeRulesFireLevels({MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE}), expected);

  // test the computeRulesFireLevels(sample_idx)
  vector<double> fire_levels;
  for (int row = 0; row < DFIN.nbrows(); row++) {
    fs.computeRulesFireLevels(row, DFIN, fire_levels);
    EXPECT_EQ(fire_levels, fs.computeRulesFireLevels(DFIN.fetchRow(row)));
  }
}

TEST_F(FuzzySystemTestNoThreshold, computeRulesImplications) {
  vector<double> fire_levels = {0, 2.0/3, 0};
  Matrix<double> res(1, 3);
  Matrix<double> expected = { {0, 2.0/3, 0} };

  FuzzySystem fs = FS_NO_THRESHOLD;
  // ======== computeRulesImplications ==========
  fs.computeRulesImplications(fire_levels, res);
  EXPECT_EQ(res, expected);

  // use custom fire levels
  fire_levels = { 0.2, 0.4, 1};
  expected = { fire_levels };
  fs.computeRulesImplications(fire_levels, res);
  EXPECT_EQ(res, expected);

  // impact of missingness --> a rule that fires NA should be ignored !
  fire_levels = { MISSING_DATA_DOUBLE, 0.4, 1};
  expected = { {0, 0.4, 1} };
  fs.computeRulesImplications(fire_levels, res);
  EXPECT_EQ(res, expected);

  fire_levels = { MISSING_DATA_DOUBLE, 0.4, MISSING_DATA_DOUBLE};
  expected = { {0, 0.4, 0} };
  fs.computeRulesImplications(fire_levels, res);
  EXPECT_EQ(res, expected);

  fire_levels = { MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE};
  expected = { {0, 0, 0} };
  fs.computeRulesImplications(fire_levels, res);
  EXPECT_EQ(res, expected);

  // === now let's change the output rules so that they all use the same output set
  fs.getRule(0).setOutputConditions({{0, 1}});
  fs.getRule(1).setOutputConditions({{0, 1}});
  fs.getRule(2).setOutputConditions({{0, 1}});

  // now all rules impact the SAME output set
  fire_levels = { 0.2, 0.4, 1};
  expected = { {0, 0.2 + 0.4 + 1, 0} };
  fs.computeRulesImplications(fire_levels, res);
  EXPECT_EQ(res, expected); 

  fire_levels = { 0.2, 0.4, MISSING_DATA_DOUBLE};
  expected = { {0, 0.2 + 0.4, 0} };
  fs.computeRulesImplications(fire_levels, res);
  EXPECT_EQ(res, expected); 
}

TEST(FuzzySystem, computeOutputVarsMaxFireLevels) {

  auto fs = FuzzySystem::load(R"(
    {
  "variables":{
    "input":{
      "in_1":{
        "in_1.1":0.0,
        "in_1.2":5.0,
        "in_1.3":10.0
      },
      "in_2":{
        "in_2.1":0.0,
        "in_2.2":50.0,
        "in_2.3":100.0
      },
      "in_3":{
        "in_3.1":0.0,
        "in_3.2":500.0,
        "in_3.3":1000.0
      }
    },
    "output":{
      "out_1":{
        "out_1.1":0.0,
        "out_1.2":100.0
      },
      "out_2":{
        "out_2.1":0.0,
        "out_2.2":10.0
      }
    }
  },
  "rules":{
    "rule1":{
      "antecedents":{
        "in_1":"in_1.1",
        "in_2":"in_2.1",
        "in_3":"in_3.1"
      },
      "consequents":{
        "out_1":"out_1.1"
      }
    },
    "rule2":{
      "antecedents":{
        "in_1":"in_1.3",
        "in_2":"in_2.3",
        "in_3":"in_3.3"
      },
      "consequents":{
        "out_1":"out_1.2"
      }
    }
  },
  "default_rules":{
    "out_1":"out_1.1",
    "out_2":"out_2.1"
  }
  } )");


  vector<double> fire_levels = { 0.2, 0.4};
  vector<double> res;

  fs.computeOutputVarsMaxFireLevels(fire_levels, res);
  // all rules have fired and contributed to the same output var 0.0
  EXPECT_DOUBLE_EQ(res[0], max(0.2, 0.4));
  // no rule contributed to the 
  EXPECT_TRUE(is_na(res[1]));

  //
  fire_levels = { MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE};
  fs.computeOutputVarsMaxFireLevels(fire_levels, res);
  EXPECT_TRUE(is_na(res[0]));
  EXPECT_TRUE(is_na(res[1]));

  // change the rules out conditions
  fs.getRule(0).setOutputConditions( {{0, 0}, {0, 1}} );
  fs.getRule(1).setOutputConditions( {{1, 1}, {1, 0}} );

  fire_levels = { 0.2, 0.4 };
  fs.computeOutputVarsMaxFireLevels(fire_levels, res);
  EXPECT_DOUBLE_EQ(res[0], 0.2);
  EXPECT_DOUBLE_EQ(res[1], 0.4);
}

TEST(FuzzySystem, defaultRules_addDefaultRulesImplications) {
  FuzzySystem fs(3, 3, 5, 5);

  vector<int> osets = {0, 1, 2, 3, 4};
  fs.setDefaultRulesConditions(osets);
  EXPECT_EQ(fs.getDefaultRulesOutputSets(), osets);
  
  auto desc = fs.describe();
  // cerr << desc;
  FuzzySystem fs2 = FuzzySystem::load(desc);
  EXPECT_EQ(fs2, fs);

  vector<double> max_fire_levels = { 0.1, 0.2, 0.3, 0.4, 0.5};
  Matrix<double> res(5, 5);
  fs.addDefaultRulesImplications(fs.getDefaultRulesOutputSets(), max_fire_levels, res);

  for (int i = 0; i < 5; i++)
    EXPECT_DOUBLE_EQ(res[i][i], 1 - max_fire_levels[i]);
  
  // it adds up
  fs.addDefaultRulesImplications(fs.getDefaultRulesOutputSets(), max_fire_levels, res);
  for (int i = 0; i < 5; i++)
    EXPECT_DOUBLE_EQ(res[i][i], 2 * (1 - max_fire_levels[i]));

  // === missingness
  max_fire_levels = { MISSING_DATA_DOUBLE, 0.2, 0.3, 0.4, MISSING_DATA_DOUBLE};
  res.reset();
  fs.addDefaultRulesImplications(fs.getDefaultRulesOutputSets(), max_fire_levels, res);

  cerr << res;
  for (int i = 1; i < 4; i++)
    EXPECT_DOUBLE_EQ(res[i][i], 1 - max_fire_levels[i]);
  EXPECT_DOUBLE_EQ(res[0][0], 0);
  EXPECT_DOUBLE_EQ(res[4][4], 0);

  // all missing
  max_fire_levels = { MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE};
  res.reset();
  fs.addDefaultRulesImplications(fs.getDefaultRulesOutputSets(), max_fire_levels, res);
  for (int i = 0; i < 5; i++)
    EXPECT_DOUBLE_EQ(res[i][i], 0);
}

TEST(FuzzySystem, defuzzify) {
  FuzzySystem fs(1, 1, 2, 2);
  Matrix<double> res(2, 2);
  // N.B: inputs are not needed for that test
  fs.setVariablesSetPositions({ {0} }, { {1, 100}, {1, 10} });
  vector<double> defuzz;

  // zero Matrix<double> --> no rule fired
  fs.defuzzify(res, defuzz);
  EXPECT_TRUE(all_zero(defuzz));

  // normal
  res = { {1, 2}, {10, 20} };
  fs.defuzzify(res, defuzz);
  EXPECT_DOUBLE_EQ(defuzz[0], (1.0*1 + 2*100) / (1.0 + 2));
  EXPECT_DOUBLE_EQ(defuzz[1], (10.0*1 + 20*10) / (10.0 + 20));

  // missingness
  res = { {1, MISSING_DATA_DOUBLE}, {10, 20} };
  fs.defuzzify(res, defuzz);
  EXPECT_DOUBLE_EQ(defuzz[0], (1.0*1 ) / (1.0 ));
  EXPECT_DOUBLE_EQ(defuzz[1], (10.0*1 + 20*10) / (10.0 + 20));
 
  res = { {MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE}, {10, 20} };
  fs.defuzzify(res, defuzz);
  EXPECT_TRUE(is_na(defuzz[0]));
  EXPECT_DOUBLE_EQ(defuzz[1], (10.0*1 + 20*10) / (10.0 + 20));
}

// TEST(FuzzySystem, defuzz_thresholds) {
//   vector<double> thresholds = { 0.7, 0.9};
//   FuzzySystem fs(1, 1, 2, 2, thresholds);

//   // === apply_threshold ==
//   EXPECT_DOUBLE_EQ(FuzzySystem::apply_threshold(0.5, -1), MISSING_DATA_DOUBLE);
//   EXPECT_DOUBLE_EQ(FuzzySystem::apply_threshold(0.5, 0), 0);
//   EXPECT_DOUBLE_EQ(FuzzySystem::apply_threshold(0.5, 0.1), 0);
//   EXPECT_DOUBLE_EQ(FuzzySystem::apply_threshold(0.5, 0.49), 0);

//   EXPECT_DOUBLE_EQ(FuzzySystem::apply_threshold(0.5, 0.5), 1);
//   EXPECT_DOUBLE_EQ(FuzzySystem::apply_threshold(0.5, 0.8), 1);
//   EXPECT_DOUBLE_EQ(FuzzySystem::apply_threshold(0.5, 1), 1);
//   EXPECT_DOUBLE_EQ(FuzzySystem::apply_threshold(0.5, 10), 1);

//   // missingness
//   EXPECT_DOUBLE_EQ(FuzzySystem::apply_threshold(0.5, MISSING_DATA_DOUBLE), MISSING_DATA_DOUBLE);

//   // ====== threshold_defuzzed_value ====
//   EXPECT_DOUBLE_EQ(fs.threshold_defuzzed_value(0, 0.6), 0);
//   EXPECT_DOUBLE_EQ(fs.threshold_defuzzed_value(0, 0.8), 1);
//   EXPECT_DOUBLE_EQ(fs.threshold_defuzzed_value(1, 0.8), 0);
//   EXPECT_DOUBLE_EQ(fs.threshold_defuzzed_value(1, 0.91), 1);

// }

TEST_F(FuzzySystemTestNoThreshold, predictSample) {
  FuzzySystem fs = FS_NO_THRESHOLD;
  vector<double> defuzzed;

  cerr << DFIN;

  // for (int row = 0; row < DFIN.nbrows(); row++) {
  //   fs.predictSample(row, DFIN, defuzzed);
  // }

  fs.predictSample(0, DFIN, defuzzed);
  EXPECT_DOUBLE_EQ(defuzzed[0], 0);
  fs.predictSample(1, DFIN, defuzzed);
  EXPECT_TRUE(defuzzed[0] >= 95);
  fs.predictSample(2, DFIN, defuzzed);
  EXPECT_DOUBLE_EQ(defuzzed[0], 50);
  fs.predictSample(3, DFIN, defuzzed);
  EXPECT_FALSE(defuzzed[0] >= 5); 
}

TEST_F(FuzzySystemTestNoThreshold, predict) {
  FuzzySystem fs = FS_NO_THRESHOLD;

  auto predicted = fs.predict(DFIN);
  NumColumn expected = {0, 98, 50, 0};
  for (auto i = 0U; i < expected.size(); i++)
    EXPECT_DOUBLE_EQ(predicted[0][i], expected[i]);
}

TEST_F(FuzzySystemTestNoThreshold, smartPredict) {
  FuzzySystem fs = FS_NO_THRESHOLD;

  auto predicted = fs.smartPredict(DF);
  auto ref = fs.predict(DFIN);
  EXPECT_EQ(predicted, ref);

  // scramble df
  DataFrame df(DFIN);
  df.fillCol(1, DFIN[0]);
  df.fillCol(0, DFIN[1]);
  vector<string> colnames = df.colnames();
  swap(colnames[0], colnames[1]);
  df.colnames(colnames);

  predicted = fs.smartPredict(df);

  EXPECT_EQ(predicted, ref);
}

TEST(FuzzySystem, setDefaultRulesConditions) {
  int nb_output_vars = 3;
  int nb_output_seta = 2;
  FuzzySystem fs(2, 3, nb_output_vars, nb_output_seta);

  vector<int> defs = { 2, 1, 5};
  fs.setDefaultRulesConditions(defs);
  vector<int> expected = { 0, 1, 0};
  EXPECT_EQ(fs.getDefaultRulesOutputSets(), expected);
}

TEST_F(FuzzySystemTestNoThreshold, load) {
  FuzzySystem fs = FS_NO_THRESHOLD;
  
  fs.setDefaultRulesConditions({1});
  auto desc = fs.describe();

  auto fs2 = FuzzySystem::load(desc);

  auto desc2 = fs2.describe();
  EXPECT_EQ(desc, desc2);

  auto fs3 = FuzzySystem::load(fs2.describe());
  EXPECT_EQ(fs3.describe(), desc);
}