#include "tests.h"
#include "fuzzy_rule.h"
#include "fuzzy_operator.h"

using namespace fuzzy_coco;

TEST(FuzzyRule, ctor_filter) {
  FuzzyVariablesDB db(2, 3, 2, 2);

  FuzzyRule rule1(db, {{0, 9}, {9, 1}}, {{0, 5}, {0, 0}}, true);
  //                     BAD    BAD       BAD     GOOD
  EXPECT_EQ(rule1.getNbInputConditions(), 0);
  EXPECT_EQ(rule1.getNbOutputConditions(), 1);
  EXPECT_EQ(rule1.getOutputConditionIndex(0), ConditionIndex({0, 0}));

  cerr << rule1 << endl;

  FuzzyRule rule2(db, {{0, 2}, {0, 0}}, {{5, 0}, {5, 0}}, true);
  //                     GOOD     DUP      BAD     BAD
  EXPECT_EQ(rule2.getNbInputConditions(), 1);
  EXPECT_EQ(rule2.getInputConditionIndex(0), ConditionIndex({0, 2})); // took the first one
  EXPECT_EQ(rule2.getNbOutputConditions(), 0);

    cerr << rule2 << endl;

  FuzzyRule rule3(db, {{0, 0}, {1, 1}}, { {0,0}, {0, 1}}, true);
  //                     GOOD     GOOD      GOOD     DUP
  EXPECT_EQ(rule3.getNbInputConditions(), 2);
  EXPECT_EQ(rule3.getNbOutputConditions(), 1);
  EXPECT_EQ(rule3.getOutputConditionIndex(0), ConditionIndex({0, 0})); // took the first one

  cerr << rule3 << endl;

  FuzzyRule rule4(db,  {{0, 0}, {1, 1}},  {{0, 0}, {1, 1}}, true);
  //                     GOOD     GOOD      GOOD     GOOD
  EXPECT_EQ(rule4.getNbInputConditions(), 2);
  EXPECT_EQ(rule4.getNbOutputConditions(), 2);


  cerr << rule4 << endl;

}

TEST(FuzzyOperatorAND, operate) {
    FuzzyOperatorAND op;

    EXPECT_DOUBLE_EQ(op.operate(0, 0), 0);
    EXPECT_DOUBLE_EQ(op.operate(-1, 0), 0);
    EXPECT_DOUBLE_EQ(op.operate(0, -1), 0);
    EXPECT_DOUBLE_EQ(op.operate(-1, -1), -1);

    EXPECT_DOUBLE_EQ(op.operate(0, 0.2), 0);
    EXPECT_DOUBLE_EQ(op.operate(0.2, 0), 0);

    EXPECT_DOUBLE_EQ(op.operate(0.2, 0.5), 0.2);
    EXPECT_DOUBLE_EQ(op.operate(0.5, 0.2), 0.2);

    // missing data
    EXPECT_DOUBLE_EQ(op.operate(MISSING_DATA_DOUBLE, 0), 0);
    EXPECT_DOUBLE_EQ(op.operate(0, MISSING_DATA_DOUBLE), 0);
    EXPECT_DOUBLE_EQ(op.operate(MISSING_DATA_DOUBLE, 0.4), 0.4);
    EXPECT_DOUBLE_EQ(op.operate(0.4, MISSING_DATA_DOUBLE), 0.4);

    EXPECT_DOUBLE_EQ(op.operate(MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE), MISSING_DATA_DOUBLE);
}

// cf coco book pp26 fig 1.12
string FUZZY_SYSTEM_112 = R"(
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


class FuzzyRuleTest112 : public testing::Test {
 protected:
    FuzzyRuleTest112() : DB(FuzzyVariablesDB::load(NamedList::parse(FUZZY_SYSTEM_112))) {}
    // void SetUp() override {
    // }
    FuzzyVariablesDB DB;
};

TEST_F(FuzzyRuleTest112, ctor) {

    // N.b: rule modified since we only have AND
    FuzzyRule rule(DB);

    cerr << rule.describe() << endl;

    // created with dummy rules
    EXPECT_EQ(rule.getNbInputConditions(), 1);
    EXPECT_EQ(rule.getNbOutputConditions(), 1);
    
    EXPECT_EQ(rule.getInputConditionIndex(0).var_idx, 0);
    EXPECT_EQ(rule.getInputConditionIndex(0).set_idx, 0);

    EXPECT_EQ(rule.getOutputConditionIndex(0).var_idx, 0);
    EXPECT_EQ(rule.getOutputConditionIndex(0).set_idx, 0);
}


TEST_F(FuzzyRuleTest112, setConditions_describe_load)
{
  // N.b: rule modified since we only have AND
  FuzzyRule rule1(DB);
  rule1.setConditions({{0, 2}, {1, 2}}, {{0, 2}});

  cerr << rule1.describe();
  cerr << "rule1:" << rule1 << endl;
  FuzzyRule rule1bis = FuzzyRule::load(rule1.describe(), DB);
  EXPECT_EQ(rule1bis, rule1);

  {
    cerr << rule1.describe(false);
    auto rule1bis = FuzzyRule::load(rule1.describe(false), DB);
    EXPECT_EQ(rule1bis, rule1);
  }

  FuzzyRule rule2(DB);
  rule2.setConditions({{0, 1}, {1, 1}}, {{0, 1}});
  cerr << "rule2:" << rule2 << endl;
 
  FuzzyRule rule3(DB);
  rule3.setConditions({{0, 0}, {1, 0}}, {{0, 0}});
  cerr << "rule2:" <<  rule3 << endl;

  // describe
  auto desc = rule3.describe();
  cerr << desc;

  string ref = R"(
{
  "antecedents": {
    "Temperature": { "Cold":17.0 }, 
    "Sunshine": { "Cloudy":30.0 }
  },
  "consequents": { "Tourists": { "Low":0.0 } }
})";
  auto lst = NamedList::parse(ref);
  EXPECT_EQ(desc, lst);

  cerr << "========  rule3bis =====" << "\n";

  FuzzyRule rule3bis(DB);
  cerr << lst;

  rule3bis.setConditions(lst);
  cerr << rule3bis;

  EXPECT_EQ(rule3bis.describe(), lst);

  FuzzyRule rule3ter = FuzzyRule::load(lst, DB);
  EXPECT_EQ(rule3ter.describe(), lst);

  {
    // edge case: bad input variable name
    string ref = R"(
      {
        "antecedents":{
          "TOTO":"Cold",
          "Sunshine":"Cloudy"
        },
        "consequents":{
          "Tourists":"Low"
        }
      }
      )";

    EXPECT_THROW(FuzzyRule::load(ref, DB), out_of_range);
  }

  {
    // edge case: bad output variable name
    string ref = R"(
      {
        "antecedents":{
          "Temperature":"Cold",
          "Sunshine":"Cloudy"
        },
        "consequents":{
          "BAD":"Low"
        }
      }
      )";

    EXPECT_THROW(FuzzyRule::load(ref, DB), out_of_range);
  }

  {
    // edge case: bad set variable name
    string ref = R"(
      {
        "antecedents":{
          "Temperature":"Cold",
          "Sunshine":"BAD"
        },
        "consequents":{
          "Tourists":"Low"
        }
      }
      )";

    EXPECT_THROW(FuzzyRule::load(ref, DB), out_of_range);
  }


  // =============================== loadRules / describeRules ==========
  {
    vector<FuzzyRule> rules = { rule1, rule2, rule3 };
    auto desc = FuzzyRule::describeRules(rules);
    auto rules2 = FuzzyRule::loadRules(desc, DB);
    EXPECT_EQ(FuzzyRule::describeRules(rules2), desc);
  }

}


TEST_F(FuzzyRuleTest112, fuzzify_evaluateFireLevel)
{

  FuzzyRule rule1(DB);
  rule1.setConditions({{0, 2}, {1, 2}}, {{0, 2}});
  FuzzyRule rule2(DB);
  rule2.setConditions({{0, 1}, {1, 1}}, {{0, 1}});
  FuzzyRule rule3(DB);
  rule3.setConditions({{0, 0}, {1, 0}}, {{0, 0}});

  // input vars fuzzy values
  const auto &TEMPERATURE = DB.getInputVariable(0);
  EXPECT_DOUBLE_EQ(TEMPERATURE.fuzzify(0, 19), 1.0 / 3);
  EXPECT_DOUBLE_EQ(TEMPERATURE.fuzzify(1, 19), 2.0 / 3);
  EXPECT_DOUBLE_EQ(TEMPERATURE.fuzzify(2, 19), 0);

  const auto &SUNSHINE = DB.getInputVariable(1);
  EXPECT_DOUBLE_EQ(SUNSHINE.fuzzify(0, 60), 0);
  EXPECT_DOUBLE_EQ(SUNSHINE.fuzzify(1, 60), 0.8);
  EXPECT_DOUBLE_EQ(SUNSHINE.fuzzify(2, 60), 0.2);

  // rules truth levels
  EXPECT_DOUBLE_EQ(rule1.evaluateFireLevel({19, 60}), min(0.0, 0.2));
  EXPECT_DOUBLE_EQ(rule1.evaluateFireLevel({19, 60}), min(0.0, 0.2));

  EXPECT_DOUBLE_EQ(rule2.evaluateInputConditionFireLevel(0, 19), 2.0 / 3);
  EXPECT_DOUBLE_EQ(rule2.evaluateInputConditionFireLevel(1, 60), 0.8);
  EXPECT_DOUBLE_EQ(rule2.evaluateFireLevel({19, 60}), min(2.0 / 3, 0.8));

  EXPECT_DOUBLE_EQ(rule3.evaluateFireLevel({19, 60}), min(1.0 / 3, 0.0));

  // using a dataframe
  DataFrame df;
  df.reset(2, 2);
  df.set(0, 0, 19);
  df.set(0, 1, 60);
  df.set(1, 0, 25);
  df.set(1, 1, 37);

  EXPECT_DOUBLE_EQ(rule2.evaluateFireLevel(df, 0), rule2.evaluateFireLevel({df.at(0, 0), df.at(0, 1)}));
  EXPECT_DOUBLE_EQ(rule2.evaluateFireLevel(df, 1), rule2.evaluateFireLevel({df.at(1, 0), df.at(1, 1)}));
}

TEST_F(FuzzyRuleTest112, missingData) {
    // ========= missing data with only one input condition
    FuzzyRule rule1(DB);
    rule1.setConditions({{0, 0}}, {{0, 0}});
    cerr << rule1 << endl;

    // temp is 10 --> it is Cold!!
    EXPECT_DOUBLE_EQ(rule1.evaluateFireLevel({10, 1}), 1);
    // temp is missing --> do not fire
    EXPECT_TRUE(is_na(rule1.evaluateFireLevel({MISSING_DATA_DOUBLE, 1})));

    // ==== missing data with 2 conditions
    FuzzyRule rule2(DB);
    // if temp is cold and sunshine is cloudy then tourists is low
    rule2.setConditions({{0, 0}, {1, 0}}, {{0, 0}});

    // no missing data: 10 degrees, 15% of sun
    EXPECT_DOUBLE_EQ(rule2.evaluateFireLevel({10, 15}), 1);
    // one missing data
    EXPECT_DOUBLE_EQ(rule2.evaluateFireLevel({MISSING_DATA_DOUBLE, 15}), 1);
    // all missing
    EXPECT_TRUE(is_na(rule2.evaluateFireLevel({MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE})));
}

TEST(ConditionIndex, basic_istringstream) {
    ConditionIndex ci{2, 1};
    EXPECT_NO_THROW(cerr << ci);
}


TEST(FuzzyRule, filterConditionIndexes) {
    // N.B: only the last one is good
    ConditionIndexes cis = {{-1, 0}, {3, 0}, {0, -1}, {0,3}, {0, 0}};
    ConditionIndexes good = FuzzyRule::filterConditionIndexes(3, 3, cis);
    EXPECT_EQ(good.size(), 1);
    EXPECT_EQ(good[0].var_idx, 0);
    EXPECT_EQ(good[0].set_idx, 0);

    auto good2 = FuzzyRule::filterConditionIndexes(3 ,3 , good);
    EXPECT_EQ(good2.size(), 1);

    // repetitions
    cis = {{2, 1}, {0, 2}, {2, 1}, {0, 1}};
    auto filtered = FuzzyRule::filterConditionIndexes(3, 3, cis);
    ConditionIndexes expected = {{2, 1}, {0, 2}};
    EXPECT_EQ(filtered, expected);

    // repetitions and out-of-range values
    cis = {{2, 1}, {0, 2}, {2, 1}, {0, 1}, {-1, 0}, {5, 0}, {0, -1}, {0, 4}, {4, 4}};
    filtered = FuzzyRule::filterConditionIndexes(3, 3, cis);
    EXPECT_EQ(filtered, expected);
}

TEST(FuzzyRule, filterConditionIndexesWhenFixedVars) {
    // N.B: only the two last ones are good
    ConditionIndexes cis = {{-1, 0}, {3, 2}, {0, -1}, {0,3}, {0, 0}, {-1, 1}};
    ConditionIndexes good = FuzzyRule::filterConditionIndexesWhenFixedVars(2, cis);
    cerr << cis << endl;
    cerr << good << endl;
    EXPECT_EQ(good.size(), 1);
    EXPECT_EQ(good[0].var_idx, 0);
    EXPECT_EQ(good[0].set_idx, 0);

    auto good2 = FuzzyRule::filterConditionIndexesWhenFixedVars(2, good);
    EXPECT_EQ(good2.size(), 1);
}
