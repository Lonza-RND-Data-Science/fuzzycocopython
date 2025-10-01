#include "tests.h"
#include "fuzzy_default_rule.h"

using namespace fuzzy_coco;

string DB_EXAMPLE = R"(
    {
      "Tourists": {
        "Low": 0,
        "High": 100
      },
      "SunShine": {
        "Cloudy": 30,
        "PartSunny": 50,
        "Sunny": 100
      },
      "Temperature": {
        "Cold": 17,
        "Warm": 20,
        "Hot": 29
      },
      "out2": {
        "piccolo": 1,
        "grande": 2
      },
    }
  )";


TEST(FuzzyDefaultRule, ctor_filter) {
  FuzzyVariablesDB db(2, 3, 2, 2);

  FuzzyDefaultRule rule1(db, 0, 1);
 
  EXPECT_EQ(rule1.getVariableIndex(), 0);
  EXPECT_EQ(rule1.getSetIndex(), 1);

  rule1.setVariableIndex(1);
  EXPECT_EQ(rule1.getVariableIndex(), 1);

  rule1.setSetIndex(0);
  EXPECT_EQ(rule1.getSetIndex(), 0);
}


TEST(FuzzyDefaultRule, describe_load_print) {
  FuzzyVariablesDB db({"Temperature", "SunShine"}, 3, {"Tourists", "out2"}, 2); 
  db.setPositions(DB_EXAMPLE);

  FuzzyDefaultRule rule(db, 0, 1);
  auto desc = rule.describe();
  FuzzyDefaultRule::printDescription(cerr, desc);

  auto rule2 = FuzzyDefaultRule::load(desc, db);
  EXPECT_EQ(rule2, rule);

  cerr << rule2 << desc;
}

TEST(FuzzyDefaultRule, convert_default_rules_to_set_idx) {
  FuzzyVariablesDB db({"Temperature", "SunShine"}, 3, {"Tourists", "out2"}, 2); 
  db.setPositions(DB_EXAMPLE);

  FuzzyDefaultRule rule1(db, 0, 1);
  vector<FuzzyDefaultRule> rules = {rule1};

  {
    auto set_idx = FuzzyDefaultRule::convert_default_rules_to_set_idx(rules);
    EXPECT_EQ(set_idx, vector<int>({1, MISSING_DATA_INT}));
  }

  FuzzyDefaultRule rule2(db, 0, 0);
  rules.push_back(rule2);
  { // overwrite
    auto set_idx = FuzzyDefaultRule::convert_default_rules_to_set_idx(rules);
    EXPECT_EQ(set_idx, vector<int>({0, MISSING_DATA_INT}));
  }

  FuzzyDefaultRule rule3(db, 1, 1);
  rules.push_back(rule3);
  { // overwrite + exhaustive
    auto set_idx = FuzzyDefaultRule::convert_default_rules_to_set_idx(rules);
    EXPECT_EQ(set_idx, vector<int>({0, 1}));
  }

}