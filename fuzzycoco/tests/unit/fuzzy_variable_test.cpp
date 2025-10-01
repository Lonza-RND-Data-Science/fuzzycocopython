#include "tests.h"
#include <cstdio>
#include "file_utils.h"

#include "fuzzy_set.h"
#include "fuzzy_variable.h"
#include "fuzzy_variables_db.h"

using namespace fuzzy_coco;
using namespace FileUtils;

TEST(FuzzyVariable, build_default_set_names) {
    auto names = FuzzyVariable::build_default_set_names(3, "base");
    vector<string> expected{"base.1", "base.2", "base.3"};
    EXPECT_EQ(names, expected);
}

TEST(FuzzyVariable, ctors) {
    FuzzyVariable var("var1", 3);
    EXPECT_EQ(var.getName(), "var1");
    EXPECT_EQ(var.getSetsCount(), 3);
    auto names = FuzzyVariable::build_default_set_names(3, "var1");
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(var.getSets()[i].getName(), names[i]);
        EXPECT_TRUE(is_na(var.getSets()[i].getPosition()));
    }
    cerr << var << endl;

    // =============
    FuzzyVariable var2("var2", {"one", "two", "three"});
    names = {"one", "two", "three"};
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(var2.getSets()[i].getName(), names[i]);
        EXPECT_TRUE(is_na(var2.getSets()[i].getPosition()));
    }
    cerr << var2 << endl;

    /// ========== with sets
    FuzzyVariable var3("var3", var2.getSets());
    EXPECT_EQ(var3.getSets(), var2.getSets());

    cerr << var3 << endl;

    // copy cstor
    FuzzyVariable var3copy(var3);
    EXPECT_EQ(var3copy.getName(), var3.getName());
}

TEST(FuzzyVariable, load_describe) {
  FuzzyVariable var("var", 3);
  var.setSets(vector<FuzzySet>({{"Cold", 17}, {"Warm", 20}, {"Hot", 29}}));

  // ============= describe ==============
  auto desc = var.describe();

  EXPECT_EQ(desc.name(), "var");
  EXPECT_EQ(desc.size(), 3);

  auto var2 = FuzzyVariable::load(desc);

  EXPECT_EQ(var2, var);
}

TEST(FuzzyVariable, setSetsPositions) {
  FuzzyVariable var("var", {"one", "two", "three"});
  var.setSetsPositions(R"(
    { "Warm": 20,
      "Cold": 17,
      "Hot": 29
    }  )");
  
  EXPECT_EQ(var.getSet(0), FuzzySet("Warm", 20));
  EXPECT_EQ(var.getSet(1), FuzzySet("Cold", 17));
  EXPECT_EQ(var.getSet(2), FuzzySet("Hot", 29));
}

TEST(FuzzyOutputVariable, output) {
    FuzzyVariable var("out1", 2);
    cerr << var << endl;
    EXPECT_EQ(var.getName(), "out1");
    EXPECT_EQ(var.getSetsCount(), 2);
    EXPECT_THROW(var.getSetIndexByName("set1"), out_of_range);

    EXPECT_EQ(var.getSet(0), FuzzySet("out1.1"));
    EXPECT_EQ(var.getSetIndexByName("out1.1"), 0);

    EXPECT_EQ(var.getSets()[1], FuzzySet{"out1.2"});
    EXPECT_EQ(var.getSetIndexByName("out1.2"), 1);
    cerr << var << endl;

    // if no positions set --> MISSING
    EXPECT_TRUE(is_na(var.defuzz( { 0, 10 })));

    var.getSet(0).setPosition(0.2);
    var.getSet(1).setPosition(0.8);

    EXPECT_EQ(var.defuzz({ 0, 10 }), 0.8);

    EXPECT_EQ(var.defuzz({ 20, 10 }), (0.8*10 + 0.2*20) / 30);

    // ====  missingness ====
    // a position is missing --> the corresponding set is ignore
    var.getSet(0).setPosition(2);
    var.getSet(1).setPosition(MISSING_DATA_DOUBLE);
    EXPECT_DOUBLE_EQ(var.defuzz( { 1, 10 }), 2.0*1 / 1.0);

    // same if the eval is missing
    var.getSet(1).setPosition(10);
    EXPECT_DOUBLE_EQ(var.defuzz( { 1, MISSING_DATA_DOUBLE }), 2.0*1 / 1.0);

    // if 2 sets are ignored --> missing
    var.getSet(0).setPosition(MISSING_DATA_DOUBLE);
    EXPECT_TRUE(is_na(var.defuzz( { 1, MISSING_DATA_DOUBLE })));

    // evals sum to 0 --> 0
    var.getSet(0).setPosition(2);
    var.getSet(1).setPosition(10);
    EXPECT_DOUBLE_EQ(var.defuzz( { 0, 0 }), 0);

    var.getSet(0).setPosition(2);
    var.getSet(1).setPosition(10);
    EXPECT_DOUBLE_EQ(var.defuzz( { -1, 1 }), 0);
}

TEST(FuzzyInputVariable, input) {
    FuzzyVariable var("in", 2);
    cerr << var << endl;
    EXPECT_EQ(var.getName(), "in");
    EXPECT_EQ(var.getSetsCount(), 2);
    EXPECT_THROW(var.getSetIndexByName("set1"), out_of_range);
    EXPECT_EQ(var.getSetIndexByName("in.1"), 0);

    // without set positions --> MISSING DATA
    EXPECT_TRUE(is_na(var.fuzzify(0, 0.1)));
    EXPECT_TRUE(is_na(var.fuzzify(1, 0)));

    var.getSet(0).setPosition(0.2);
    var.getSet(1).setPosition(0.8);

    EXPECT_DOUBLE_EQ(var.fuzzify(0, 0.1), 1);
    EXPECT_DOUBLE_EQ(var.fuzzify(1, 0), 0);
    cerr << var << endl;

    EXPECT_DOUBLE_EQ(var.fuzzify(0, 0.2), 1);
    EXPECT_DOUBLE_EQ(var.fuzzify(1, 0), 0);

    EXPECT_DOUBLE_EQ(var.fuzzify(0, 0.8), 0);
    EXPECT_DOUBLE_EQ(var.fuzzify(1, 0.8), 1);

    EXPECT_DOUBLE_EQ(var.fuzzify(0, 0.5), 0.5);
    EXPECT_DOUBLE_EQ(var.fuzzify(1, 0.5), 0.5);

    EXPECT_DOUBLE_EQ(var.fuzzify(0, 1), 0);
    EXPECT_DOUBLE_EQ(var.fuzzify(1, 1), 1);
}

// the fuzzycoco book example on pp21 fig 1.6.
TEST(FuzzyInputVariable, book1_6) {
    // FuzzyInputVariable temp("Temperature", vector<FuzzySet>{{"Cold", 17}, {"Warm1", 20}, {"Warm2", 26}, {"Hot", 29}});
    auto temp = FuzzyVariable::load(
      R"( "Temperature": { "Cold":17, "Warm1":20, "Warm2":26, "Hot": 29} )");
    
    EXPECT_DOUBLE_EQ(temp.fuzzify(0, 19), 1.0/3);
    EXPECT_DOUBLE_EQ(temp.fuzzify(1, 19), 2.0/3);
    EXPECT_DOUBLE_EQ(temp.fuzzify(2, 19), 0);
}


TEST(FuzzyVariablesDB, build_default_var_names) {
  auto names = FuzzyVariablesDB::build_default_var_names(3, "in");
  EXPECT_EQ(names, vector<string>({"in_1", "in_2", "in_3"}));
}

TEST(FuzzyVariablesDB, ctor1) {
  FuzzyVariablesDB db(3, 2, 2, 4);
  EXPECT_EQ(db.getNbInputVars(), 3);
  EXPECT_EQ(db.getNbInputSets(), 2);
  EXPECT_EQ(db.getNbOutputVars(), 2);
  EXPECT_EQ(db.getNbOutputSets(), 4);
}

TEST(FuzzyVariablesDB, basic) {

    FuzzyVariablesDB db({"in1", "in2"}, 3, {"out1"}, 2);
    
    EXPECT_EQ(db.getNbInputVars(), 2);
    EXPECT_EQ(db.getNbInputSets(), 3);
    EXPECT_EQ(db.getNbOutputVars(), 1);
    EXPECT_EQ(db.getNbOutputSets(), 2);
    cerr << db;

    EXPECT_EQ(db.getInputVariable(0), FuzzyVariable("in1", 3));
    EXPECT_EQ(db.getInputVariable(1), FuzzyVariable("in2", 3));

    EXPECT_EQ(db.getOutputVariable(0), FuzzyVariable("out1", 2));

    // set positions
    db.getInputVariable(1).getSet(1).setPosition(10);
    EXPECT_EQ(db.getInputVariable(1).getSet(1).getPosition(), 10);

    Matrix<double> matin(2, 3);
    for (int var_idx = 0; var_idx < 2; var_idx++)
        for (int set_idx = 0; set_idx < 3; set_idx++)
            matin[var_idx][set_idx] = var_idx*10 + set_idx;
    Matrix<double> matout(1, 2);
    for (int var_idx = 0; var_idx < 1; var_idx++)
        for (int set_idx = 0; set_idx < 2; set_idx++)
            matout[var_idx][set_idx] = var_idx*10 + set_idx + 100;   
    db.setPositions(matin, matout);

    cerr << db;

    EXPECT_EQ(matin, db.fetchInputPositions());
    EXPECT_EQ(matout, db.fetchOutputPositions());

    // check that setPositions() sort the values
    Matrix<double> matin2 = matin;

    swap(matin2[0][0], matin2[0][2]);
    swap(matin2[1][0], matin2[1][1]);
    Matrix<double> matout2 = matout;
    swap(matout2[0][0], matout2[0][1]);

    db.setPositions(matin2, matout2);

    EXPECT_EQ(matin, db.fetchInputPositions());
    EXPECT_EQ(matout, db.fetchOutputPositions());
}

TEST(FuzzyVariablesDB, describe_and_load) {
  FuzzyVariablesDB db({"Temperature", "SunShine"}, 3, {"Tourists", "out2"}, 2); 

  db.setPositions(R"(
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
  )");

  // describe
  auto desc = db.describe();
  cerr << desc;
  auto db2 = FuzzyVariablesDB::load(desc);

  EXPECT_EQ(db.describe(), db2.describe());
}

TEST(FuzzyVariablesDB, subset) {
    FuzzyVariablesDB db({"in1", "in2", "in3"}, 3, {"out1", "out2", "out3"}, 2);

    Matrix<double> matin(3, 3);
    for (int var_idx = 0; var_idx < 3; var_idx++)
        for (int set_idx = 0; set_idx < 3; set_idx++)
            matin[var_idx][set_idx] = var_idx*10 + set_idx;
    Matrix<double> matout(3, 2);
    for (int var_idx = 0; var_idx < 3; var_idx++)
        for (int set_idx = 0; set_idx < 2; set_idx++)
            matout[var_idx][set_idx] = var_idx*10 + set_idx + 100;   
    db.setPositions(matin, matout);

    auto db2 = db.subset({2, 0}, {1});
    EXPECT_EQ(db2.getNbInputVars(), 2);
    EXPECT_EQ(db2.getNbOutputVars(), 1);
    EXPECT_EQ(db2.getNbInputSets(), db.getNbInputSets());
    EXPECT_EQ(db2.getNbOutputSets(), db.getNbOutputSets());

    EXPECT_EQ(db2.fetchInputPositions()[0], db.fetchInputPositions()[2]);
    EXPECT_EQ(db2.fetchInputPositions()[1], db.fetchInputPositions()[0]);
    EXPECT_EQ(db2.fetchOutputPositions()[0], db.fetchOutputPositions()[1]);

    cerr << db2;
}

TEST(FuzzyVariablesDB, setPositions) {
  FuzzyVariablesDB db({"Temperature", "SunShine"}, 3, {"Tourists", "out2"}, 2); 

  EXPECT_THROW(db.setPositions(R"( { "Toto": { "Low": 0, "High": 100  }  }  )"), runtime_error);

  db.setPositions(R"(
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
  )");
  
  cerr << db.describe();
  auto& temp = db.getInputVariable(0);
  EXPECT_EQ(temp.getSet(0).getName(), "Cold");
  EXPECT_EQ(temp.getSet(0).getPosition(), 17);
  EXPECT_EQ(temp.getSet(1).getName(), "Warm");
  EXPECT_EQ(temp.getSet(1).getPosition(), 20);
  EXPECT_EQ(temp.getSet(2).getName(), "Hot");
  EXPECT_EQ(temp.getSet(2).getPosition(), 29);

  auto& sun = db.getInputVariable(1);
  EXPECT_EQ(sun.getSet(0).getName(), "Cloudy");
  EXPECT_EQ(sun.getSet(0).getPosition(), 30);
  EXPECT_EQ(sun.getSet(1).getName(), "PartSunny");
  EXPECT_EQ(sun.getSet(1).getPosition(), 50);
  EXPECT_EQ(sun.getSet(2).getName(), "Sunny");
  EXPECT_EQ(sun.getSet(2).getPosition(), 100);

  auto& tourists = db.getOutputVariable(0);
  EXPECT_EQ(tourists.getSet(0).getName(), "Low");
  EXPECT_EQ(tourists.getSet(0).getPosition(), 0);
  EXPECT_EQ(tourists.getSet(1).getName(), "High");
  EXPECT_EQ(tourists.getSet(1).getPosition(), 100);

  auto& out2 = db.getOutputVariable(1);
  EXPECT_EQ(out2.getSet(0).getName(), "piccolo");
  EXPECT_EQ(out2.getSet(0).getPosition(), 1);
  EXPECT_EQ(out2.getSet(1).getName(), "grande");
  EXPECT_EQ(out2.getSet(1).getPosition(), 2);
}
