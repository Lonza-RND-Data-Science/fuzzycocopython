#include "tests.h"
#include "fuzzy_coco.h"
#include "file_utils.h"

using namespace fuzzy_coco;
using namespace FileUtils;

string CSV = 
R"(Days;Temperature;Sunshine;Tourists
day1;19;25;55
day2;40;99;95
day3;24;NA;70
day4;5;3;2
)";

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

const string RULES = R"(
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

FuzzyCocoCodec GET_FUZZY_COCO_CODEC() {
  DataFrame DF(CSV, true);
  DataFrame DFIN = DF.subsetColumns(0, DF.nbcols() - 2);
  DataFrame DFOUT = DF.subsetColumns(DF.nbcols() - 1, DF.nbcols() - 1);
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());

  return FuzzyCocoCodec(DFIN, DFOUT, params);
}

FuzzySystem GET_FUZZY_SYSTEM() {
  DataFrame DF(CSV, true);
  DataFrame DFIN = DF.subsetColumns(0, DF.nbcols() - 2);
  DataFrame DFOUT = DF.subsetColumns(DF.nbcols() - 1, DF.nbcols() - 1);
  FuzzyCocoParams params = GET_SAMPLE_PARAMS(DFIN.nbcols());

  auto db = FuzzyVariablesDB::load(NamedList::parse(FUZZY_SYSTEM_112));
  FuzzySystem fs(db);

  return fs;
}

TEST(FuzzyCocoCodec, createDiscretizersForData) {
  DataFrame df(CSV, true);

  auto ds = FuzzyCocoCodec::createDiscretizersForData(df, 2);

  EXPECT_EQ(ds.size(), df.nbcols());
}

TEST(FuzzyCocoCodec, setRulesGenome) {
  FuzzyCocoCodec fcs = GET_FUZZY_COCO_CODEC();
  FuzzySystem fs = GET_FUZZY_SYSTEM();

  EXPECT_EQ(fcs.getNbRuleAntedecents(), 2);

  auto rules = FuzzyRule::loadRules(RULES, fs.getDB());
  // cerr << FuzzyRule::describeRules(rules);
  FuzzyDefaultRule defrule = FuzzyDefaultRule::load(R"("Tourists":"High")", fs.getDB());

  
  Genome geno_rules = fcs.encode(rules, defrule);
  // now decode it
  fcs.setRulesGenome(fs, geno_rules);
  cerr << fs;
  EXPECT_EQ(fs.getRule(0), rules[0]);
  EXPECT_EQ(fs.getRule(1), rules[1]);
  EXPECT_EQ(fs.getRule(2), rules[2]);
}

double relative_error(double x, double ref) { return abs(x - ref) / ref; }
double max_relative_error(const vector<double>& x, const vector<double>& ref) {
  double the_max = 0;
  const int nb = x.size();
  for (int i = 0; i < nb; i++)  {
    the_max = max(the_max, relative_error(x[i], ref[i]));
  }
  return the_max;
}
double max_relative_error(const Matrix<double>& x, const Matrix<double>& ref) {
  double the_max = 0;
  const int nb = x.size();
  for (int i = 0; i < nb; i++)  {
    the_max = max(the_max, max_relative_error(x[i], ref[i]));
  }
  return the_max;
}


TEST(FuzzyCocoCodec, setMFsGenome) {
  FuzzyCocoCodec fcs = GET_FUZZY_COCO_CODEC();
  auto fs = GET_FUZZY_SYSTEM();

  Matrix<double> pos_in = {  {10, 20, 30}, {25, 50, 75} };
  Matrix<double> pos_out = { {5, 50, 100} };

  Genome mfs_genome = fcs.encode(pos_in, pos_out);
  fcs.setMFsGenome(fs, mfs_genome);

  Matrix<double> res_in, res_out;
  fcs.decode(mfs_genome, res_in, res_out);

  // N.B: positions have been discretized
  EXPECT_EQ(fs.getDB().fetchInputPositions(), res_in);
  EXPECT_EQ(fs.getDB().fetchOutputPositions(), res_out);

  // discretized but close
  EXPECT_LT(max_relative_error(res_in, pos_in), 0.01);

  // for ouput vars, only 2 bits --> low precision
  EXPECT_GT(max_relative_error(res_out, pos_out), 0.5);
  EXPECT_LT(max_relative_error(res_out, pos_out), 0.7);
}


TEST(FuzzyCocoCodec, modifyRuleAntecedent) {
  FuzzyCocoCodec fcs = GET_FUZZY_COCO_CODEC();
  FuzzySystem fs = GET_FUZZY_SYSTEM();
  auto rules = FuzzyRule::loadRules(RULES, fs.getDB());
  FuzzyDefaultRule defrule = FuzzyDefaultRule::load(R"("Tourists":"High")", fs.getDB());

  Genome rules_geno = fcs.encode(rules, defrule);
  vector<ConditionIndexes> in_idx, out_idx;
  vector<int> default_rules;
  int nb_in_conditions = rules[0].getNbInputConditions();
  for (size_t i = 0; i < rules.size(); i++) {
    for (int j = 0; j < nb_in_conditions; j++) {
      fcs.decode(rules_geno, in_idx, out_idx, default_rules);

      Genome modified = rules_geno;
      fcs.modifyRuleAntecedent(modified, i, j, i + j);

      fcs.decode(modified, in_idx, out_idx, default_rules);
      int var_idx = in_idx[i][j].var_idx;

      EXPECT_EQ(var_idx, i + j);
    }
  }

}