#include "fuzzy_coco_codec.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace logging;

void FuzzyCocoCodec::init(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params)
{
  const int nb_input_vars = dfin.nbcols();
  const int nb_output_vars = dfout.nbcols();

    // params
  const int nb_input_sets = params.input_vars_params.nb_sets;
  const int nb_output_sets = params.output_vars_params.nb_sets;
  const int nb_rules = params.global_params.nb_rules;

  PosParams pos_input(nb_input_vars, nb_input_sets, params.input_vars_params.nb_bits_pos);
  PosParams pos_output(nb_output_vars, nb_output_sets, params.output_vars_params.nb_bits_pos);
  auto disc_in = createDiscretizersForData(dfin, pos_input.nb_bits);
  auto disc_out = createDiscretizersForData(dfout, pos_output.nb_bits);

  _vars_codec_ptr = make_unique<DiscretizedFuzzySystemSetPositionsCodec>(pos_input, pos_output, disc_in, disc_out);

  // pre-sizing
  _rules_in.reserve(nb_rules);
  _rules_out.reserve(nb_rules);
}

FuzzyCocoCodec::FuzzyCocoCodec(
    const DataFrame& dfin, const DataFrame& dfout, 
    const FuzzyCocoParams& params)
  : 
    _nb_rules_antecedents(min(dfin.nbcols(), params.global_params.nb_max_var_per_rule)),
    _rules_codec(params.global_params.nb_rules, 
      {_nb_rules_antecedents, params.input_vars_params.nb_bits_vars, params.input_vars_params.nb_bits_sets},
      {dfout.nbcols(), params.output_vars_params.nb_bits_vars, params.output_vars_params.nb_bits_sets})
{ 
  init(dfin, dfout, params); 
}


Genome FuzzyCocoCodec::encode(const vector<FuzzyRule>& rules, const vector<FuzzyDefaultRule>& default_rules)
{
  auto set_idx_by_var = FuzzyDefaultRule::convert_default_rules_to_set_idx(default_rules);
  return encode(rules, set_idx_by_var);
}

Genome FuzzyCocoCodec::encode(const vector<FuzzyRule>& rules, const FuzzyDefaultRule& default_rule)
{
  vector<FuzzyDefaultRule> defs = { default_rule };
  return encode(rules, defs);
}


// only for test purposes - no need to be optimized
Genome FuzzyCocoCodec::encode(const vector<FuzzyRule>& rules, const vector<int>& default_rules)
{
  vector<ConditionIndexes> in_cis, out_cis;
  in_cis.reserve(rules.size());
  out_cis.reserve(rules.size());

  for (size_t i = 0; i < rules.size(); i++) {
    in_cis.push_back(rules[i].getInputConditionIndexes());
    out_cis.push_back(rules[i].getOutputConditionIndexes());
  }

  Genome rules_geno;
  encode(in_cis, out_cis, default_rules, rules_geno);
  return rules_geno;
}

void FuzzyCocoCodec::encode(const vector<ConditionIndexes>& in_cis, const vector<ConditionIndexes>& out_cis, 
  const vector<int>& default_rules, Genome& rules_geno)
{
  rules_geno.resize(getRulesCodec().size());
  getRulesCodec().encode(in_cis, out_cis, default_rules, rules_geno);
}

Genome FuzzyCocoCodec::encode(const Matrix<double>& pos_in, const Matrix<double>& pos_out)
{
  Genome geno(getMFsCodec().size());
  getMFsCodec().encode(pos_in, pos_out, geno);
  return geno;
}//KCOV IGNORE

void FuzzyCocoCodec::decode(const Genome& mfs_genome, Matrix<double>& pos_in, Matrix<double>& pos_out)
{
  auto it = mfs_genome.cbegin();
  getMFsCodec().decode(it, pos_in, pos_out);
}

// void FuzzyCocoCodec::decode(const Genome& rules_genome, const FuzzyVariablesDB& db, vector<FuzzyRule>& rules, vector<FuzzyDefaultRule>& default_rules) 
// {
//   decode(rules_genome, _rules_in, _rules_out, _default_rules);
//   rules.clear();
//   const int nb = _rules_in.size();
//   rules.reserve(nb);
//   for (int i = 0; i < nb; i++) {
//      FuzzyRule rule(db, _rules_in[i], _rules_out[i], true);
//      rules.push_back(rule);
//   }
//   default_rules.clear();
//   default_rules.reserve(_default_rules.size());
//   for (size_t i = 0; i < _default_rules.size(); i++) {
//     FuzzyDefaultRule rule(db, i, _default_rules[i]);
//     default_rules.push_back(rule);
//  }
// }

void FuzzyCocoCodec::decode(
    const Genome& rules_genome, 
    vector<ConditionIndexes>& rules_in,
    vector<ConditionIndexes>& rules_out,
    vector<int>& default_rules)
{
  auto it1 = rules_genome.cbegin();
  getRulesCodec().decode(it1, rules_in, rules_out, default_rules);
}

void FuzzyCocoCodec::setRulesGenome(FuzzySystem& fs, const Genome& rules_genome) {
  decode(rules_genome, _rules_in, _rules_out, _default_rules);

  const int MAX_NB_RULES = _rules_in.size();
  fs.resetRules(MAX_NB_RULES);

  for (int i = 0; i < MAX_NB_RULES; i++) {
    FuzzyRule rule(fs.getDB(), _rules_in[i],  _rules_out[i], true); // NB: filter==true
    if (rule.getNbInputConditions() > 0 && rule.getNbOutputConditions() > 0)
      fs.addRule(std::move(rule));
  }

  fs.setDefaultRulesConditions(_default_rules);
}

void FuzzyCocoCodec::setMFsGenome(FuzzySystem& fs, const Genome& mfs_genome) {
  decode(mfs_genome, _pos_in, _pos_out);
  fs.setVariablesSetPositions(_pos_in, _pos_out);
}

vector<Discretizer> FuzzyCocoCodec::createDiscretizersForData(const DataFrame& df, int nb_bits) {
  const int nb = df.nbcols();
  vector<Discretizer> res;
  res.reserve(nb);
  for (int i = 0; i < nb; i++) {
    res.push_back({nb_bits, df[i]});
  }

  return res;
}//KCOV IGNORE

void FuzzyCocoCodec::modifyRuleAntecedent(Genome& rules_genome, int rule_idx, int ant_idx, int var_idx)
{
  assert(rule_idx >= 0 && rule_idx < getNbRules());
  assert(ant_idx >= 0 && ant_idx < getNbRuleAntedecents());
  
  decode(rules_genome, _rules_in, _rules_out, _default_rules);
  _rules_in[rule_idx][ant_idx].var_idx = var_idx;
  encode(_rules_in, _rules_out, _default_rules, rules_genome);
}