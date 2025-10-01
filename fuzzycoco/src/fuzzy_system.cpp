#include "fuzzy_system.h"
#include <unordered_map>
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace logging;

FuzzySystem FuzzySystem::load(const NamedList& desc) {

  FuzzySystem fs(FuzzyVariablesDB::load(desc.get_list("variables")));

  auto rules = FuzzyRule::loadRules(desc.get_list("rules"), fs.getDB());
  fs.resetRules(rules.size());
  for (const auto& rule: rules)
    fs.addRule(rule);
  
  // default rules
  auto default_rules = FuzzyDefaultRule::loadDefaultRules(desc.get_list("default_rules"), fs.getDB());
  auto set_idxs = FuzzyDefaultRule::convert_default_rules_to_set_idx(default_rules);
  fs.setDefaultRulesConditions(set_idxs);

  return fs;
}

FuzzySystem::FuzzySystem(const FuzzyVariablesDB& db) 
    : _vars_db(db), _state(db.getNbInputVars(), db.getNbOutputVars(), db.getNbOutputSets()) 
{
    _default_rules_out_sets.resize(_vars_db.getNbOutputVars(), 0);
}

FuzzySystem::FuzzySystem(int nb_input_vars, int nb_input_sets, int nb_output_vars, int nb_output_sets) 
    : FuzzySystem(FuzzyVariablesDB(nb_input_vars, nb_input_sets, nb_output_vars, nb_output_sets)) 
{}

FuzzySystem::FuzzySystem(const vector<string>& input_var_names, const vector<string>& output_var_names, 
  int nb_input_sets, int nb_output_sets)
    : FuzzySystem({input_var_names, nb_input_sets, output_var_names, nb_output_sets})
{}


vector<FuzzyDefaultRule> FuzzySystem::fetchDefaultRules() const
{
  vector<FuzzyDefaultRule> rules;
  const auto& set_idxs = getDefaultRulesOutputSets();
  const int nb = set_idxs.size();
  rules.reserve(nb);
  for (int var_idx = 0; var_idx < nb; var_idx++) {
    rules.emplace_back(getDB(), var_idx, set_idxs[var_idx]);
  }
  return rules;
}//KCOV IGNORE


// N.B: if the set idx of a default rule is out of range --> set it to 0
void FuzzySystem::setDefaultRulesConditions(const vector<int>& out_sets) {
  assert(out_sets.size() == (size_t)getDB().getNbOutputVars());
  // need to check the output set indexes, and set to 0 if wrong
  _default_rules_out_sets = out_sets;
  const int nb = getDB().getNbOutputSets();
  for (auto& i : _default_rules_out_sets)
    if (i >= nb) i = 0; // N.B: change in-place
}

void FuzzySystem::resetRules(int reserve) {
  _rules.clear();
  if (reserve > 0) {
    _rules.reserve(reserve);
  }
}

void FuzzySystem::setDefaultRules(const vector<FuzzyDefaultRule>& rules)
{
  auto set_idxs = FuzzyDefaultRule::convert_default_rules_to_set_idx(rules);
  setDefaultRulesConditions(set_idxs);
}

void FuzzySystem::setDefaultRule(const FuzzyDefaultRule& rule)
{
  setDefaultRules({rule});
}

void FuzzySystem::setRules(const vector<FuzzyRule>& rules)
{
  resetRules(rules.size());
  for (const auto& rule: rules)
    addRule(rule);
}

// void FuzzySystem::addRuleConditions(const ConditionIndexes& input_cond, const ConditionIndexes& output_cond)
// {
//   // _input_rules_idx.push_back(input_cond);
//   // _output_rules_idx.push_back(output_cond);
// }
// set the Fuzzy system Rules
// these rules are usually created stochastically so they can be incompatible or inconsistent
// void FuzzySystem::setRulesConditions(const vector<ConditionIndexes>& input_cond_lst, const vector<ConditionIndexes>& output_cond_lst)
// {
//   // the max possible new rules
//   const int MAX_NB_RULES = input_cond_lst.size();
//   assert(output_cond_lst.size() == MAX_NB_RULES);

//   resetRules(MAX_NB_RULES);

//   // filter out the abnormal rule conditions (out-of-range, repetitions...)
//   // and rules which have empty either input or output condictions
//   for (int i = 0; i < MAX_NB_RULES; i++) {
//     auto cis_in = FuzzyRule::filterConditionIndexes(getDB().getNbInputVars(), getDB().getNbInputSets(), input_cond_lst[i]);
//     if (!cis_in.empty()) {
//       auto cis_out = FuzzyRule::filterConditionIndexes(getDB().getNbOutputVars(), getDB().getNbOutputSets(), output_cond_lst[i]);
//       if (!cis_out.empty()) {
//           addRuleConditions(cis_in, cis_out);
//       }
//     }
//   }
// }

int FuzzySystem::computeTotalInputVarsUsedInRules() const {
  int total = 0;
  for (int i = 0; i < getNbRules(); i++) {
    total += getRule(i).getNbInputConditions();
  }
  return total;
}

int FuzzySystem::fetchInputVariablesUsage(vector<bool>& used) const
{
  const int nb = getDB().getNbInputVars();
  used.clear();
  used.resize(nb, false);
  int nb_used = 0;
  for (int i = 0; i < getNbRules(); i++) {
    for (const auto& ci: getRule(i).getInputConditionIndexes()) {
      used[ci.var_idx] = true;
      nb_used++;
    }
  }

  return nb_used;
}

vector<int> FuzzySystem::getUsedInputVariables() const
{
  const int nb = getDB().getNbInputVars();
  vector<bool> used(nb, false);
  int nb_used = fetchInputVariablesUsage(used);

  vector<int> res;
  res.reserve(nb_used);
  for (int i = 0; i < nb; i++)
      if (used[i])
          res.push_back(i);
  return res;
}

vector<int> FuzzySystem::getUsedOutputVariables() const
{
  const int nb = getDB().getNbOutputVars();
  vector<bool> used(nb, false);

  for (int i = 0; i < getNbRules(); i++) 
    for (const auto& co: getRule(i).getOutputConditionIndexes()) 
      used[co.var_idx] = true;
  
  vector<int> res;
  res.reserve(nb);
  for (int i = 0; i < nb; i++)
    if (used[i])
      res.push_back(i);

  return res;
}

void FuzzySystem::setVariablesSetPositions(const Matrix<double>& insets_pos_mat, const Matrix<double>& outsets_pos_mat)
{
    getDB().setPositions(insets_pos_mat, outsets_pos_mat);
}

NamedList FuzzySystem::describe() const {
  NamedList desc;
  // NamedList params;
  // params.add("nb_rules", getNbRules());
  // params.add("nb_input_sets", getDB().getNbInputSets());
  // params.add("nb_output_sets", getDB().getNbOutputSets());
  // desc.add("parameters", params);

  desc.add("variables", getDB().describe());
  desc.add("rules", FuzzyRule::describeRules(_rules));

  // NamedList default_rules;
  // const auto& defs = getDefaultRulesOutputSets();
  // const int nb = getDB().getNbOutputVars();
  // for (int var_idx = 0; var_idx < nb; var_idx++) {
  //     FuzzyDefaultRule rule(getDB(), var_idx, defs[var_idx]);
  //     auto rule_desc = rule.describe();
  //     default_rules.add(rule_desc.name(), rule_desc);
  // }
  desc.add("default_rules", FuzzyDefaultRule::describeDefaultRules(fetchDefaultRules()));
    
  return desc;
}

void FuzzySystem::printDescription(ostream& out, const NamedList& desc)
{
  out << "FuzzySystem:" << endl;

  NamedList vars = desc["variables"];
  out << "#" <<  "Fuzzy Variables:" << endl;
  NamedList ivars = vars["input"];
  out << "## input variables:" << endl;
  for (const auto& var : ivars) {
    FuzzyVariable::printDescription(out, var);
    out << endl;
  }
  NamedList ovars = vars.get_list("output");
  out << "## output variables:" << endl;
  for (const auto& var : ovars) {
    FuzzyVariable::printDescription(out, var);
    out << endl;
  }

  auto& rules = desc.get_list("rules");
  out << "## rules:" << endl;
  for (auto& rule : rules) {
    FuzzyRule::printDescription(out, rule);
    out << endl;
  }

  auto& defrules = desc.get_list("default_rules");
  out << "## default rules:" << endl;
  out << "ELSE ";
  for (auto& defrule : defrules) {
      out << quoted(defrule.name()) 
      << " SHOULD BE " << defrule.get_string() << endl;
  }
}

namespace fuzzy_coco {
  ostream& operator<<(ostream& out, const FuzzySystem& fs) {
    auto desc = fs.describe();
    FuzzySystem::printDescription(out, fs.describe());
    return out;
  }
}


void FuzzySystem::computeRulesImplications(const vector<double>& rule_fire_levels, Matrix<double>& results) const {
  const int nb_rules = getNbRules();
  const int nb_vars = getDB().getNbOutputVars();
  const int nb_sets = getDB().getNbOutputSets();

  assert(results.nbrows() == nb_vars && results.nbcols() == nb_sets);
  assert(rule_fire_levels.size() == (size_t)nb_rules);

  results.reset(); // important since we add the levels

  for (int rule_idx = 0; rule_idx < nb_rules; rule_idx++) {
    const double fire_level = rule_fire_levels[rule_idx];
    // if the firelevel is missing the rule does not fire, thus it is ignored
    if (is_na(fire_level)) continue;

    for (const auto& ci : getRule(rule_idx).getOutputConditionIndexes()) 
      results[ci.var_idx][ci.set_idx] += fire_level;
          
  }
}

void FuzzySystem::addDefaultRulesImplications(const vector<int>& default_rules_set_idx,  const vector<double>& outvars_max_fire_levels, Matrix<double>& results) const {
    const size_t nb_out_vars = getDB().getNbOutputVars();

    assert(default_rules_set_idx.size() == nb_out_vars);
    assert(outvars_max_fire_levels.size() == nb_out_vars);
    assert((size_t)results.nbrows() == nb_out_vars && results.nbcols() == getDB().getNbOutputSets());

    for (size_t var_idx = 0; var_idx < nb_out_vars; var_idx++) {
        const int set_idx = default_rules_set_idx[var_idx];
        const double max_fire_level = outvars_max_fire_levels[var_idx];
        double current_level = results[var_idx][set_idx];
        assert(!is_na(current_level));
        if (!is_na(max_fire_level)) {
            // N.B: only use the value if it is not missing.
            results[var_idx][set_idx] += 1 - max_fire_level;
        }
    }
}


// N.B: put results in fire_levels[]
void FuzzySystem::computeRulesFireLevels(int sample_idx, const DataFrame& df, vector<double>& fire_levels) const {
  assert(df.nbcols() == getDB().getNbInputVars());
  assert(sample_idx >= 0 && sample_idx < df.nbrows());

  const int nb_rules = getNbRules();
  fire_levels.clear();
  fire_levels.reserve(nb_rules);

  for (int rule_idx = 0; rule_idx < nb_rules; rule_idx++) 
    fire_levels.push_back( getRule(rule_idx).evaluateFireLevel(df, sample_idx) );
  
}
// N.B: mostly for testing and convenience purposes
vector<double> FuzzySystem::computeRulesFireLevels(const vector<double>& input_values) const {
  vector<double> fire_levels;
  fire_levels.reserve(getNbRules());
  for (int rule_idx = 0; rule_idx < getNbRules(); rule_idx++) {
    fire_levels.push_back( getRule(rule_idx).evaluateFireLevel(input_values) );
  }
  return fire_levels;
}//KCOV IGNORE

// the fire levels for each rule are applied and MAXed to the corresponding consequent output vars
// N.B: those max fire levels are used for the default rules
void FuzzySystem::computeOutputVarsMaxFireLevels(const vector<double>& rules_fire_levels, vector<double>& outvars_max_fire_levels) const {
  const size_t nb_rules = getNbRules();
  const size_t nb_out_vars = getDB().getNbOutputVars();
  assert(nb_rules > 0);
  assert(rules_fire_levels.size() == nb_rules);
  
  outvars_max_fire_levels.resize(0);
  outvars_max_fire_levels.resize(nb_out_vars, MISSING_DATA_DOUBLE);

  for (size_t rule_idx = 0; rule_idx < nb_rules; rule_idx++) {
    const double fire_level = rules_fire_levels[rule_idx];
    const auto& rule = getRule(rule_idx);
    for (const auto& co : rule.getOutputConditionIndexes()) {
      outvars_max_fire_levels[co.var_idx] = max(outvars_max_fire_levels[co.var_idx], fire_level);
    }
  }
}
// defuzzify the output variables
void FuzzySystem::defuzzify(const Matrix<double>& results, vector<double>& defuzz_values) const {
    const size_t nb_out_vars = getDB().getNbOutputVars();
    const size_t nb_sets = getDB().getNbOutputSets();
    assert(results.size() == nb_out_vars && results[0].size() == (size_t)getDB().getNbOutputSets());
    
    vector<double> set_evals(nb_sets, 0);
    defuzz_values.clear();

    for (size_t var_idx = 0; var_idx < nb_out_vars; var_idx++) {
        const auto& results_for_var = results[var_idx];
        // gather values for this output variable sets
        set_evals.clear();
        for (size_t set_idx = 0; set_idx < nb_sets; set_idx++) {
            set_evals.push_back(results_for_var[set_idx]);
        }
        defuzz_values.push_back(getDB().getOutputVariable(var_idx).defuzz(set_evals));
    }
}


// double FuzzySystem::threshold_defuzzed_value(int out_var_idx, double defuzz) const {
//     const int nb_out = getDB().getNbOutputVars();
//     assert(out_var_idx >= 0 && out_var_idx < nb_out);
//     assert(areDefuzzThresholdsEnabled());
//     return apply_threshold(_defuzz_thresholds[out_var_idx], defuzz);
// }

// double FuzzySystem::apply_threshold(double threshold, double value) {
//     if (value >= threshold)
//         value = 1;
//     else if (value  >= 0)
//         value = 0;
//     else value = MISSING_DATA_DOUBLE;
//     return value;
// }

// evaluate the fuzzy system on data: and output the defuzzed values in predicted
DataFrame FuzzySystem::predict(const DataFrame& input)
{
    const int nb_samples = input.nbrows();
    const int nb_out_vars = getDB().getNbOutputVars();
    DataFrame res(nb_samples, nb_out_vars);
    vector<double> defuzzed(res.nbcols());
    
    vector<string> output_names;
    output_names.reserve(nb_out_vars);
    for (int i = 0; i < nb_out_vars; i++)
        output_names.push_back(getDB().getOutputVariable(i).getName());
    res.colnames(output_names);

    for (int i = 0; i < nb_samples; i++) {
        predictSample(i, input, defuzzed);
        res.fillRow(i, defuzzed);
    }

    return res;
}

// same as predict, but is smart on the input data frame:  will use the column names
// to match the values on the corresponding variables
DataFrame FuzzySystem::smartPredict(const DataFrame& df)
{
    const int nb = getDB().getNbInputVars();
    vector<string> dfin_colnames(nb);
    for (int i = 0; i < nb; i++) {
        dfin_colnames[i] = getDB().getInputVariable(i).getName();
    }

    return predict(df.subsetColumns(dfin_colnames));
}

// evaluate the fuzzy system on data for a sample: and output the defuzzed value by output var in defuzzed
void FuzzySystem::predictSample(int sample_idx, const DataFrame& dfin, vector<double>& defuzzed)
{
    assert(sample_idx >= 0);

    // rules processing
    vector<double> fire_levels; // TODO: put in state for reuse ?
    computeRulesFireLevels(sample_idx, dfin, fire_levels);
    computeRulesImplications(fire_levels, getState().output_sets_results);

    computeOutputVarsMaxFireLevels(fire_levels, getState().output_vars_max_fire_levels);
    addDefaultRulesImplications(getDefaultRulesOutputSets(), getState().output_vars_max_fire_levels, getState().output_sets_results);
    // karl TODO: Magali to check if that threshold is relevant, and if it should be a parameter or a constant
    // computeRulesFireLevelsStatistics(fire_levels, 0.2, getState().rules_fired, getState().rules_winners);

    defuzzify(getState().output_sets_results, defuzzed);

}

