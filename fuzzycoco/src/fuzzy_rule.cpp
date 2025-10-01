
#include <iostream>
#include <sstream>
#include <algorithm>
#include "fuzzy_rule.h"
#include "fuzzy_operator.h"
#include "types.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace logging;

ConditionIndexes FuzzyRule::filterConditionIndexes(int nb_vars, int nb_sets, const ConditionIndexes& cis)
{
  ConditionIndexes goods;
  goods.reserve(cis.size());
  vector<bool> used(nb_vars, false);
  for (auto pair : cis) {
      const int var_idx = pair.var_idx;
      if (var_idx >= 0 && var_idx < nb_vars && !used[var_idx] &&
          pair.set_idx >= 0 && pair.set_idx < nb_sets) {
          used[var_idx] = true;
          goods.push_back(pair);
      }
  }

  return goods;   
}

ConditionIndexes FuzzyRule::filterConditionIndexesWhenFixedVars(int nb_sets, const ConditionIndexes& cis)
{
    // trick: we just copy and fix the var_idx, then reuse filterConditionIndexes()
    ConditionIndexes cis2(cis);
    const int nb_pairs = cis.size();
    // in this case of fixed vars, the var_idx is just the index of the pair
    for (int i = 0; i < nb_pairs; i++)
        cis2[i].var_idx = i;
    
    return filterConditionIndexes(1, nb_sets, cis2);
}

// // FIXME: Output values don't support -1 due to this implementation.
// #define DONT_CARE_EVAL_RULE -1.0

FuzzyRule::FuzzyRule(const FuzzyVariablesDB& db) : _db(db) {
    setConditions({{0,0}}, {{0,0}});
}

FuzzyRule::FuzzyRule(const FuzzyVariablesDB& db, const ConditionIndexes& input_pairs, const ConditionIndexes& output_pairs, bool filter)
: FuzzyRule(db) 
{
  if (filter) {
    setInputConditions(filterConditionIndexes(getDB().getNbInputVars(), getDB().getNbInputSets(), input_pairs));
    setOutputConditions(filterConditionIndexes(getDB().getNbOutputVars(), getDB().getNbOutputSets(), output_pairs));
  } else {
    setConditions(input_pairs, output_pairs);
  }
}

void FuzzyRule::setInputConditions(const ConditionIndexes& input_conds) {
  // assert(!input_conds.empty());
  _input_cond = input_conds;
}

void FuzzyRule::setOutputConditions(const ConditionIndexes& output_conds) {
  // assert(!output_conds.empty());
  _output_cond = output_conds;
}

void FuzzyRule::setConditions(const ConditionIndexes& input_conds, const ConditionIndexes& output_conds)
{
  setInputConditions(input_conds);
  setOutputConditions(output_conds);
}

ConditionIndexes FuzzyRule::convert_to_conditions(const NamedList& lst, const vector<FuzzyVariable>& vars) {
  const int nb = vars.size();
  auto var_to_idx = FuzzyVariablesDB::build_var_names_to_index_mapper(vars);
  auto set_names_to_idx = FuzzyVariablesDB::build_vars_set_names_to_index_mapper(vars);

  ConditionIndexes conds;
  conds.reserve(nb);

  auto lookup = [&](auto& mapper, const string& key) { 
    auto search = mapper.find(key);
    if (search == mapper.end()) 
      throw out_of_range("in FuzzyRule::, unknown key: " + key);
    return search->second;
  };

  for (size_t i = 0; i < lst.size(); i++) {
    int var_idx = lookup(var_to_idx, lst[i].name());
    string set_name = lst[i].is_list() ? lst[i][0].name() : lst[i].get_string();
    int set_idx = lookup(set_names_to_idx[var_idx], set_name);
    conds.push_back({var_idx, set_idx});
  }
  return conds;
}

void FuzzyRule::setConditions(const NamedList& lst)
{
  auto in_cond = convert_to_conditions(lst.get_list("antecedents"), getDB().getInputVariables());
  auto out_cond = convert_to_conditions(lst.get_list("consequents"), getDB().getOutputVariables());

  setConditions(in_cond, out_cond);
}

NamedList FuzzyRule::describe(bool set_pos) const
{
  auto describe_conditions = [set_pos](const ConditionIndexes& conds, const FuzzyVariables& vars) {
    NamedList desc;
    for (size_t i = 0; i < conds.size(); i++)  {
      const auto& index = conds[i];
      const auto& var = vars[index.var_idx];
  
      if (set_pos) {
        NamedList set;
        NamedList setpos(var.getSet(index.set_idx).getName(), var.getSet(index.set_idx).getPosition());
        set.add(var.getSet(index.set_idx).getName(), setpos);
        desc.add(var.getName(), set);
      } else {
        desc.add(var.getName(), var.getSet(index.set_idx).getName());
      }
    }
    return desc;
  };

  NamedList desc;
  desc.add("antecedents", describe_conditions(getInputConditionIndexes(), getDB().getInputVariables()));
  desc.add("consequents", describe_conditions(getOutputConditionIndexes(), getDB().getOutputVariables()));

  return desc;
}

FuzzyRule FuzzyRule::load(const NamedList& desc, const FuzzyVariablesDB& db) 
{
  FuzzyRule rule(db);
  rule.setConditions(desc);
  return rule;
}

NamedList FuzzyRule::describeRules(const vector <FuzzyRule>& rules, bool set_pos)
{
  NamedList desc;
  const int nb_rules = rules.size();
  for (int i = 0; i < nb_rules; i++) {
    desc.add("rule" + std::to_string(i + 1), rules[i].describe(set_pos));
  }
  return desc;
}//KCOV IGNORE

vector<FuzzyRule> FuzzyRule::loadRules(const NamedList& desc, const FuzzyVariablesDB& db)
{
  const int nb_rules = desc.size();
  vector<FuzzyRule> rules;
  rules.reserve(nb_rules);
  for (int i = 0; i < nb_rules; i++) 
    rules.emplace_back(FuzzyRule::load(desc[i], db));
  return rules;
}//KCOV IGNORE

void FuzzyRule::printDescription(ostream& out, const NamedList& desc)
{
  out << "IF ";
  auto &antecedents = desc.get_list("antecedents");
  int nb = antecedents.size();
  for (int i = 0; i < nb; i++)
  {
    auto &antecedent = antecedents[i];
    out << quoted(antecedent.name());
    out << " IS ";
    out << antecedent[0].name();
    if (i != nb - 1)
      out << " AND ";
  }
  out << " THEN ";
  auto &consequents = desc["consequents"];
  nb = consequents.size();
  for (int i = 0; i < nb; i++)
  {
    auto &consequent = consequents[i];
    out << quoted(consequent.name());
    out << " SHOULD BE ";
    out << consequent[0].name();
    if (i != nb - 1)
      out << " AND ";
  }
}

namespace fuzzy_coco {
  ostream& operator<<(ostream& out, const FuzzyRule& rule) {
    FuzzyRule::printDescription(out, rule.describe());
    return out;
  }
}

// static
double FuzzyRule::evaluateInputConditionFireLevel(const FuzzyVariablesDB& db, const ConditionIndex& ci, double value)
{
    return is_na(value) ? MISSING_DATA_DOUBLE 
        : db.getInputVariable(ci.var_idx).fuzzify(ci.set_idx, value);     
}

double FuzzyRule::evaluateInputConditionFireLevel(int idx, double value) const { 
    return evaluateInputConditionFireLevel(getDB(), getInputConditionIndex(idx), value);
}
// static
double FuzzyRule::evaluateFireLevel(const FuzzyVariablesDB& db, const ConditionIndexes& cis, const vector<double>& input_vars_values)
{
    const int nb_input = cis.size();
    assert(nb_input > 0);
    assert(input_vars_values.size() == (size_t)db.getNbInputVars());

    vector<double> evals(nb_input, 0);
    for (int i = 0; i < nb_input; i++) {
        const auto& ci = cis[i];
        evals[i] = evaluateInputConditionFireLevel(db, ci, input_vars_values[ci.var_idx]);
    }

    return combineFireLevels(evals);
}

double FuzzyRule::evaluateFireLevel(const vector<double>& input_vars_values) const {
    return evaluateFireLevel(getDB(), getInputConditionIndexes(), input_vars_values);
}

// evaluate the firing level of a rule where the input values are stored in a row of a dataframe
// the input variables are assumed to be in the same order as the df columns
double FuzzyRule::evaluateFireLevel(const DataFrame& df, const int row) const {
    return evaluateFireLevel(getDB(), getInputConditionIndexes(), df, row);
}

// static 
double FuzzyRule::evaluateFireLevel(const FuzzyVariablesDB& db, const ConditionIndexes& cis,const DataFrame& df, const int row)  {
    const int nb_input = cis.size();
    assert(nb_input > 0);
    assert(df.nbcols() == db.getNbInputVars());
    assert(row >= 0 && row < df.nbrows());

    vector<double> evals(nb_input, 0);
    for (int i = 0; i < nb_input; i++) {
        const auto& ci = cis[i];
        evals[i] = evaluateInputConditionFireLevel(db, ci, df.at(row, ci.var_idx));
    }
    
    return combineFireLevels(evals);
}

double FuzzyRule::combineFireLevels(const vector<double>& fire_levels) {
    const int nb_input = fire_levels.size();
    assert(nb_input > 0);

    double eval = fire_levels[0];
    if (nb_input > 1) {
        //TODO: The operator should be provided as a param
        FuzzyOperatorAND op;
        for (int i = 1; i < nb_input; i++) {
            eval = op.operate(eval, fire_levels[i]);
        }
    }
    return eval;
}