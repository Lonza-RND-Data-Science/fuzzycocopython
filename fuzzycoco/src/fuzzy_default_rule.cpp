#include "fuzzy_default_rule.h"
using namespace fuzzy_coco;

FuzzyDefaultRule FuzzyDefaultRule::load(const NamedList& desc, const FuzzyVariablesDB& db)
{
  auto var_to_idx = FuzzyVariablesDB::build_var_names_to_index_mapper(db.getOutputVariables());
  auto set_names_to_idx = FuzzyVariablesDB::build_vars_set_names_to_index_mapper(db.getOutputVariables());

  int var_idx = var_to_idx.at(desc.name());
  int set_idx = set_names_to_idx.at(var_idx).at(desc.get_string());

  return FuzzyDefaultRule(db, var_idx, set_idx);
}


NamedList FuzzyDefaultRule::describe() const
{
  const auto& var = getDB().getOutputVariable(getVariableIndex());
  NamedList set_desc(var.getName(), var.getSet(getSetIndex()).getName());
  return set_desc;
}

NamedList FuzzyDefaultRule::describeDefaultRules(const vector<FuzzyDefaultRule>& rules)
{
  NamedList desc;
  const int nb = rules.size();
  for (int i = 0; i < nb; i++) {
    auto rule_desc = rules[i].describe();
    desc.add(rule_desc.name(), rule_desc);
  }
  return desc;
}//KCOV IGNORE

vector<FuzzyDefaultRule> FuzzyDefaultRule::loadDefaultRules(const NamedList& desc, const FuzzyVariablesDB& db)
{
  vector<FuzzyDefaultRule> default_rules;
  default_rules.reserve(desc.size());
  for (size_t i = 0; i < desc.size(); i++) 
    default_rules.emplace_back(FuzzyDefaultRule::load(desc[i], db));
  return default_rules;
}//KCOV IGNORE

void FuzzyDefaultRule::printDescription(ostream& out, const NamedList& desc)
{
  out << quoted(desc.name()) << " SHOULD BE " << desc.get_string() << endl;
}

namespace fuzzy_coco {
  ostream& operator<<(ostream& out, const FuzzyDefaultRule& rule) {
    FuzzyDefaultRule::printDescription(out, rule.describe());
    return out;
  }
}


// convert FuzzyDefaultRules to a vector wich stores v[var_idx]=set_idx
vector<int> FuzzyDefaultRule::convert_default_rules_to_set_idx(const vector<FuzzyDefaultRule>& rules)
{
  assert(!rules.empty());
  const auto& db = rules.front().getDB();

  vector<int> set_idx_by_var(db.getNbOutputVars(), MISSING_DATA_INT);
  for (const auto& rule: rules) {
    if (rule.getVariableIndex() >= 0 && rule.getVariableIndex() < db.getNbOutputVars()) {
      if (rule.getSetIndex() >= 0 && rule.getSetIndex() < db.getNbOutputSets())
        set_idx_by_var[rule.getVariableIndex()] = rule.getSetIndex();
    }
  }
  return set_idx_by_var;
}
