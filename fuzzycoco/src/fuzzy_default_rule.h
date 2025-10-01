/**
  * @section DESCRIPTION
  *
  * This class represent a fuzzy default rule, a special case of FuzzyRule, only applied to one output variable, and to one
  * of its (output) set.
  *
  */

#ifndef FUZZY_DEFAULT_RULE_H
#define FUZZY_DEFAULT_RULE_H

#include <cassert>
#include "fuzzy_variables_db.h"
#include "named_list.h"

namespace fuzzy_coco {

class FuzzyDefaultRule
{
public:
    FuzzyDefaultRule(const FuzzyVariablesDB& db, int var_idx = 0, int set_idx = 0)
      : _db(db), _var_idx(var_idx), _set_idx(set_idx) {}


    void setVariableIndex(int var_idx) { _var_idx = var_idx; }
    void setSetIndex(int set_idx) { _set_idx = set_idx; }

    // accessors
    const FuzzyVariablesDB& getDB() const { return _db; }

    int getVariableIndex() const { return _var_idx; }
    int getSetIndex() const { return _set_idx; }


    NamedList describe() const;
    static FuzzyDefaultRule load(const NamedList& desc, const FuzzyVariablesDB& db);
    static FuzzyDefaultRule load(const string& content, const FuzzyVariablesDB& db) {
      return load(NamedList::parse(content), db);
    }

    static NamedList describeDefaultRules(const vector <FuzzyDefaultRule>& rules);
    static vector<FuzzyDefaultRule> loadDefaultRules(const NamedList& desc, const FuzzyVariablesDB& db);

    static void printDescription(ostream& out, const NamedList& desc);
    friend ostream& operator<<(ostream& out, const FuzzyDefaultRule& rule);
  
    inline bool operator==(const FuzzyDefaultRule& rule) const { 
      return getVariableIndex() == rule.getVariableIndex() && 
        getSetIndex() == rule.getSetIndex();
    }

  static vector<int> convert_default_rules_to_set_idx(const vector<FuzzyDefaultRule>& rules);

private:
  const FuzzyVariablesDB& _db;
  int _var_idx;
  int _set_idx;
};

}
#endif // FUZZY_DEFAULT_RULE_H
