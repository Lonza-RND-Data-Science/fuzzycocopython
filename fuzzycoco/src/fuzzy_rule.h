/**
  * @section DESCRIPTION
  *
  * This class represent a fuzzy rule. The rule is composed of [input/output variables - sets] pairs.
  * The rule is created by reading a rule genome and fetching the corresponding variables in an array
  * of variables passed in parameter. The variable array contains all the existing variables of the
  * system. The variables used in the rule are selected according to the rule genome and stored in
  * internal arrays.
  *
  * Once the rule has been evaluated, the result is stored in the corresponding set of the corresponding
  * variable for later defuzzification.
  */

#ifndef FUZZYRULE_H
#define FUZZYRULE_H

#include <vector>
#include <cassert>

#include "fuzzy_variables_db.h"
#include "dataframe.h"
#include "named_list.h"

namespace fuzzy_coco {

// a data structure associating a Fuzzy/Variable index with a FuzzySet index to form a pair: (var_idx, set_idx)
// Then a FuzzyRule is a list of input and output FuzzyRuleItems FuzzyRule= {input={FuzzyRuleItem}_i, output={FuzzyRuleItem}_j}
// the input FuzzyRuleItems are the antecedents and the output the consequents

struct ConditionIndex {
  int var_idx = -1;
  int set_idx = -1;

  ConditionIndex(int vi = -1, int si = -1) { var_idx = vi; set_idx = si;}

  ConditionIndex(const ConditionIndex& p) {
    var_idx = p.var_idx;
    set_idx = p.set_idx;
  }
};
inline bool operator==(const ConditionIndex& a, const ConditionIndex& b) { 
  return a.var_idx == b.var_idx && a.set_idx == b.set_idx;
}


using ConditionIndexes = vector<ConditionIndex>;
using FuzzyVariables = vector<FuzzyVariable>;
// using FuzzyInputVariables = vector<const FuzzyInputVariable*>;
// using FuzzyOutputVariables = vector<const FuzzyOutputVariable*>;

// a Fuzzy Rule is defined by input an output Fuzzy Conditions rule=(input_conditions=(IC_i), output_conditions=(OC_j))
// each condiction reference a fixed set of Input and Output Variables

// note to myself: the FuzzyVariable instances are extern (not owned by the FuzzyRule instance) and shared
// across the rules (actually owned e.g. by FuzzySystem)
// the main method is evaluate()
class FuzzyRule
{
public:
    // build the fuzzy rule from the pairs. Note that the idx in the pairs
    // N.B: constructed with dummy conditions: (0,0) --> (0,0)
    FuzzyRule(const FuzzyVariablesDB& db);
    FuzzyRule(const FuzzyVariablesDB& db, const ConditionIndexes& input_pairs, const ConditionIndexes& output_pairs, bool filter = false);
    ~FuzzyRule() {}

    // very important setter
    // N.B: may need to be filtered !!
    void setInputConditions(const ConditionIndexes& input_pairs);
    void setOutputConditions(const ConditionIndexes& output_pairs);
    void setConditions(const ConditionIndexes& input_pairs, const ConditionIndexes& output_pairs);
    void setConditions(const NamedList& lst);

    // ConditionIndexes convert_to_conditions(const NamedList& lst, const vector<FuzzyVariable>& vars)

    // accessors
    const FuzzyVariablesDB& getDB() const { return _db; }

    // Returns the number of input variables/sets pairs.
    int getNbInputConditions() const { return _input_cond.size(); }
    int getNbOutputConditions() const {  return _output_cond.size(); }
    const ConditionIndex& getInputConditionIndex(int pos) const { return _input_cond[pos]; }
    const ConditionIndex& getOutputConditionIndex(int pos) const { return _output_cond[pos]; }

    const ConditionIndexes& getInputConditionIndexes() const { return _input_cond; }
    const ConditionIndexes& getOutputConditionIndexes() const { return _output_cond; }

    // ============== methods that actually compute ======================
    
    // combine the fire levels of each input condition into the final rule fire level
    // double combineFireLevels(const vector<double>& fire_levels) const;
    // void evaluate(const vector<double>& input_values, vector<double>& fire_levels);
    // evaluate all input conditions

    //N.B: input_vars_values contains all values for InputVars in the correct order
    // N.B: use is_na() to detect missing data --> return MISSING_DATA_DOUBLE
    double evaluateFireLevel(const vector<double>& input_vars_values) const;


    double evaluateFireLevel(const DataFrame& df, const int row) const;

    // evaluate one input condition. N.B: if the value is missing, the fire level is 0
    // N.B: use is_na() to detect missing data
    double evaluateInputConditionFireLevel(int pos, double value) const;
    // Returns the fire level for the corresponding output variable.
    // double getFireLevel(int varNum) const { return fireLevel[varNum]; }
    // const vector<int>& getUsedOutVars() const { return _usedOutVars; }

    // if set_pos == true, add the set positions in the description
    NamedList describe(bool set_pos = true) const;

    static FuzzyRule load(const NamedList& desc, const FuzzyVariablesDB& db);
    static FuzzyRule load(const string& content, const FuzzyVariablesDB& db) {
      return FuzzyRule::load(NamedList::parse(content), db);
    }

    static void printDescription(ostream& out, const NamedList& desc);
    friend ostream& operator<<(ostream& out, const FuzzyRule& rule);
  
    inline bool operator==(const FuzzyRule& rule) const { 
      return describe() == rule.describe();
    }

public:
  static NamedList describeRules(const vector<FuzzyRule>& rules, bool set_pos = true);
  static vector<FuzzyRule> loadRules(const NamedList& desc, const FuzzyVariablesDB& db);
  static vector<FuzzyRule> loadRules(const string& content, const FuzzyVariablesDB& db) {
    return loadRules(NamedList::parse(content), db);
  }

  static double evaluateInputConditionFireLevel(const FuzzyVariablesDB& db, const ConditionIndex& ci, double value);
  static double evaluateFireLevel(const FuzzyVariablesDB& db, const ConditionIndexes& cis, const vector<double>& input_vars_values);
  static double evaluateFireLevel(const FuzzyVariablesDB& db, const ConditionIndexes& cis, const DataFrame& df, const int row);
  
  // combine the fire levels of each input condition into the final rule fire level
  static double combineFireLevels(const vector<double>& fire_levels);

  // could/should go elsewhere ?
  // filter out the "wrong" pairs, which indices are out of range (the pairs usually come from the genetic algorithm)
  static ConditionIndexes filterConditionIndexes(int nb_vars, int nb_sets, const ConditionIndexes& cis);
  // fixed vars mean that all vars are used, in the same order in pairs as in vars (i.e. we ignore the var index) 
  static ConditionIndexes filterConditionIndexesWhenFixedVars(int nb_sets, const ConditionIndexes& cis);

  static ConditionIndexes convert_to_conditions(const NamedList& lst, const vector<FuzzyVariable>& vars);

private:
  const FuzzyVariablesDB& _db;
  ConditionIndexes _input_cond;
  ConditionIndexes _output_cond;
};

inline ostream& operator<<(ostream& out, const ConditionIndex& ci) {
  out << "[" << ci.var_idx << "," << ci.set_idx << "]";
  return out;
}

}
#endif // FUZZYRULE_H
