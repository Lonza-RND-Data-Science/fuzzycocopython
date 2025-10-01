#ifndef FUZZYSYSTEM_H
#define FUZZYSYSTEM_H
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>


#include "fuzzy_set.h"
#include "fuzzy_variables_db.h"
#include "fuzzy_rule.h"
#include "fuzzy_default_rule.h"
#include "dataframe.h"
#include "matrix.h"

namespace fuzzy_coco {

using namespace std;

// small utility
template<typename T> void reset(vector<T>& v, T value) { 
    int nb = v.size();
    v.clear();
    v.resize(nb, value);
}
// instantiate template function
template void reset(vector<bool>&, bool);
template void reset(vector<double>&, double);

struct FuzzySystemEvaluationState {
    FuzzySystemEvaluationState(int nb_in_vars, int nb_out_vars, int nb_output_sets) : 
        output_sets_results(nb_out_vars, nb_output_sets),
        input_value_missing(nb_in_vars),
        input_values(nb_in_vars),
        output_vars_max_fire_levels(nb_out_vars)
    {}
    
    void reset() {
        output_sets_results.reset();
        fuzzy_coco::reset(input_value_missing, false);
        fuzzy_coco::reset(input_values, 0.0);
        // ::reset(input_used, false);
        fuzzy_coco::reset(output_vars_max_fire_levels, 0.0);

        defuzz_values.clear();
    }

    // r[i][j] is the value for input var i, and its output set j
    Matrix<double> output_sets_results;
    // may not be needed (could be encoded in input_values[])
    vector<bool> input_value_missing;
    // stores an input value (from the sample data) for each input var
    vector<double> input_values;

    // =========== output vars related =================

    // output vars max fire levels: for each output var, record its max fire level
    vector<double> output_vars_max_fire_levels;
    vector<double> defuzz_values;
};

class FuzzySystem 
{
public:
    FuzzySystem(int nb_input_vars, int nb_input_sets, int nb_output_vars, int nb_output_sets);
    FuzzySystem(const vector<string>& input_var_names, const vector<string>& output_var_names,
        int nb_input_sets, int nb_output_sets);
    FuzzySystem(const FuzzyVariablesDB& db);
    FuzzySystem(const FuzzySystem& fs) = default;

    ~FuzzySystem() {}

    // ========= MAIN ACCESSORS ====================
    bool ok() const {
        return getNbRules() > 0 && 
            getDefaultRulesOutputSets().size() == (size_t)getDB().getNbOutputVars();
    }

    const FuzzyVariablesDB& getDB() const { return _vars_db; }
    FuzzyVariablesDB& getDB() { return _vars_db; }

    int getNbRules() const { return _rules.size(); }
    // const FuzzyRule& getRule(int idx) const { return _rules[idx]; }
    const vector<int>& getDefaultRulesOutputSets() const { return _default_rules_out_sets; }

    // N.B: need to build them from getDefaultRulesOutputSets()
    vector<FuzzyDefaultRule> fetchDefaultRules() const;

    const vector<FuzzyRule>& getRules() const { return _rules; }
    const FuzzyRule& getRule(int i) const { return _rules[i]; }
    FuzzyRule& getRule(int i) { return _rules[i]; }

    // vector<ConditionIndexes> getInputRulesConditions() const { return _input_rules_idx; }
    // vector<ConditionIndexes> getOutputRulesConditions() const { return _output_rules_idx; }
    int fetchInputVariablesUsage(vector<bool>& used) const;
    vector<int> getUsedInputVariables() const;
    vector<int> getUsedOutputVariables() const;

    // used for the fitness calculation
    int computeTotalInputVarsUsedInRules() const;

    // ================== main setters ====================
    void resetRules(int reserve = 0);
    void addRule(FuzzyRule&& rule) { _rules.push_back(rule); }
    void addRule(const FuzzyRule& rule) { _rules.push_back(rule); }
    void setRules(const vector<FuzzyRule>& rules);

    void setDefaultRules(const vector<FuzzyDefaultRule>& rules);
    void setDefaultRule(const FuzzyDefaultRule& rule);

    void setDefaultRulesConditions(const vector<int>& out_sets);
    void setVariablesSetPositions(const Matrix<double>& insets_pos_mat, const Matrix<double>& outsets_pos_mat);
    void setVariablesSetPositions(const NamedList& pos_desc) { getDB().setPositions(pos_desc); }
    void setVariablesSetPositions(const string& content) { getDB().setPositions(content); }
    // ================== main methods ====================

    DataFrame predict(const DataFrame& input);
    DataFrame smartPredict(const DataFrame& input);
    void predictSample(int sample_idx, const DataFrame& dfin, vector<double>& defuzzed);

    // ==== internal accessors ====
    FuzzySystemEvaluationState& getState() { return _state; }

    // describe the current set of rules and positions
    NamedList describe() const;
    static FuzzySystem load(const NamedList& desc);
    static FuzzySystem load(const string& content) {
      return load(NamedList::parse(content)); 
    }

    friend ostream& operator<<(ostream& out, const FuzzySystem& fs);

    bool operator==(const FuzzySystem& fs) const { 
      return describe() == fs.describe();
    }


    static void printDescription(ostream& out, const NamedList& desc);


public:
    // ========== computations related to predictSample()
    // self-explanatory
    void computeRulesFireLevels(int sample_idx, const DataFrame& df, vector<double>& fire_levels) const;
    // for testing
    vector<double> computeRulesFireLevels(const vector<double>& input_values) const;

    // 
    // compute the implications for the consequent conditionsrules from the rules fire/truth levels
    // N.B: the fire/truth levels are summed. You probably need to reset the results Matrix<double> before
    void computeRulesImplications(const vector<double>& rule_fire_levels, Matrix<double>& results) const;
    // compute and add the implications for the consequent conditionsrules from the rules fire/truth levels
    // N.B: the fire/truth levels are added to the current values of thje output sets
    void addDefaultRulesImplications(const vector<int>& default_rules_set_idx,  const vector<double>& outvars_max_fire_levels, Matrix<double>& results) const;

 
    void computeOutputVarsMaxFireLevels(const vector<double>& rules_fire_levels, vector<double>& outvars_max_fire_levels) const;
    void defuzzify(const Matrix<double>& results, vector<double>& defuzz_values) const;
   
private:
    FuzzyVariablesDB _vars_db;
    
    // variables to be set after the construction
    vector<FuzzyRule> _rules;
    vector<int> _default_rules_out_sets; // store the default rules output sets for each output var

    // variable state
    FuzzySystemEvaluationState _state;
};

}
#endif // FUZZYSYSTEM_H
