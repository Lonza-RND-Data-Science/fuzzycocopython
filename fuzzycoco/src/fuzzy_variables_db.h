
/**
  * @file   fuzzy_variables_db.h
  * @author Karl Forner <karl.forner@gmail.com>
  * @date   09.2024
  * @brief a Database of Input and Output Variables mean to be shared across Fuzzy Rules
  * @class FuzzyVariablesDB
  * 
*/
#ifndef FUZZY_VARIABLES_DB_H
#define FUZZY_VARIABLES_DB_H

#include <map>
#include "fuzzy_variable.h"
#include "matrix.h"

namespace fuzzy_coco {

// base class for Variables, basically just a name and FuzzySets container
class FuzzyVariablesDB
{
public:
    FuzzyVariablesDB() {}
    FuzzyVariablesDB(int nb_input_vars, int nb_input_sets, int nb_output_vars, int nb_output_sets);
    FuzzyVariablesDB(const vector<string>& input_names, int nb_in_sets, 
      const vector<string>& output_names, int nb_out_sets);
    FuzzyVariablesDB(const FuzzyVariablesDB& db) : _input_vars(db._input_vars), _output_vars(db._output_vars) {}
    FuzzyVariablesDB(FuzzyVariablesDB&& db) : _input_vars(std::move(db._input_vars)), _output_vars(std::move(db._output_vars)) {}
    ~FuzzyVariablesDB() {}

  // meta data accessors
  int getNbInputVars() const { return _input_vars.size(); }
  int getNbInputSets() const { return _input_vars[0].getSetsCount(); }

  int getNbOutputVars() const { return _output_vars.size(); }
  int getNbOutputSets() const { return _output_vars[0].getSetsCount(); }
  
  // accessors: N.B: bounds checking
  FuzzyVariable& getInputVariable(int idx) { return _input_vars.at(idx); }
  const FuzzyVariable& getInputVariable(int idx) const { return _input_vars.at(idx); }

  FuzzyVariable& getOutputVariable(int idx) { return _output_vars.at(idx); }
  const FuzzyVariable& getOutputVariable(int idx) const { return _output_vars.at(idx); }

  const vector<FuzzyVariable>& getInputVariables() const { return _input_vars; }
  const vector<FuzzyVariable>& getOutputVariables() const { return _output_vars; }

  // ========== main setters ===============
  void setPositions(const string& content) { setPositions(NamedList::parse(content)); }
  void setPositions(const NamedList& lst);
  void setPositions(const Matrix<double>& insets_pos_mat, const Matrix<double>& outsets_pos_mat);

  // fetcher
  Matrix<double> fetchInputPositions() const { return fill_matrix_with_vars(_input_vars); }
  Matrix<double> fetchOutputPositions() const  { return fill_matrix_with_vars(_output_vars); }

  // operations
  FuzzyVariablesDB subset(const vector<int>& input_var_idx, const vector<int>& output_var_idx) const;

  static FuzzyVariablesDB load(const string& content);
  static FuzzyVariablesDB load(const NamedList& desc);
  NamedList describe() const;
  static void printDescription(ostream& out, const NamedList& desc);
  friend ostream& operator<<(ostream& out, const FuzzyVariablesDB& db);

public: // utils
  static vector<string> build_default_var_names(int nbvars, const string& base_name);
  static map<string, int> build_var_names_to_index_mapper(const vector<FuzzyVariable>& vars);
  static map<string, int> build_set_names_to_index_mapper(const FuzzyVariable& var);
  static vector< map<string, int> > build_vars_set_names_to_index_mapper(const vector<FuzzyVariable>& vars);

protected:
 template<class T> 
 static Matrix<double> fill_matrix_with_vars(const vector<T>& vars);

private:
  vector<FuzzyVariable> _input_vars;
  vector<FuzzyVariable> _output_vars;
};


template<class T> 
Matrix<double> FuzzyVariablesDB::fill_matrix_with_vars(const vector<T>& vars) {
  const int nb_vars = vars.size();
  const int nb_sets = vars.front().getSetsCount();
  Matrix<double> mat(nb_vars, nb_sets);
  for (int var_idx = 0; var_idx < nb_vars; var_idx++) {
    auto& row = mat[var_idx];
    auto& var = vars[var_idx];
    for (int set_idx = 0; set_idx < nb_sets; set_idx++) {
      row[set_idx] = var.getSet(set_idx).getPosition();
    }
  }
  return mat;
}

}
#endif // FUZZY_VARIABLES_DB_H
