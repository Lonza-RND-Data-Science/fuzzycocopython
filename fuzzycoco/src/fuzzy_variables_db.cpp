#include "fuzzy_variables_db.h"
#include <algorithm>
#include <map>

using namespace fuzzy_coco;

FuzzyVariablesDB::FuzzyVariablesDB(int nb_input_vars, int nb_input_sets, int nb_output_vars, int nb_output_sets) 
  : FuzzyVariablesDB(
      build_default_var_names(nb_input_vars, "in"), nb_input_sets, 
      build_default_var_names(nb_output_vars, "out"), nb_output_sets)
{}


FuzzyVariablesDB::FuzzyVariablesDB(
  const vector<string>& input_names, int nb_in_sets, 
  const vector<string>& output_names, int nb_out_sets)
{
  const int nb_in_vars = input_names.size();
  const int nb_out_vars = output_names.size();
  
  assert(nb_in_vars > 0);
  assert(nb_in_sets > 0);
  assert(nb_out_vars > 0);
  assert(nb_out_sets > 0);

  _input_vars.reserve(nb_in_vars);
  for (auto name : input_names)
    _input_vars.push_back({name, nb_in_sets});

  _output_vars.reserve(nb_out_vars);
  for (auto name : output_names)
    _output_vars.push_back({name, nb_out_sets});
}



void FuzzyVariablesDB::setPositions(const NamedList& lst) 
{
  // map to convert from var names to idx
  map<string, int> in_var_to_idx;
  for (int i = 0; i < getNbInputVars(); i++) 
    in_var_to_idx[getInputVariable(i).getName()] = i;
  map<string, int> out_var_to_idx;
  for (int i = 0; i < getNbOutputVars(); i++) 
    out_var_to_idx[getOutputVariable(i).getName()] = i;

  // convert to Matrix positions
  Matrix<double> pos_in(getNbInputVars(), getNbInputSets());
  Matrix<double> pos_out(getNbOutputVars(), getNbOutputSets());
  for (const auto& var_name: lst.names()) {
    const auto& var_lst = lst.get_list(var_name);
    // identify variable
    if (in_var_to_idx.count(var_name) > 0) {
      int idx = in_var_to_idx[var_name];
      getInputVariable(idx).setSetsPositions(var_lst);
    } else if (out_var_to_idx.count(var_name) > 0) {
      int idx = out_var_to_idx[var_name];
      getOutputVariable(idx).setSetsPositions(var_lst);
    } else {
      throw runtime_error("error, variable name unknown: " + var_name);
    }
  }
}



void FuzzyVariablesDB::setPositions(const Matrix<double>& insets_pos_mat, const Matrix<double>& outsets_pos_mat)
{
    assert(insets_pos_mat.nbrows() == getNbInputVars() && insets_pos_mat.nbcols() == getNbInputSets());
    assert(outsets_pos_mat.nbrows() == getNbOutputVars() && outsets_pos_mat.nbcols() == getNbOutputSets());

    auto fill = [](const Matrix<double>& m, auto& vars) {
        const int nb_vars = m.nbrows();
        const int nb_sets = m.nbcols();
        for (int var_idx = 0; var_idx < nb_vars; var_idx++) {
            auto row = m[var_idx];
            sort(row.begin(), row.end());
            auto& var = vars[var_idx];
            for (int set_idx = 0; set_idx < nb_sets; set_idx++) {
                var.getSet(set_idx).setPosition(row[set_idx]);
            }
        }
    };

    fill(insets_pos_mat, _input_vars);
    fill(outsets_pos_mat, _output_vars);
}

NamedList FuzzyVariablesDB::describe() const {
  NamedList vars;

  NamedList input_vars;
  for (const auto& var : _input_vars) 
    input_vars.add(var.getName(), var.describe());
  vars.add("input", input_vars);
  
  NamedList output_vars;
  for (const auto& var : _output_vars) {
    output_vars.add(var.getName(), var.describe());
  }
  vars.add("output", output_vars);

  return vars;
}

FuzzyVariablesDB FuzzyVariablesDB::load(const string& content) {
  return load(NamedList::parse(content));
}

FuzzyVariablesDB FuzzyVariablesDB::load(const NamedList& desc) {
  // ASSUMPTIONS: at least one input and one output var
  // all input vars have the same number of sets, idem for output vars
  const auto& input = desc.get_list("input");
  const auto& output = desc.get_list("output");

  int nb_input_sets = 0;
  int nb_output_sets = 0;

  if (!input.empty()) 
    nb_input_sets = input[0].size();
  if (!output.empty())
    nb_output_sets = output[0].size();

  FuzzyVariablesDB db(input.names(), nb_input_sets, output.names(), nb_output_sets);
  db.setPositions(input);
  db.setPositions(output);

  return db;
}//KCOV IGNORE

FuzzyVariablesDB FuzzyVariablesDB::subset(const vector<int>& input_var_idx, const vector<int>& output_var_idx) const {
  FuzzyVariablesDB db2;
  
  const int nbin = getNbInputVars();
  for (int idx : input_var_idx) {
    assert(idx >= 0 && idx < nbin);
    db2._input_vars.push_back(_input_vars[idx]);
  }

  const int nbout = getNbOutputVars();
  for (int idx : output_var_idx) {
    assert(idx >= 0 && idx < nbout);
    db2._output_vars.push_back(_output_vars[idx]);
  }

  return db2;
} //KCOV IGNORE

vector<string> FuzzyVariablesDB::build_default_var_names(int nbvars, const string& base_name)
{
  vector<string> names;
  names.reserve(nbvars);
  for (int i = 0; i < nbvars; i++)
    names.push_back(base_name + '_' + std::to_string(i + 1));
  return names;
}//KCOV IGNORE

map<string, int> FuzzyVariablesDB::build_var_names_to_index_mapper(const vector<FuzzyVariable>& vars)
{
  const int nb = vars.size();
  map<string, int> var_to_idx;
  for (int i = 0; i < nb; i++)
    var_to_idx[vars[i].getName()] = i;
  return var_to_idx;
}//KCOV IGNORE

map<string, int> FuzzyVariablesDB::build_set_names_to_index_mapper(const FuzzyVariable& var)
{
  map<string, int> set_to_idx;
  for (int j = 0; j < var.getSetsCount(); j++)
    set_to_idx[var.getSet(j).getName()] = j;
  return set_to_idx;
}//KCOV IGNORE

vector< map<string, int> > FuzzyVariablesDB::build_vars_set_names_to_index_mapper(const vector<FuzzyVariable>& vars)
{
  const int nb = vars.size();
  vector< map<string, int> > set_names_to_idx(nb);
  for (int i = 0; i < nb; i++) 
    set_names_to_idx[i] = build_set_names_to_index_mapper(vars[i]);
  
  return set_names_to_idx;
}//KCOV IGNORE

namespace fuzzy_coco {
  ostream& operator<<(ostream& out, const FuzzyVariablesDB& db) 
  {
    out << "Fuzzy Variables Database:" << endl;
    out << "## input variables:" << endl;
    for (const auto& var : db._input_vars)
      out << var << endl;

    out << "## output variables:" << endl;
    for (const auto& var : db._output_vars)
      out << var << endl;
    return out;
  }
}