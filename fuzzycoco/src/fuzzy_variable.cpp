#include "fuzzy_variable.h"
using namespace fuzzy_coco;

FuzzyVariable::FuzzyVariable(string name, const vector<string>& set_names) : _name(std::move(name))
{ 
  _sets.reserve(set_names.size());
  for (const auto& name : set_names)
    _sets.push_back({name});
}

FuzzyVariable::FuzzyVariable(string name, int nbsets) 
  : FuzzyVariable(name, build_default_set_names(nbsets, name)) {}

vector<string> FuzzyVariable::build_default_set_names(int nbsets, const string& set_base_name)
{
  vector<string> set_names;
  set_names.reserve(nbsets);
  for (int i = 0; i < nbsets; i++)
    set_names.push_back(set_base_name + '.' + std::to_string(i + 1));
  return set_names;
}//KCOV IGNORE

/**
  * Return the index of an attached set (for example : by default MF0 is 0, MF1 is 1, ...).
  *
  * @param setName Name of the set to be retrieved.
  */
int FuzzyVariable::getSetIndexByName(const string& name)
{
  int nb = getSetsCount();
  for (int i = 0; i < nb; i++)
  {
    if (getSet(i).getName() == name)
      return i;
  }
  throw out_of_range("set name not found");
}

NamedList FuzzyVariable::describe() const {
  NamedList lst(getName());
  int nb = getSetsCount();
  for (int i = 0; i < nb; i++)
    lst.add(getSet(i).getName(), getSet(i).describe());

  return lst;
}//KCOV IGNORE

FuzzyVariable FuzzyVariable::load(const NamedList& desc) {
  FuzzyVariable var(desc.name(), desc.size());
  var.setSetsPositions(desc);
  return var;
}//KCOV IGNORE

FuzzyVariable FuzzyVariable::load(const string& content) {
  return load(NamedList::parse(content));
}

void FuzzyVariable::setSetsPositions(const NamedList& lst)
{
  if (!lst.is_list()) throw runtime_error("not a list");
  const int nb = lst.size();
  if (nb != getSetsCount()) throw runtime_error("incompatible number of sets");
  for (int i = 0; i < nb; i++) {
    getSet(i).setName(lst[i].name());
    getSet(i).setPosition(lst[i].scalar().get_numeric());
  }
}

void FuzzyVariable::printDescription(ostream& out, const NamedList& desc)  {
  out << desc;
}

namespace fuzzy_coco {
  ostream& operator<<(ostream& out, const FuzzyVariable& var) 
  {
    var.printDescription(out, var.describe());
    return out;
  }
}

// ostream& operator<<(ostream& out, const FuzzyInputVariable& var) 
// {
//   out << "FuzzyInputVariable " << static_cast<const FuzzyVariable&>(var);
//   return out;
// }

// ostream& operator<<(ostream& out, const FuzzyOutputVariable& var) 
// {
//   out << "FuzzyOutputVariable " << static_cast<const FuzzyVariable&>(var);
//   return out;
// }

double FuzzyVariable::fuzzify(int set_idx, double value) const
{
  // TODO: essayer de supprimer les protections pour voir la diff de vitesse
  // TODO: zone critique en temps à essayer d'optimiser
  assert(getSetsCount() > 1);
  assert(set_idx >= 0 && set_idx < getSetsCount());
  const int lastSetNum = getSetsCount() - 1;

  const double position = getSet(set_idx).getPosition();

  if (is_na(position))
    return MISSING_DATA_DOUBLE;

  // In any case
  if (value == position)
    return 1.0;

  // Last set : part after pn = 1 AND Internal Set with value < than position
  if (set_idx == lastSetNum || (set_idx != 0 && value < position))
  {

    if (value > position)
      return 1.0;

    // The set num is not the first!!!
    const double beforePosition = getSet(set_idx - 1).getPosition();

    if (value <= beforePosition)
      return 0.0;
    else
      return (value - beforePosition) / (position - beforePosition);
  }
  else
  {
    // It's the first set

    if (value < position)
      return 1.0;

    // the set num is not the last (tested just before), so I can do that without checking setList size
    const double afterPosition = getSet(set_idx + 1).getPosition();

    if (value >= afterPosition)
      return 0.0;
    else
      return 1.0 - ((value - position) / (afterPosition - position));
  }
}

// FuzzyInputVariable FuzzyInputVariable::load(const NamedList& desc) {
//   FuzzyInputVariable var(desc.name(), desc.size());
//   var.setSetsPositions(desc);
//   return var;
// }
// FuzzyInputVariable FuzzyInputVariable::load(const string& content) {
//   return load(NamedList::parse(content));
// }



// FuzzyOutputVariable FuzzyOutputVariable::load(const NamedList& desc) {
//   FuzzyOutputVariable var(desc.name(), desc.size());
//   var.setSetsPositions(desc);
//   return var;
// }
// FuzzyOutputVariable FuzzyOutputVariable::load(const string& content) {
//   return load(NamedList::parse(content));
// }

/**
  * Compute the defuzzificated value for this variable. The variable
  * must be an output variable and its sets must have been evaluated
  * first.
  *
  * @param precision Precision of the defuzzification process. If the
  * defuzzification method choosed is COA, it represents the step (on x-axis)
  * of the discrete sum.
  */
  // Karl: embedded DefuzzMethodSingleton::defuzzVariable
  // missingness support:
  // - is a set position OR set evaluation is missing --> the set is ignored
  // - if all sets are ignored --> return MISSING DATA
double FuzzyVariable::defuzz(const vector<double>& set_evals) const {
  // - if not set has fired --> 0
  const int nb = getSetsCount();
  assert(nb > 0 && set_evals.size() == (size_t)nb);
  double evalSum = 0.0;
  double evalProduct = 0.0;
  int nb_non_missing = 0;
  for (int i = 0; i < nb; i++) {
      const double eval = set_evals[i];
      const double pos = getSet(i).getPosition();
      if (!is_na(pos) && !is_na(eval)) {
        nb_non_missing++;
        evalSum += eval;
        evalProduct += (eval * pos);
      }
  }
  // Karl: all sets ignored --> NA
  if (nb_non_missing == 0) return MISSING_DATA_DOUBLE;
  return (evalSum == 0.0) ? 0.0 : evalProduct / evalSum;
}

/**
  * Set the default rule activation level to the corresponding set.
  *
  * @param setNum Index of the set referred by the default rule.
  * @param value Level of default rule activation.
  */
// void FuzzyVariable::setDefaultRule(int setNum, double value)
// {
//     this->getSet(setNum)->setEval(value);
// }
