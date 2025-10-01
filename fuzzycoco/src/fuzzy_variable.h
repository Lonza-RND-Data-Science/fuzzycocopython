#ifndef FUZZY_VARIABLE_H
#define FUZZY_VARIABLE_H

#include <stdexcept>
#include <cassert>
#include <iostream>
#include <vector>


#include "fuzzy_set.h"
#include "named_list.h"

namespace fuzzy_coco {
using namespace std;

// base class for Variables, basically just a name and FuzzySets container
class FuzzyVariable
{
public:
    FuzzyVariable(string name, int nbsets);
    FuzzyVariable(string name, const vector<string>& set_names);
    FuzzyVariable(string name, vector<FuzzySet> sets) : _name(std::move(name)), _sets(std::move(sets)) {}
    FuzzyVariable(const FuzzyVariable& var) : _name(var._name), _sets(var._sets) {   }
    FuzzyVariable(FuzzyVariable&& var) : _name(std::move(var._name)), _sets(std::move(var._sets)) {}
    ~FuzzyVariable() {}

public: // ================= main methods =========================
  double fuzzify(int set_idx, double input) const;
  double defuzz(const vector<double>& set_eval) const;

public: // ============ accessors / setters ========================
    string getName() const {  return _name; }

    const vector<FuzzySet>& getSets() const { return _sets; }
    void setSets(const vector<FuzzySet>& sets) { _sets = sets; }

    FuzzySet& getSet(int idx) {
      assert(idx >= 0 && idx < getSetsCount());
      return _sets[idx]; 
    }
    const FuzzySet& getSet(int idx) const { 
      assert(idx >= 0 && idx < getSetsCount());
      return _sets[idx]; 
    }
    // void addSet(FuzzySet& set) {  _sets.push_back(set); }
    int getSetsCount() const { return _sets.size(); }
    int getSetIndexByName(const string& name);

    void setSetsPositions(const string& desc) { 
      setSetsPositions(NamedList::parse(desc));
    }
    void setSetsPositions(const NamedList& lst);
    // // setter
    // void setSetPositions(const vector<double>& set_positions);

    // for testing purposes
    bool operator==(const FuzzyVariable& var) const {
      return getName() == var.getName() && getSets() == var.getSets();
    }

    static FuzzyVariable load(const NamedList& desc);
    static FuzzyVariable load(const string& content);

    NamedList describe() const;
    static void printDescription(ostream& out, const NamedList& desc);
    friend ostream& operator<<(ostream& out, const FuzzyVariable& rule);

    static vector<string> build_default_set_names(int nbsets, const string& set_base_name);
private:
    string _name;
    vector<FuzzySet> _sets;
};

// class FuzzyInputVariable : public FuzzyVariable {
// public:
//   using FuzzyVariable::FuzzyVariable;

//   // fuzzify: conpute the membership function associated with the input value (from the variable universe)
//   // N.B: the result should be in [0, 1]
//   double fuzzify(int set_idx, double input) const;
//   // static FuzzyInputVariable load(const NamedList& desc);
//   // static FuzzyInputVariable load(const string& content);

//   friend ostream& operator<<(ostream& out, const FuzzyInputVariable& rule);
// };

// class FuzzyOutputVariable : public FuzzyVariable {
// public:
//   using FuzzyVariable::FuzzyVariable;

//   double defuzz(const vector<double>& set_eval) const;
//   // static FuzzyOutputVariable load(const NamedList& desc);
//   // static FuzzyOutputVariable load(const string& content);

//   friend ostream& operator<<(ostream& out, const FuzzyOutputVariable& rule);
// };


}
#endif // FUZZY_VARIABLE_H
