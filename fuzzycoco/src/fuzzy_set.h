
#ifndef FUZZYSET_H
#define FUZZYSET_H

#include <string>
#include <iomanip>

#include "types.h"
#include "named_list.h"
#include "string_utils.h"

namespace fuzzy_coco {

using namespace std;

class FuzzySet
{
public:
    FuzzySet(string name, double pos = MISSING_DATA_DOUBLE) : _name(name), _position(pos) {}
    FuzzySet(const FuzzySet& set) : _name(set._name), _position(set._position) {}
    FuzzySet(FuzzySet&& set) : _name(std::move(set._name)), _position(set._position) {}
    ~FuzzySet() {}
    FuzzySet& operator=(FuzzySet set) { 
      swap(_name, set._name);
      _position = set._position;
      return *this;
    }

    const string& getName() const { return _name; }
    // Retrieve the position of the set.
    double getPosition() const {return _position; }
    // Set the position of the set.
    void setPosition(double pos) { _position = pos; }
    void setName(const string& name) { _name = name; }

    NamedList describe() const {
      return NamedList(getName(), getPosition());
    }
    static FuzzySet load(const NamedList& desc) {
      return FuzzySet(desc.name(), desc.scalar().get_double());
    }

    static void printDescription(ostream& out, const NamedList& desc) {    out << desc;  }

    // for debug purposes
    bool operator==(const FuzzySet& set) const { 
      return getName() == set.getName() && getPosition() == set.getPosition();
    } 

private:
    string _name;
    double _position;
};

inline ostream& operator<<(ostream& out, const FuzzySet& set) {
  set.printDescription(out, set.describe());
  return out;
}

}
#endif // FUZZYSET_H
