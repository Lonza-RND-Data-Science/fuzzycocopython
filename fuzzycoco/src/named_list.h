#ifndef NAMED_LIST_H
#define NAMED_LIST_H

// implement a named list (like in R)
#include <iostream>
#include <string>
#include <memory>
#include <cassert>
#include <variant>
#include <vector>
#include <map>

namespace fuzzy_coco {

using namespace std;

class Scalar : public variant<monostate, bool, int, double, string> {
public:
    using variant<monostate, bool, int, double, string>::variant;
    // inherit constructors

    bool is_bool() const { return holds_alternative<bool>(*this);  }
    bool is_int() const { return holds_alternative<int>(*this);  }
    bool is_double() const { return holds_alternative<double>(*this);  }
    bool is_string() const { return holds_alternative<string>(*this);  }
    bool is_null() const { return holds_alternative<monostate>(*this); }
    bool is_numeric() const { return is_int() || is_double(); }

    const string& get_string() const { return get<string>(*this); }
    int get_int() const { return get<int>(*this); }
    bool get_bool() const { return get<bool>(*this); }
    double get_double() const { return get<double>(*this); }
    double get_numeric() const { return is_int() ? get_int() : get_double(); }
    int get_as_int() const {
      double v = get_numeric();
      int i = static_cast<int>(v);
      if (i != v) throw runtime_error("numeric value is not an integer: " + std::to_string(v));
      return i;
    }

    // bool operator==(bool b) const { return is_bool() &&  get<bool>(*this) == b; }
    bool operator==(int i) const { return is_int() &&  get<int>(*this) == i; }
    bool operator==(double d) const { return is_double() &&  get<double>(*this) == d; }
    bool operator==(const string& s) const { return is_string() &&  get<string>(*this) == s; }

    void print(ostream& out) const;
    static Scalar parse(istream& in);
    static Scalar parse(const string& content);

    friend ostream& operator<<(ostream& out, const Scalar& scalar) {
        scalar.print(out);
        return out;
    }
};

class NamedList {
public:
    NamedList() {}
    NamedList(const string& name) { _name = name; }
    NamedList(const string& name, bool b)  { _name = name; _value = b; }
    NamedList(const string& name, int i)  { _name = name; _value = i; }
    NamedList(const string& name, const string& s)  { _name = name; _value = s; }
    NamedList(const string& name, double v) { _name = name; _value = v; }
    NamedList(const string& name, const Scalar& scalar) : _name(name), _value(scalar) {}

public: // ========= main read-only user interface =================
  bool empty() const { return is_list() && size() == 0; }
  bool is_scalar() const { return !_value.is_null(); }
  bool is_list() const { return !is_scalar(); }
  // N.B: size==0 for scalar
  size_t size() const { return _children.size(); }

  // accessing elements
  const NamedList& operator[](int idx) const { return _children.at(idx); }
  NamedList& operator[](int idx)  { return _children.at(idx); }

  // // access elements by name: N.B linear in size(), only the first matching name is returned, or an exception
  // is thrown
  const NamedList& operator[](const string& name) const { return fetch(name); }
  NamedList& operator[](const string& name)  { return fetch(name); }

  const string& name() const { return _name; }
  vector<string> names() const;

  bool has(const string& name) const { return find_first_idx(name) >= 0; }

  const NamedList& get_list(const string& name) const { return fetch(name); }
  const NamedList& get_list(const string& name, const NamedList& default_lst) const;
  const string& get_string(const string& name) const { return fetch_scalar(name).get_string(); }
  bool get_bool(const string& name) const { return fetch_scalar(name).get_bool(); }
  int get_int(const string& name) const { return fetch_scalar(name).get_int(); }
  double get_double(const string& name) const { return fetch_scalar(name).get_double(); }
  double get_numeric(const string& name) const { return fetch_scalar(name).get_numeric(); }
  int get_as_int(const string& name) const { return fetch_scalar(name).get_as_int(); }

  const string& get_string(const string& name, const string& default_value) const;
  bool get_bool(const string& name, bool default_value) const;
  int get_int(const string& name, int default_value) const;
  int get_as_int(const string& name, int default_value) const;
  double get_double(const string& name, double default_value) const;

  const string& get_string() const { return scalar_check().get_string(); }
  int get_int() const { return scalar_check().get_int(); }
  double get_double() const { return scalar_check().get_double(); }
  double get_numeric() const { return scalar_check().get_numeric(); }

public: // ========= main setter user interface =================
  void add(const string& name, bool b) { add({name, b}); }
  void add(const string& name, int i) { add({name, i}); }
  void add(const string& name, double d) { add({name, d}); }
  void add(const string& name, const char* s) { add({name, string(s)}); }
  void add(const string& name, const string& s) { add({name, s}); }
  void add(const string& name, const NamedList& elt);

  void add(const string& name, const vector<double>& v);
  void add(const string& name, const map<string, double>& hash);

public: //

  // N.B: not optimized: check all names and return the first found...
  // if not found throw runtime_error
  // const shared_ptr<NamedList> fetch(const string& name) const;
  const NamedList& fetch(const string& name) const;
  NamedList& fetch(const string& name);

  const Scalar& fetch_scalar(const string& name) const { return fetch(name).scalar(); }

  // N.B: return -1 if item is not found or is scalar
  
  int find_first_idx(const string& name) const;
  // iterators
  // Define the iterator type (use underlying std::vector's iterator for simplicity)
  using iterator = vector<NamedList>::iterator;
  using const_iterator = vector<NamedList>::const_iterator;
  iterator begin() { return _children.begin(); }
  iterator end() { return _children.end(); }
  const_iterator begin() const { return _children.begin(); }
  const_iterator end() const { return _children.end(); }

  const Scalar& scalar_check() const { 
    if (!is_scalar()) throw runtime_error("not a scalar!");
    return _value; 
  }
  const Scalar& scalar() const { return _value; }
  const Scalar& value() const { return _value; }

  void print(ostream& out, int indent = 0, bool toplevel = true) const;

  bool operator==(const NamedList& l) const;

  static NamedList parse(istream& in);
  static NamedList parse(const string& content);

  friend ostream& operator<<(ostream& out, const NamedList& list) {
      list.print(out);
      return out;
  }

  vector<double> as_numeric_vector() const;
  map<string, double> as_string_numeric_map() const;

  string to_string() const;

protected:
  void add(const NamedList& elt) {
      assert(is_list());
      _children.push_back(elt);
  } 

private:
    string _name;
    Scalar _value;
    vector<NamedList> _children;
};

  const string NA_INT_STRING = "NA";
  const string NA_DOUBLE_STRING = "NA.";

}
#endif // NAMED_LIST_H