#include <iomanip>

#include "named_list.h"
#include "types.h"
#include "string_utils.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace logging;



void Scalar::print(ostream& out) const {
    if (is_null()) {
      // nothing
      // out << "-";
    } else if (is_int()) {
      if (is_na(get_int()))
        out << quoted(NA_INT_STRING);
      else
        out << get_int();
    } else if (is_double()) {
      if (is_na(get_double()))
        out << quoted(NA_DOUBLE_STRING);
      else     
        out << StringUtils::prettyDistinguishableDoubleToString(get_double());
    } else if (is_string()) {
      out << quoted(get_string());
    } else if (is_bool()) {
      out << boolalpha << get_bool();
    } else {
      // MUST NOT HAPPEN
    }
}

Scalar Scalar::parse(istream& in) {
  const char QUOTE = '"';
  string item;
  in >> item;

  logger() << "Scalar::parse(): item='" << item << "'\n"; 

  // null
  if (item.empty()) return Scalar();

  if (item.back() == '}')
  in.unget();

  if (item.at(0) == QUOTE) { // string
    int last_quote_idx = item.find_last_of(QUOTE);
    string s(item.begin() + 1, item.begin() + last_quote_idx);

    // it may by a NA string
    if (s == NA_DOUBLE_STRING)
      return Scalar(MISSING_DATA_DOUBLE);
    else if (s == NA_INT_STRING)
      return Scalar(MISSING_DATA_INT);

    return Scalar(s);
  }

  // bool case
  if (item[0] == 't' || item[0] == 'f') {
    if (item.length() >= 4 && item.substr(0, 4) == "true") 
      return Scalar(true);
    if (item.length() >= 5 && item.substr(0, 5) == "false") 
      return Scalar(false);
    THROW_WITH_LOCATION("in Scalar::parse, unable to parse item: " + item);
  }

  // look for the dot
  if (item.find_first_of('.') != item.npos) // double
    return Scalar(stod(item));

  return Scalar(stoi(item));
}
Scalar Scalar::parse(const string& content) {
  istringstream str = istringstream(content);
  return parse(str);
}

bool NamedList::operator==(const NamedList& l) const {
  if (_name != l._name || _value != l._value) return false;

  const size_t nb = _children.size();
  if (nb != l._children.size()) return false;
  for (size_t i = 0; i < nb; i++) {
      if (! (_children[i] == l._children[i]) ) return false;
  }
  
  return true;
}


NamedList NamedList::parse(const string& content) {
    istringstream str = istringstream(content);
    return parse(str);
}

NamedList NamedList::parse(istream &in)
{
  logger() << "NamedList::parse()\n";
  const char QUOTE = '"';
  const char SPACE = ' ';
  const char COMMA = ',';



  auto peek_next_significant = [&](){ while(in && in.peek() <= SPACE) in.get(); return in.peek(); };
  auto skip_comma = [&](){ if (in.peek() == COMMA) in.get(); };

  char ch = peek_next_significant();

  logger() << "NamedList::parse() ch='" << ch << "'\n";

  if (in.eof()) return(NamedList());

  if (ch == '}')
  { // end of list
    THROW_WITH_LOCATION(string("parsing error, current character=") + ch);
  }

  if (ch == '{')
  { // list
    NamedList list;
    in.get(); // consume {
    while (in && peek_next_significant() != '}')
    {
      skip_comma();
      auto child = NamedList::parse(in);
      skip_comma();
      list.add(child);
    }
    in.get(); // consume }

    return list;
  }

  if (ch == QUOTE)
  { // element
    char c = 0;
    in.get(c); // consume "
    string elt_name;
    getline(in, elt_name, QUOTE);
    c = peek_next_significant();
    assert(c == ':');
    in.get(c); // consume :
    if (peek_next_significant() == '{') {
      auto sublist = NamedList::parse(in);
      skip_comma();
      sublist._name = elt_name;

      return sublist;
    }
    auto scalar = Scalar::parse(in);
    skip_comma();
    return NamedList(elt_name, scalar);
  }

  // THROW_WITH_LOCATION(string("parsing error, current character=") + ch);

  return NamedList();
}

int NamedList::find_first_idx(const string& name) const 
{
  if (!is_list()) return -1;
  const int nb = size();
  for (int i = 0; i < nb; i++)
    if ( (*this)[i].name() == name) return i;

  return -1;
}


vector<double> NamedList::as_numeric_vector() const
{
  if (!is_list()) { // scalar case
    if (!scalar().is_numeric()) THROW_WITH_LOCATION("not numeric");
    return vector<double>{scalar().get_numeric()};
  }
  vector<double> v(size());
  for (size_t i = 0; i < size(); i++) {
    if (!_children[i].is_scalar()) throw runtime_error("item is not a scalar!");
    const auto& scalar = _children[i].scalar();
    if (!scalar.is_numeric()) 
      throw runtime_error("item #" + std::to_string(i) + " is not numeric!");
    v[i] = scalar.get_numeric();
  }
  return v;
}

map<string, double> NamedList::as_string_numeric_map() const
{
  if (!is_list()) throw runtime_error("not a list!");
  map<string, double> hash;
  for (size_t i = 0; i < size(); i++) {
    if (!_children[i].is_scalar()) throw runtime_error("item is not a scalar!");
    const auto& scalar = _children[i].scalar();
    if (!scalar.is_numeric()) 
      throw runtime_error("item #" + std::to_string(i) + " is not numeric!");
    hash[_children[i].name()] = scalar.get_numeric();
  }
  return hash;
}


const NamedList& NamedList::fetch(const string& name) const {
  int idx = find_first_idx(name);
  if (idx < 0) THROW_WITH_LOCATION("name not found in list: " + name);
  return _children[idx];
}

NamedList& NamedList::fetch(const string& name) {
  int idx = find_first_idx(name);
  if (idx < 0) THROW_WITH_LOCATION("name not found in list: " + name);
  return _children[idx];
}

const NamedList& NamedList::get_list(const string& name, const NamedList& default_lst) const { 
  return has(name) ? get_list(name) : default_lst;
}

const string& NamedList::get_string(const string& name, const string& default_value) const {
  return has(name) ? get_string(name) : default_value; 
}
bool NamedList::get_bool(const string& name, bool default_value) const {
  return has(name) ? get_bool(name) : default_value; 
}
int NamedList::get_int(const string& name, int default_value) const {
  return has(name) ? get_int(name) : default_value; 
}
int NamedList::get_as_int(const string& name, int default_value) const {
  return has(name) ? get_as_int(name) : default_value; 
}
double NamedList::get_double(const string& name, double default_value) const {
  return has(name) ? get_double(name) : default_value; 
}

vector<string> NamedList::names() const {
  vector<string> res;
  res.reserve(size());
  for (const auto& child : _children)
      res.push_back(child._name);
  return res;
}//KCOV IGNORE

void NamedList::print(ostream& out, int indent, bool toplevel) const {
  char SPACE = ' ';
  int INDENT = 2;
  string SPACER(indent, SPACE);
  out << SPACER;
  if (!name().empty())
      out << quoted(name()) << ":";
  if (is_list()) {
      out << "{";
      if (!empty()) 
          out << endl;
      const int nb = size();
      for (int i = 0; i < nb; i++) {
          _children[i].print(out, indent + INDENT, false);
          if (i != (nb - 1))
              out << ",";
          out << endl;
      }
      if (!empty()) 
          out << SPACER;
      out << "}";
  } else {
      _value.print(out);
      // out << endl;
  }
  if (toplevel) 
      out << endl;

}

string NamedList::to_string() const
{
  stringstream buf;
  print(buf);
  return buf.str();
}


void NamedList::add(const string& name, const NamedList& elt) {
  assert(is_list());
  NamedList node(elt);
  node._name = name;
  _children.push_back(node);
}

void NamedList::add(const string& name, const vector<double>& v) {
  NamedList lst;
  for (size_t i = 0; i < v.size(); i++) {
    lst.add(std::to_string(i + 1), v[i]);
  }
  add(name, lst);
}

void NamedList::add(const string& name, const map<string, double>& hash) {
  NamedList lst;
  for (const auto& [key, value] : hash) {
    lst.add(key, value);
  }
  add(name, lst);
}