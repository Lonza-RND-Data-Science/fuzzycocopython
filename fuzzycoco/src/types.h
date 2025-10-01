/**
  * @file   types.h
  * @author Karl Forner <karl.forner@gmail.com>
  * @date   09.2024
  * @brief common types definitions
  */


#ifndef FUZZY_COCO_TYPES_H
#define FUZZY_COCO_TYPES_H

#include <limits>
#include <iostream>
#include <vector>
#include <string>

namespace fuzzy_coco {

using namespace std;

// using double_or_missing = double;
// sentinel value to encode for missing data in a double var
const double MISSING_DATA_DOUBLE = numeric_limits<double>::lowest();
const double NA_D = MISSING_DATA_DOUBLE;
const double INFINITY_DOUBLE = numeric_limits<double>::infinity();
const int MISSING_DATA_INT = numeric_limits<int>::lowest();
const int NA_I = MISSING_DATA_INT;

inline bool is_na(double v) { return v == MISSING_DATA_DOUBLE; }
inline bool is_na(int v) { return v == MISSING_DATA_INT; }



// inline 

template<typename T>
ostream& operator<<(ostream& out, const vector<T>& v) {
  out << "{";
  const int nb = v.size() - 1;
  for (int i = 0; i < nb; ++i) {
      out << v[i] << ", "; 
  }
  if (nb >= 0) out << v[nb];
  out << "}";
  return out;
}



inline void throwWithLocation(const string& message, const char* file, int line, const char* func) {
    string fullMessage = "Exception: " + message + "\n"
                              + "File: " + string(file )
                              + ":" + std::to_string(line) + string("\n")
                              + string("Function: ") + string(func);
    throw runtime_error(fullMessage);
}

// A macro to simplify throwing exceptions with location info
#define THROW_WITH_LOCATION(msg) throwWithLocation(msg, __FILE__, __LINE__, __func__)



}
#endif // FUZZY_COCO_TYPES_H
