#ifndef TEST_H
#define TEST_H

#include <gtest/gtest.h>


// utilities for the unit tests
#include <numeric>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "named_list.h"

using namespace std;
using namespace fuzzy_coco;

auto sum = [](auto v) { return std::accumulate(v.begin(), v.end(), 0); };

template<typename T, class UnaryPred>
bool all(const vector<T>& v, UnaryPred pred) { 
  return std::all_of(v.cbegin(), v.cend(), pred);
}

template<typename T>
bool all_zero(const vector<T>& v) { return all(v, [](T t) { return t == 0; });
}

bool cmp(const NamedList& l1, const NamedList& l2, double epsilon = 1e-2) {
  if (l1.is_scalar()) {
    if (!l2.is_scalar()) return false;

    if (!l1.scalar().is_double()) return l1 == l2;
    if (!l1.scalar().is_double()) return false;
    return (abs(l1.get_double() - l2.get_double()) < epsilon);
  }

  const size_t nb = l1.size();
  if (nb != l2.size()) return false;
  for (size_t i = 0; i < nb; i++) {
    if (! cmp(l1[i], l2[i], epsilon) ) return false;
  }
  
  return true;  
}


// template<typename T>
// ostream& operator<<(ostream& out, const vector<T>& v) {
//   out << "{";
//   const int nb = v.size() - 1;
//   for (int i = 0; i < nb; ++i) {
//       out << v[i] << ", "; 
//   }
//   if (nb >= 0) out << v[nb];
//   out << "}";
//   return out;
// }



#endif