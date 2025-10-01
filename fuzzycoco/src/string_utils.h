#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <string>

namespace fuzzy_coco {

using namespace std;
namespace StringUtils {
  // remove trailing zeroes except the one just after the dot
  // i.e 1.100 --> 1.1 but 1.00000 -> 1.0
  // --> a double is distinguishable from an integer
  string prettyDistinguishableDoubleToString(double value);

  // remove comment lines from a (multiline) string
  string stripComments(const string& content);

}

}
#endif // STRING_UTILS_H