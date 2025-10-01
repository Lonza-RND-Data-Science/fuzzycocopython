
#include <sstream>
#include <algorithm>
#include <regex>
#include "string_utils.h"

using namespace fuzzy_coco;

string StringUtils::prettyDistinguishableDoubleToString(double value) {
  ostringstream oss;
  // Use fixed and setprecision to avoid scientific notation
  oss << fixed << value;
  
  string result = oss.str();
  
  int idx = result.find_last_not_of('0') + 1;
  int dot_idx = result.find_last_of('.');
  int zeroes_idx = max(idx, dot_idx + 2);
  if ((size_t)zeroes_idx < result.length()) {
      result.erase(zeroes_idx, string::npos);
  }
  return result;
}

string StringUtils::stripComments(const string& content)
{
  regex inline_comment_regex(R"(#[^"]*$)");  // Match # and everything after
  regex commented_line_regex(R"(^\s*#.*$)");
  istringstream lines(content);
  ostringstream out;
  string line;

  while (getline(lines, line)) {
    // Strip comment from line
    line = regex_replace(line, inline_comment_regex, "");
    line = regex_replace(line, commented_line_regex, "");
    // Trim trailing whitespace (optional)
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (!line.empty())
      out << line << '\n';
  }

  return out.str();
}