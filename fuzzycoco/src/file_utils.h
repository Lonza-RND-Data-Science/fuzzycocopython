
#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <string>
#include <vector>
#include <filesystem>
using namespace std;
using namespace std::filesystem;
#include "dataframe.h"

namespace fuzzy_coco {

namespace FileUtils
{

  void mkdir_if_needed(path file_path);

  void parseCSV(const string& content, vector<vector<string>>& tokens, char delim = ';');
  void parseCSV(istream& in, vector<vector<string>>& tokens, char delim = ';');
  void parseCSV(const path& filename, vector<vector<string>>& tokens, char delim = ';');

  // N.B: do not write the rownames
  void writeCSV(ostream& out, const DataFrame& df, char delim = ';');

  string slurp(const path& filename);

  // a poor man's version of tmpnam (forbidden by some compilers, not secure but only used in tests
  string poor_man_tmpnam(string prefix, string dir = temp_directory_path().string());

};


}
#endif // FILE_UTILS_H
