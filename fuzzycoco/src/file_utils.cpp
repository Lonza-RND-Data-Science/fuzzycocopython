#include "file_utils.h"
#include <sstream>
#include <fstream>
#include <regex>

#include "types.h"

using namespace fuzzy_coco;
using namespace std;
using namespace std::filesystem;

void FileUtils::mkdir_if_needed(path file_path) {
  path dir_path = file_path.parent_path();
  if (!dir_path.empty())
    create_directories(dir_path);
}

void FileUtils::parseCSV(const path& filename, vector<vector<string>>& tokens, char delim) {
  ifstream in(filename);
  
  if (!in.is_open())
    throw std::filesystem::filesystem_error("error opening file", filename, error_code());
  return parseCSV(in, tokens);
}

void FileUtils::parseCSV(const string& content, vector<vector<string>>& tokens, char delim) {
  istringstream str = istringstream(content);
  return parseCSV(str, tokens);
}

void FileUtils::parseCSV(istream& in, vector<vector<string>>& tokens, char delim) {
  string line;
  vector<string> line_tokens;
  string token;
  const regex semicolon(string{delim});
  while(getline(in, line, '\n')) {
    // N.B: ignore any empty line
    if (line.length()) {
      copy(
        sregex_token_iterator(line.begin(), line.end(), semicolon, -1),
        sregex_token_iterator(),
        back_inserter(line_tokens));

        tokens.push_back(line_tokens);
        line_tokens.resize(0);
    }
  }
}


void print_na(ostream& out, double v) {
  if (is_na(v)) {
      out << "NA";
  } else {
      out << v;
  }
}

void FileUtils::writeCSV(ostream& out, const DataFrame& df, char delim) {
  const auto& colnames = df.colnames();
  const int nbcols = df.nbcols();
  for(int i = 0; i < nbcols; i++) {
    out << colnames[i];
    if (i != nbcols - 1) 
      out << delim;
  }
  out << endl;

  const int nbrows = df.nbrows();
  for(int i = 0; i < nbrows; i++) {
    for (int j = 0; j < nbcols; j++) {
      print_na(out, df.at(i, j));
      if (j != nbcols - 1)
        out << delim;
    }
    out << endl;
  }
}

string FileUtils::slurp(const path& filename)
{
  ifstream in(filename);
  string res;
  int size = file_size(filename);
  res.reserve(size);
  string line;
  line.reserve(size);
  while (std::getline(in, line))
    res += line + "\n";

  return res;
}

string FileUtils::poor_man_tmpnam(string prefix, string dir) {
  static int counter = 0;
  ++counter;

  // Use time + counter to reduce collisions
  auto now = time(nullptr);

  return dir + "/" + prefix +  "_" + std::to_string(now) + "_" + std::to_string(counter);
}