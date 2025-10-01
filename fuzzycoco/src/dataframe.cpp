#include "dataframe.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <unordered_map>
#include "file_utils.h"

using namespace fuzzy_coco;

DataFrame::DataFrame(const string& csv, bool rownames) {
  vector<vector<string>> tokens;
  FileUtils::parseCSV(csv, tokens);
  assign(tokens, rownames);
}

void DataFrame::reset(int nbrows, int nbcols) {
  _nbrows = nbrows;
  _nbcols = nbcols;
  _rownames.resize(0);
  _colnames.resize(0);
  _colnames.resize(nbcols);
  _cols.resize(nbcols);
  for (auto& col : _cols) {
    col.resize(0);
    col.resize(nbrows, 0);
  }
}

void DataFrame::assign(const vector<vector<string>>& rows, bool rownames) {
  int first_col = rownames ? 1 : 0;

  // total number of columns
  int nbrows = rows.size();
  if (nbrows == 0) {
    THROW_WITH_LOCATION("Error in DataFrame::assign(): no rows");
  }

  auto nbcols = rows[0].size();
  if (nbcols == 0) {
    THROW_WITH_LOCATION("Error in DataFrame::assign(): no columns");
  }
  //
  reset(nbrows - 1, nbcols - first_col);
  _colnames = rows[0];

  if (rownames) {
    // rownames
    _rownames.resize(0);
    for (int i = 1; i < nbrows; i++) {
      _rownames.push_back(rows[i][0]);
    }
  }

  // colnames: N.B: we skip the rownames colname
  // _rownames_name = rows[0][0];
  _colnames.assign(rows[0].begin() + first_col, rows[0].end());

  // columns
  int row = 0, col = 0;
  for (size_t j = first_col; j < nbcols; j++) {
    col = j - first_col;
    for (int i = 1; i < nbrows; i++)  {
      if (rows[i].size() != nbcols) {
          THROW_WITH_LOCATION(string("Error in DataFrame::assign(): bad number of columns:") + std::to_string(rows[i].size()) + " at row " + std::to_string(i) );
      }
      row = i - 1;
      try {
        _cols[col][row] = stod(rows[i][j]);
      } catch(...) {
        _cols[col][row] = MISSING_DATA_DOUBLE;
      }
    }
  }
}

void DataFrame::colnames(const vector<string>& names) {
  if (names.size() != (size_t)nbcols())
      THROW_WITH_LOCATION("bad colnames size!");
  _colnames = names; 
}

void DataFrame::rownames(const vector<string>& names) {
  if (names.size() != (size_t)nbrows() && names.size() != 0)
      THROW_WITH_LOCATION("bad rownames size!");
  _rownames = names; 
}


NumColumn DataFrame::fetchRow(int row) const {
  assert(row >= 0 && row < _nbrows);
  NumColumn values;
  values.resize(nbcols());
  for (int col = 0; col < _nbcols; col++) {
    values[col] = _cols[col][row];
  }
  return values;
}//KCOV IGNORE

void DataFrame::fillRow(int row, const vector<double>& values) {
  assert(row >= 0 && row < _nbrows);
  for (int col = 0; col < _nbcols; col++) {
    _cols[col][row] = values[col];
  }
}

void DataFrame::fillCol(int col, const NumColumn& values) {
  assert(col >= 0 && col < _nbcols);
  _cols[col] = values;
}

DataFrame DataFrame::subsetColumns(int col1, int col2) const {
  assert(col1 >= 0 && col1 < _nbcols);
  assert(col2 >= col1 && col2 < _nbcols);

  if (col1 > col2) swap(col1, col2);

  const int nb = col2 - col1 + 1;
  vector<int> col_idx(nb);
  for (int i = 0; i < nb; i++)
    col_idx[i] = col1 + i;

  return subsetColumns(col_idx);
}

DataFrame DataFrame::subsetColumns(const vector<int>& col_idx) const {
  DataFrame df;
  df.reset(nbrows(), col_idx.size());
  df._rownames = _rownames;

  for (int i = 0; i < df._nbcols; i++) {
    if (! (col_idx[i] >=0 && col_idx[i] < _nbcols))
      THROW_WITH_LOCATION("bad column index " + std::to_string(col_idx[i]));
    int col = col_idx[i];
    df._colnames[i] = _colnames[col];
    df._cols[i] = _cols[col];
  }

  return df;
}//KCOV IGNORE

DataFrame DataFrame::subsetColumns(const vector<string>& col_names)  const{
  unordered_map<string, int> colname_to_idx;
  colname_to_idx.reserve(_nbcols);
  for (int i = 0; i < _nbcols; i++) {
      if (_colnames[i].empty())
          THROW_WITH_LOCATION("in subsetColumns(col_names), the data frame must have defined colnames");
      colname_to_idx[_colnames[i]] = i;
  }

  const int nb = col_names.size();
  vector<int> col_idx(nb);
  for (int i = 0; i < nb; i++) {

    auto search = colname_to_idx.find(col_names[i]);
    if (search == colname_to_idx.end()) // not found
      THROW_WITH_LOCATION("column name " + col_names[i] + " can not be found in the data frame");
    col_idx[i] = search->second;
  }

  return subsetColumns(col_idx);
}

DataFrame DataFrame::load(const string& filename, bool rownames)
{
  vector<vector<string>> tokens;
  tokens.reserve(1000);
  FileUtils::parseCSV(path(filename), tokens);
  DataFrame df(tokens, rownames);
  // cerr << df;
  return df;
}

namespace fuzzy_coco {
  ostream& operator<<(ostream& out, const DataFrame& df) {
    const auto& cns = df.colnames();
    const auto& rns = df.rownames();
    int nbcols = df.nbcols();
    int nbrows = df.nbrows();

    if (nbcols == 0 || nbrows == 0) {
      out << "empty DataFrame (0 columns)" << endl;
      return out;
    }

    auto longest = [](const string& s1, const string& s2) {return s1.length() < s2.length(); };
    int rownamesw = 0;
    if (df.hasRowNames()) {
      auto result = max_element(df.rownames().cbegin(), df.rownames().cend(), longest);
      rownamesw = result->length();
    }
    
    string DELIM = "|";
    string MISSING = "NA";
    int numwidth = 4;
    char SPACE = ' ';

    // compute the width of each column

    vector<int> widths;
    for (const auto& cn: cns) widths.push_back(max(int(cn.length()), numwidth));

    auto extend = [SPACE](const string& s, int width) {
      int missing = width - s.length();
      return s + string(missing, SPACE);
    };

    // print the header line
    out << string(rownamesw, SPACE) << SPACE << DELIM;
    for (int col = 0; col < nbcols; col++) {
      out << SPACE << extend(cns[col], widths[col]) << SPACE << DELIM;
    }
    out << endl;
    // print the separation line
    out << string(rownamesw, '-') << SPACE << DELIM;
    for (int col = 0; col < nbcols; col++) {
      out << string(widths[col] + 2, '-')  << DELIM;
    }
    out << endl;

    // print each row
    for (int row = 0; row < nbrows; row++) {
      // print row name
      if (df.hasRowNames())
        out << extend(rns[row], rownamesw) ;
      out << SPACE << DELIM;
      // print values
      for (int col = 0; col < nbcols; col++) {
        out  << SPACE;
        if (df.missing(row, col))
          out << extend(MISSING, widths[col]);
        else
          out << setw(widths[col]) << df[col][row];
        out << SPACE << DELIM;
      }
      out << endl;
    }
    out << endl;

    return out;
  }
}//KCOV IGNORE
