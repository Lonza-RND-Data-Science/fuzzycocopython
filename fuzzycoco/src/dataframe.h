/**
    * @file   dataframe.h
  * @author Karl Forner <karl.forner@gmail.com>
  * @date   09.2024
  * @brief a poor man's data frame
  * @class DataFrame: assumes that the first column contains the rownames, i.e the subjects
  * and that all other columns are numerical. Also handle missing data
  * 
  */

#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <vector>
#include <string>
#include <cassert>

#include "types.h"  // for MISSING_DATA_DOUBLE

namespace fuzzy_coco {
using namespace std;

typedef vector<double> NumColumn;
typedef vector<bool> BoolColumn;
typedef vector<string> StrColumn;

class DataFrame {
public:
  // empty dataframe
  DataFrame(int nbrows = 0, int nbcols = 0) { reset(nbrows, nbcols); }

  // a convenient constructor for tests
  DataFrame(const string& csv, bool rownames);
  DataFrame(const vector<vector<string>>& rows, bool rownames) {
    assign(rows, rownames);
  }


  // parse and assign data from strings (as parsed from a CSV, cf FuzzyCocoUtils::parseCSV())
  // rownames: if TRUE, consider the first column as the rownames
  void assign(const vector<vector<string>>& rows, bool rownames);

  void reset(int nbrows = 0, int nbcols = 0);

  // extract a dataframe from this one with only columns from col1 --> col2. All columns from col1 
  DataFrame subsetColumns(int col1, int col2) const;
  // to col2 (included) are in the returned dataframe

  DataFrame subsetColumns(const vector<int>& col_idx) const;
  DataFrame subsetColumns(const vector<string>& col_names) const;

  // accessors
  int nbrows() const { return _nbrows; }
  int nbcols() const { return _nbcols; }

  bool hasRowNames() const { return !_rownames.empty(); }
  const StrColumn& rownames() const { return _rownames; }
  const StrColumn& colnames() const { return _colnames; }

  void colnames(const vector<string>& names);
  void rownames(const vector<string>& names);

  const NumColumn& getColumn(int col) const { return _cols[col]; }
  const NumColumn& operator[](int col) const { return getColumn(col); }
  
  NumColumn fetchRow(int row) const; 

  double at(int row, int col) const {
    check_indexes(row, col);
    return _cols[col][row]; 
  }
  bool missing(int row, int col) const {
    check_indexes(row, col);
    return is_na(at(row, col));
  }

  void set(int row, int col, double value) { _cols[col][row] = value; }
  void fillRow(int row, const vector<double>& values);
  void fillCol(int col, const NumColumn& values);

  void check_indexes(int row, int col) const { 
    assert(row >=0 && row < _nbrows && col >=0 && col < _nbcols);
  }

  bool operator!=(const DataFrame& df) const { return ! (*this == df); }
  bool operator==(const DataFrame& df) const {
    return _cols == df._cols && 
    _rownames == df._rownames &&
    _colnames == df._colnames;
  }

  static DataFrame load(const string& filename, bool rownames);

  friend ostream& operator<<(ostream& out, const DataFrame&);

private:
  int _nbcols = 0;
  int _nbrows = 0;
  vector<NumColumn> _cols;
  StrColumn _rownames;
  StrColumn _colnames;
};


}
#endif // DATAFRAME_H
 