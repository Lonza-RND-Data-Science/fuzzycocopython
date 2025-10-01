/**
  * @file   Matrix<double>.h
  * @author Karl Forner <karl.forner@gmail.com>
  * @date   09.2024
  * @brief a poor man's numeric Matrix<double>
  * @class Matrix<double>
  * 
*/

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>

namespace fuzzy_coco {

using namespace std;

template <typename T>
class Matrix : public vector<vector<T>> {
public:
  // inherit constructors
  using vector<vector<T>>::vector;

  // empty dataframe
  Matrix(int nbrows, int nbcols) : vector<vector<T>>(nbrows) {
    assert(nbrows > 0 && nbcols > 0);
    for (auto& row : *this) row.resize(nbcols, 0);  }
  ~Matrix() {}

  int nbrows() const { return vector<vector<T>>::size(); }
  int nbcols() const { return vector<vector<T>>::front().size(); }

  void redim(int nbrows, int nbcols) {
    vector<vector<T>>::resize(nbrows);
    for (auto& row : *this) row.resize(nbcols);  
  }

  void reset() {
    for (auto& row : *this) row.assign(row.size(), 0);
  }

  inline friend ostream& operator<<(ostream& out, const Matrix<T>& m) {
    for (const auto& row : m) {
      for (const T& v : row) out << v << "\t";
      out << endl;
    }
    return out;
  }

};

}
#endif // MATRIX_H
 