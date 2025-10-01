#include "tests.h"
#include <cstdio>
#include <fstream>
#include "file_utils.h"
#include "dataframe.h"

using namespace fuzzy_coco;
using namespace FileUtils;

string CSV1 = 
R"(CASE;Sepal.Length;Sepal.Width;Petal.Length;Petal.Width;OUT
subject2;4.9;3;1.4;0.2;0
subject3;4.7;3.2;1.3;0.2;0
subject4;4.6;3.1;1.5;0.2;0
subject5;5;3.6;1.4;0.2;0 
)";

string CSV_NO_ROWNAMES = 
R"(Sepal.Length;Sepal.Width;Petal.Length;Petal.Width;OUT
4.9;3;1.4;0.2;0
4.7;3.2;1.3;0.2;0
4.6;3.1;1.5;0.2;0
5;3.6;1.4;0.2;0 
)";

TEST(df, ctors) {
  DataFrame df(CSV1, true);

  // we test via assign() since it has already been tested
  DataFrame ref;
  vector<vector<string>> tokens;
  parseCSV(CSV1, tokens);
  ref.assign(tokens, true);

  EXPECT_EQ(df, ref);
}

TEST(df, basic) {
  vector<vector<string>> tokens;
  DataFrame df;
  // edge cases
  EXPECT_THROW(df.assign({}, true), runtime_error); // no rows
  EXPECT_THROW(df.assign({{}}, true), runtime_error); // no cols
  EXPECT_THROW(df.assign({ {"one", "two"}, {"one"}}, true), runtime_error); // different row sizes

  // assign
  parseCSV(CSV1, tokens);
  df.assign(tokens, true);

  EXPECT_EQ(df.nbrows(), 4);
  EXPECT_EQ(df.nbcols(), 5);

  vector<string> rownames = {"subject2", "subject3", "subject4", "subject5"};
  EXPECT_EQ(df.rownames(), rownames); 

  vector<string> colnames = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "OUT"};
  EXPECT_EQ(df.colnames(), colnames); 

  EXPECT_EQ(df[2][1], 1.3);
  EXPECT_EQ(df.at(1, 2), 1.3);
  EXPECT_EQ(df[0][0], 4.9);
  EXPECT_EQ(df.at(0, 0), 4.9);
  EXPECT_EQ(df[4][3],0);
  EXPECT_EQ(df.at(3, 4),0);
  cerr << df;

  // assign no rownames
  tokens.clear();
  parseCSV(CSV_NO_ROWNAMES, tokens);
  DataFrame df2;
  df2.assign(tokens, false);
  DataFrame ref(df);
  ref.rownames({});

  cerr << df2;
  EXPECT_NE(df2, df);
  EXPECT_EQ(df2, ref);


  // fetchRow
  vector<double> expected;
  expected = {4.9, 3, 1.4, 0.2, 0};
  vector<double> res = df.fetchRow(0);
  EXPECT_EQ(res, expected);
  expected = {5, 3.6, 1.4, 0.2, 0 };
  EXPECT_EQ(df.fetchRow(3), expected);

  // reset
  df.reset(1,2);
  EXPECT_EQ(df.nbrows(), 1);
  EXPECT_EQ(df.nbcols(), 2);

  cerr << df;
  // reset to 0
  df.reset();
  EXPECT_EQ(df.nbrows(), 0);
  EXPECT_EQ(df.nbcols(), 0);

  cerr << df;
}


TEST(df, subsetColumns) {
  using strings = vector<string>;

  DataFrame df(CSV1, true);

  // subset
  auto df2 = df.subsetColumns(2, 2);
  EXPECT_EQ(df2. nbrows(), df.nbrows());
  EXPECT_EQ(df2.nbcols(), 1);

  EXPECT_EQ(df2.rownames(), df.rownames()); 
  const auto& cn = df.colnames();
  EXPECT_EQ(df2.colnames(), strings(&cn[2], &cn[2] + 1)); 
  EXPECT_EQ(df2[0][1], 1.3);
  cerr << df2;

  // subset
  df2 = df.subsetColumns(2, 3);
  EXPECT_EQ(df2. nbrows(), df.nbrows());
  EXPECT_EQ(df2.nbcols(), 2);

  EXPECT_EQ(df2.rownames(), df.rownames()); 
  EXPECT_EQ(df2.colnames(), strings(&cn[2], &cn[3] + 1)); 
  EXPECT_EQ(df2[0][1], 1.3);
  cerr << df2;
}


TEST(df, subsetColumns_col_idx) {
  DataFrame df(CSV1, true);

  // subset
  vector<int> col_idx = {3, 1, 0};
  auto df2 = df.subsetColumns(col_idx);

  EXPECT_EQ(df2. nbrows(), df.nbrows());
  EXPECT_EQ(df2.nbcols(), 3);
  EXPECT_EQ(df2.rownames(), df.rownames()); 

  for (auto i = 0U; i < col_idx.size(); i++) {
    EXPECT_EQ(df2.colnames()[i], df.colnames()[col_idx[i]]);
    EXPECT_EQ(df2[i], df[col_idx[i]]);
  }

  EXPECT_THROW(df.subsetColumns({1000, 2000}), runtime_error);
}

TEST(df, subsetColumns_col_names) {
  DataFrame df(CSV1, true);

  // subset
  vector<string> col_names = {"Petal.Width", "Sepal.Length"};
  auto df2 = df.subsetColumns(col_names);
  cerr << df << df2;

  EXPECT_EQ(df2.colnames(), col_names);

  // bad colnames
  EXPECT_THROW(df.subsetColumns({"toto"}), runtime_error);

  // ============ no colnames in df ===========
  DataFrame df0(1, 1);
  EXPECT_THROW(df0.subsetColumns(col_names), runtime_error);
}


TEST(df, missingData) {
  string CSV_MISSING_DATA = 
R"(ID; VAR1; VAR2
id1;1;PI
id2;toto;3.2
)";

  DataFrame df(CSV_MISSING_DATA, true);
  
  EXPECT_DOUBLE_EQ(df.at(0, 0), 1);
  EXPECT_FALSE(df.missing(0, 0));

  EXPECT_DOUBLE_EQ(df.at(1, 0), MISSING_DATA_DOUBLE);
  EXPECT_TRUE(df.missing(1, 0));

  cerr << df;
}

TEST(df, set_reset) {
  DataFrame df;
  df.reset(2, 3);
  EXPECT_EQ(df.nbrows(), 2);
  EXPECT_EQ(df.nbcols(), 3);

  for (int row = 0; row < 2; row++)
    for (int col = 0; col < 3; col++)
      EXPECT_DOUBLE_EQ(df.at(row, col), 0);

  df.set(1, 2, 0.1);
  EXPECT_DOUBLE_EQ(df.at(1, 2), 0.1);

  df.set(1,2, MISSING_DATA_DOUBLE);
  EXPECT_TRUE(is_na(df.at(1, 2)));
  EXPECT_TRUE(df.missing(1, 2));
}


TEST(df, fillRowCol) {
  DataFrame df(CSV1, true);

  auto row = df.fetchRow(2);
  for (auto& e : row) e++;
  cerr << row << endl;
  df.fillRow(1, row);
  EXPECT_EQ(df.fetchRow(1), row);

  // fill Col
  auto col2 = df[2];
  for (auto& e : col2) e+= 10;
 
  df.fillCol(0, col2);
  EXPECT_EQ(df[0], col2);
}

TEST(df, colnames) {
  DataFrame df(CSV1, true);

  EXPECT_THROW(df.colnames({"only one"}), runtime_error);

  vector<string> cols = {"1", "2", "3", "4", "5"};
  df.colnames(cols);
  EXPECT_EQ(df.colnames(), cols);

  cerr << df;
}

TEST(df, rownames) {
  DataFrame df(CSV1, true);

  EXPECT_EQ(df.rownames(),  vector<string>({"subject2", "subject3", "subject4", "subject5"}));

  // set rownames
  df.rownames({});
  EXPECT_EQ(df.rownames(),  vector<string>());

  df.rownames(vector<string>({"1", "2", "3", "4"}));
  EXPECT_EQ(df.rownames(),  vector<string>({"1", "2", "3", "4"}));

  // bad rownames size
  EXPECT_THROW(df.rownames({"only one"}), runtime_error);
}

TEST(df, output) {
  // output with rownames
  DataFrame df(CSV1, true);

  EXPECT_THROW(df.colnames({"only one"}), runtime_error);

  vector<string> cols = {"1", "2", "3", "4", "5"};
  df.colnames(cols);
  EXPECT_EQ(df.colnames(), cols);

  cerr << df;

  // regression: no rownames
  vector<vector<string>> rows = { {"header"}, { "1.0" } };
  DataFrame df2(rows, false);
  cerr << df2;
}

TEST(df, load) {
  string tmp = poor_man_tmpnam("DF_load");
  ofstream out(tmp);
  out << CSV1;
  out.close();

  EXPECT_EQ(DataFrame::load(tmp, true), DataFrame(CSV1, true));
  EXPECT_EQ(DataFrame::load(tmp, false), DataFrame(CSV1, false));

  remove(tmp);
}