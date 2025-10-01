#include "tests.h"
#include "discretizer.h"

using namespace fuzzy_coco;

TEST(Discretizer, basic) {
  Discretizer ds(2, 0, 1);

  EXPECT_DOUBLE_EQ(ds.getStep(), 1.0 / 3);


  EXPECT_EQ(ds.discretize(0), 0);

  EXPECT_EQ(ds.discretize(0.3), 1);
  EXPECT_EQ(ds.discretize(0.34), 1);

  EXPECT_EQ(ds.discretize(0.6), 2);
  EXPECT_EQ(ds.discretize(0.7), 2);

  EXPECT_EQ(ds.discretize(0.9), 3);
  EXPECT_EQ(ds.discretize(1), 3);
  EXPECT_EQ(ds.discretize(1.1), 3);

  EXPECT_EQ(ds.discretize(-1), -3);

  // undiscretize
  EXPECT_DOUBLE_EQ(ds.undiscretize(0), 0);
  EXPECT_DOUBLE_EQ(ds.undiscretize(1), 1.0/3);
  EXPECT_DOUBLE_EQ(ds.undiscretize(2), 2.0/3);
  EXPECT_DOUBLE_EQ(ds.undiscretize(3), 1);
  EXPECT_DOUBLE_EQ(ds.undiscretize(4), 4.0/3);
}

TEST(Discretizer, other) {
  Discretizer ds(4, -7, 24);
  cerr << ds;

  EXPECT_DOUBLE_EQ(ds.getStep(), 31.0 / 15);

  EXPECT_EQ(ds.discretize(0), 3);

  EXPECT_EQ(ds.discretize(-7), 0);
  EXPECT_EQ(ds.discretize(15), 11);
  EXPECT_EQ(ds.discretize(24), 15);
  EXPECT_EQ(ds.discretize(26), 16);


  // undiscretize
  double step = 31.0 / 15;
  EXPECT_DOUBLE_EQ(ds.undiscretize(0), -7);
  EXPECT_DOUBLE_EQ(ds.undiscretize(1), -7 + step);
  EXPECT_DOUBLE_EQ(ds.undiscretize(2), -7 + 2*step);
  EXPECT_DOUBLE_EQ(ds.undiscretize(3), -7 + 3*step);
  EXPECT_DOUBLE_EQ(ds.undiscretize(15), 24);

  for (int i = 0; i < 16; i++)
    EXPECT_EQ(ds.discretize(ds.undiscretize(i)), i);
}

TEST(Discretizer, ctorFromData) {
  // normal case
  {
    vector<double> data = {4, 0, 3, 9, 15, 3};
    Discretizer ds(4, data);
    EXPECT_EQ(ds, Discretizer(4, 0, 15));
    EXPECT_DOUBLE_EQ(ds.getStep(), 1);
  }
  
  // with missing data
  {
    vector<double> data = {4, 0, MISSING_DATA_DOUBLE, 9, 15, MISSING_DATA_DOUBLE};
    Discretizer ds(4, data);
    EXPECT_EQ(ds, Discretizer(4, 0, 15));
    EXPECT_DOUBLE_EQ(ds.getStep(), 1);
  }

  // all missing
  {
    vector<double> data = {MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE};
    EXPECT_THROW(Discretizer ds(4, data), runtime_error);
    EXPECT_THROW(Discretizer ds(4, { MISSING_DATA_DOUBLE }), runtime_error);
  }

  // empty data
  {
    vector<double> data = {};
    EXPECT_THROW(Discretizer ds(4, data), runtime_error);
  }

}