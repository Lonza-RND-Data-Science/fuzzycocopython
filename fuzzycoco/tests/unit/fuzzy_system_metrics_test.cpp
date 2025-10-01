#include "tests.h"

#include "fuzzy_system_metrics_computer.h"

using namespace fuzzy_coco;

TEST(FuzzySystemMetricsComputer, computeForOneValue) {
  FuzzySystemMetricsComputer cp;

  auto sum_posneg = [](const FuzzySystemMetrics& m) {
    return m.true_positives + m.true_negatives + m.false_positives + m.false_negatives;
  };
 
  FuzzySystemMetrics metrics;

  // equal and negative
  metrics = cp.computeForOneValue(0, 0, 50);
    cerr << metrics << endl;
  EXPECT_EQ(sum_posneg(metrics), 1);
  EXPECT_EQ(metrics.true_negatives, 1);
  EXPECT_DOUBLE_EQ(metrics.mse, 0);
  EXPECT_DOUBLE_EQ(metrics.distanceThreshold, 1);

  // equal and at the threshold
  metrics = cp.computeForOneValue(50, 50, 50);
  cerr << metrics << endl;
  EXPECT_EQ(sum_posneg(metrics), 1);
  EXPECT_EQ(metrics.true_positives, 1);
  EXPECT_DOUBLE_EQ(metrics.mse, 0);
  EXPECT_DOUBLE_EQ(metrics.distanceThreshold, 0);

  // close and positive
  metrics = cp.computeForOneValue(90, 100, 50);
  cerr << metrics << endl;
  EXPECT_EQ(sum_posneg(metrics), 1);
  EXPECT_EQ(metrics.true_positives, 1);
  EXPECT_DOUBLE_EQ(metrics.distanceThreshold, 4.0/5);
  EXPECT_DOUBLE_EQ(metrics.mse, 10.0*10);

  // false negative
  metrics = cp.computeForOneValue(10, 80, 50);
  cerr << metrics << endl;
  EXPECT_EQ(sum_posneg(metrics), 1);
  EXPECT_EQ(metrics.false_negatives, 1);
  EXPECT_DOUBLE_EQ(metrics.distanceThreshold, 0.0); // N.B: 0 because negative
  EXPECT_DOUBLE_EQ(metrics.mse, 70.0*70);

  // false positive
  metrics = cp.computeForOneValue(80, 10, 50);
  cerr << metrics << endl;
  EXPECT_EQ(sum_posneg(metrics), 1);
  EXPECT_EQ(metrics.false_positives, 1);
  EXPECT_DOUBLE_EQ(metrics.distanceThreshold, 0.0);
  EXPECT_DOUBLE_EQ(metrics.mse, 70.0*70);

  // NA
  {
    auto m1 = cp.computeForOneValue(0, MISSING_DATA_DOUBLE, 0);
    EXPECT_EQ(m1, FuzzySystemMetrics());
  }

}

TEST(FuzzySystemMetricsComputer, computeForOneVariable) {
  FuzzySystemMetricsComputer cp;

  vector<double> predicted  = {0,   90,   10,   80};
  vector<double> actual     = {0 ,  100,  80,   10};

  auto metrics = cp.computeForOneVariable(predicted, actual, 50);

  EXPECT_EQ(metrics.true_negatives, 1);
  EXPECT_EQ(metrics.true_positives, 1);
  EXPECT_EQ(metrics.false_negatives, 1);
  EXPECT_EQ(metrics.false_positives, 1);

  EXPECT_DOUBLE_EQ(metrics.mse, (0.0 + 10*10 + 70*70 + 70*70) / 4);
  EXPECT_DOUBLE_EQ(metrics.rmse, sqrt( (0.0 + 10*10 + 70*70 + 70*70) / 4));

  // all missing: should not happen
  predicted  = {MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE, MISSING_DATA_DOUBLE};
  metrics = cp.computeForOneVariable(predicted, actual, 50);
  // unchanged, only default values
  EXPECT_EQ(metrics, FuzzySystemMetrics());

  // === regression? ==========
  actual = {0, 50, 100};
  metrics = cp.computeForOneVariable(actual, actual, 50);
  EXPECT_DOUBLE_EQ(metrics.sensitivity, 1);
  EXPECT_DOUBLE_EQ(metrics.specificity, 1);
  EXPECT_DOUBLE_EQ(metrics.accuracy, 1);
}

TEST(FuzzySystemMetricsComputer, compute) {
  FuzzySystemMetricsComputer cp;
  DataFrame actual(4, 4), predicted(4, 4);
  actual.fillCol(0, {10, 20 , 30, 40});
  predicted.fillCol(0, {10, 20 , 30, 40}); // perfect fit

  actual.fillCol(1, {0, 0, 0, 0});
  predicted.fillCol(1, {1, -1, 3, -3});

  actual.fillCol(2, {MISSING_DATA_DOUBLE, 0, 100, MISSING_DATA_DOUBLE});
  predicted.fillCol(2, {10, 20 , 30, 40});

  actual.fillCol(3, {10, 1, 100, 0});
  predicted.fillCol(3, {9, 0 , 98, -1}); // close
  vector<double> thresholds = {20, 0, 50, 50};
  cerr << actual << predicted;

  auto m = cp.compute(predicted, actual, thresholds);
  cerr << m;

  EXPECT_EQ(m.true_positives, 3 + 0 + 2 + 1);
  EXPECT_EQ(m.true_negatives, 1 + 0 + 1 + 3);
  EXPECT_EQ(m.false_positives, 0 + 0 + 0 + 0);
  EXPECT_EQ(m.false_negatives, 0 + 2 + 1 + 0);

  // =========== on first column =============

  {
    int colidx = 0;
    auto ref = actual.subsetColumns(colidx, colidx);
    auto col = predicted.subsetColumns(colidx, colidx);
    auto thr = vector<double>(thresholds.begin() + colidx, thresholds.begin() + colidx + 1);
    auto m = cp.compute(col, ref, thr);

    // cerr << ref << col << m.describe();

    EXPECT_EQ(m.sensitivity, 1);
    EXPECT_EQ(m.specificity, 1);
    EXPECT_EQ(m.accuracy, 1);
    EXPECT_EQ(m.ppv, 1);
    EXPECT_EQ(m.rmse, 0);
    EXPECT_EQ(m.rrse, 0);
    EXPECT_EQ(m.rae, 0);
    EXPECT_EQ(m.mse, 0);
    // EXPECT_EQ(m.distanceThreshold, 0.5); // TODO: check that
  }

  {
    int colidx = 1;
  // =========== on 2nd column (no NAs) =============
    auto ref = actual.subsetColumns(colidx, colidx);
    auto col = predicted.subsetColumns(colidx, colidx);
    auto thr = vector<double>(thresholds.begin() + colidx, thresholds.begin() + colidx + 1);
    auto m = cp.compute(col, ref, thr);

  // cerr << ref << col << m.describe();

    EXPECT_EQ(m.sensitivity, 0.5);
    EXPECT_EQ(m.specificity, 0);
    EXPECT_EQ(m.accuracy, 0.5);
    EXPECT_EQ(m.mse, (1 + 1 + 3*3 + 3*3) / 4.0);
    EXPECT_EQ(m.rmse, sqrt((1 + 1 + 3*3 + 3*3) / 4.0));
    EXPECT_EQ(m.rrse, 2.0);
    EXPECT_EQ(m.rae, 2.0);
  }

  {
    int colidx = 2;
  // =========== on 2nd column (no NAs) =============
    auto ref = actual.subsetColumns(colidx, colidx);
    auto col = predicted.subsetColumns(colidx, colidx);
    auto thr = vector<double>(thresholds.begin() + colidx, thresholds.begin() + colidx + 1);
    auto m = cp.compute(col, ref, thr);

  cerr << ref << col << m.describe();

    EXPECT_EQ(m.true_positives, 0);
    EXPECT_EQ(m.true_negatives, 1);
    EXPECT_EQ(m.false_positives, 0);
    EXPECT_EQ(m.false_negatives, 1);
    EXPECT_EQ(m.sensitivity, 0);
    EXPECT_EQ(m.specificity, 1);
    EXPECT_EQ(m.accuracy, 0.5);
    EXPECT_NEAR (m.mse, (20*20 + 70*70) / 2.0, 1e-4);
    EXPECT_NEAR (m.rmse, sqrt((20*20 + 70*70) / 2.0), 1e-4);
    EXPECT_NEAR(m.rrse, 1.6062, 1e-4);
    EXPECT_NEAR(m.rae, 1.53846, 1e-4);
  }
}

TEST(FuzzySystemMetrics, describe) {
  FuzzySystemMetrics metrics;

  metrics.sensitivity = 1;
  metrics.specificity = 0.8;

  auto desc = metrics.describe();

  EXPECT_DOUBLE_EQ(desc.get_double("sensitivity"), 1);
  EXPECT_DOUBLE_EQ(desc.get_double("specificity"), 0.8);

  cerr << desc;

  FuzzySystemMetrics metrics2(desc);
  EXPECT_EQ(metrics2, metrics);

  // load with defaults
  NamedList quasi_empty;
  quasi_empty.add("rmse", 100.05);
  FuzzySystemMetrics m3(quasi_empty);
  cerr << m3;
}

TEST(FuzzySystemMetrics, setValues) {
  FuzzySystemMetrics metrics;

  metrics.sensitivity = 1;
  metrics.specificity = 0.8;

  NamedList values;
  values.add("specificity", 0.6);
  values.add("rae", -1.0);

  metrics.setValues(values);

  FuzzySystemMetrics ref;
  ref.sensitivity = 1;
  ref.specificity = 0.6;
  ref.rae = -1;

  EXPECT_EQ(metrics, ref);
}