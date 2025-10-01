/**
  * @file   fuzzy_system_metrics_computer.h
  * @author Karl Forner <karl.forner@gmail.com>
  * @author Lonza
  * @date   09.2024
  * @class FuzzySystemMetricsComputer
  * @brief a base class to compute FuzzySystemMetrics
  */

#ifndef FUZZY_SYSTEM_METRICS_COMPUTER_H
#define FUZZY_SYSTEM_METRICS_COMPUTER_H

#include <cassert>
#include <cmath>
#include "types.h"
#include "fuzzy_system_metrics.h"
#include "dataframe.h"

namespace fuzzy_coco {

class FuzzySystemMetricsComputer {
public:
  FuzzySystemMetricsComputer() {}
  virtual ~FuzzySystemMetricsComputer() {}

  // main user-level method
  virtual FuzzySystemMetrics compute(const DataFrame& predicted, const DataFrame& actual, const vector<double>& thresholds);

  // increments metrics. N.B: metrics will NOT be reset, so the results depend on the metrics initial content
  // N.B : some metrics (such as rmse) must be adjusted for the number of evaluations. Here there are NOT.
  virtual FuzzySystemMetrics computeForOneValue(double predicted, double actual, double threshold);

  // N.B: metrics will be reset first. Then compute(double, double) will be called on each pair of values, then values are ajusted
  // N.B: those are values for a give Output Variable
  virtual FuzzySystemMetrics computeForOneVariable(const vector<double>& predicted, const vector<double>& actual, double threshold);



  // ====== static methods and constants ===========
  static constexpr double EPSILON = 1e-9;

  // fix a denum by adding an EPSILON to avoid division by zero
  static double fix_denum(double denum) {
    if (fabs(denum) < EPSILON) // too close to zero
      denum += EPSILON;
    return denum;
  }

  static double error(double predicted, double actual) { return predicted - actual; }

  static double average_error(double predicted, double actual) {
    return fabs((predicted + actual) / 2.0);
  }

  static double relative_error(double predicted, double actual) {
    return error(predicted, actual) / fix_denum(average_error(predicted, actual));
  }

  static double rrse(double predicted, double actual) {
    const double re = relative_error(predicted, actual);
    return re * re;
  }

  static double rae(double predicted, double actual) {
    return fabs(relative_error(predicted, actual));
  }

  static double mse(double predicted, double actual) {
    const double err = error(predicted, actual);
    return err * err;
  }

  static double distanceToThreshold(double predicted, double actual, double threshold) {
    return (predicted - threshold) / fix_denum(actual - threshold);
  }

  static double distanceToThresholdAggregate(double sumDistAbove, double sumDistBelow, int tp, int tn, int fp, int fn) {
    double denum_below = tn + fp;
    if (denum_below == 0) return MISSING_DATA_DOUBLE;
    double denum_above = tp + fn;
    if (denum_above == 0) return MISSING_DATA_DOUBLE;

    return ( (sumDistBelow / denum_below) + (sumDistAbove / denum_above) ) / 2.0;
  }

  static bool is_positive(double value,  double threshold) {
    assert(!is_na(value));
    return value >= threshold;
  }

// #define MAX_ADM 0.71428
  // todo: check what it is exactly
  static double distanceMin(double dist) {
    if (dist >= 0.71428) return 1;
    return dist * ( 2.8 - ( 1.96 * dist ) );
  }

  static double sensitivity(int tp, int fn) {
    const int denum = tp + fn;
    if (denum == 0) return MISSING_DATA_DOUBLE;
    return double(tp) / denum;
  }

  static double specificity(int tn, int fp) {
    const int denum = tn + fp;
    if (denum == 0) return MISSING_DATA_DOUBLE;
    return double(tn) / denum;
  }

  static double accuracy(int tp, int tn, int fp, int fn) {
    const int denum = tp + tn + fp + fn;
    if (denum == 0) return MISSING_DATA_DOUBLE;
    return double(tp + tn) / denum;
  }

  static double ppv(int tp, int fp) {
    const int denum = tp + fp;
    if (denum == 0) return MISSING_DATA_DOUBLE;
    return double(tp) / denum;
  }

  // some metrics 

};

}
#endif // FUZZY_SYSTEM_METRICS_COMPUTER_H
