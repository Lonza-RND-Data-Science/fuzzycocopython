/**
  * @file   fuzzy_system_metrics.h
  * @author Karl Forner <karl.forner@gmail.com>
  * @author Lonza
  * @date   09.2024
  * @struct FuzzySystemMetrics
  * @brief a structure that contains a FuzzySystem Performance metrics
  */

#ifndef FUZZY_SYSTEM_METRICS_H
#define FUZZY_SYSTEM_METRICS_H

#include <string>
#include <iostream>
#include <vector>
#include "types.h"
#include "named_list.h"

namespace fuzzy_coco {

struct FuzzySystemMetrics
{
  // N.B: there are all double so that they can be also used as weights, e.g. in FuzzySystemWeightedFitness

  double sensitivity;
  double specificity;
  double accuracy;
  double ppv;
  double rmse;
  double rrse;
  double rae;
  double mse;
  double distanceThreshold;
  double distanceMinThreshold;
  double nb_vars; // the number of variables used in the system (can be used to penalize huge systems)
  double overLearn;
  double true_positives;
  double false_positives;
  double true_negatives;
  double false_negatives;

  FuzzySystemMetrics() { reset(); }
  FuzzySystemMetrics(const NamedList& desc);
  void reset();
  void setValues(const NamedList& desc);
  bool operator==(const FuzzySystemMetrics& p) const;
  void operator+=(const FuzzySystemMetrics& m);

  NamedList describe() const;

  friend ostream& operator<<(ostream& out, const FuzzySystemMetrics& p);
};

}
#endif // FUZZY_SYSTEM_METRICS_H
