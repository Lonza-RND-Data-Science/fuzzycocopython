#include "fuzzy_system_metrics_computer.h"

using namespace fuzzy_coco;


FuzzySystemMetrics FuzzySystemMetricsComputer::computeForOneValue(double predicted, double actual, double threshold)
{
  // N.B: if no prediction no impact on the metrics
  assert(!is_na(predicted));
  FuzzySystemMetrics metrics;

  if (is_na(predicted) || is_na(actual)) {\
    return metrics;
  }

  const double err = error(predicted, actual);
  if (err != 0.0) { // probably useless?
    metrics.rrse = rrse(predicted, actual);
    metrics.rae = rae(predicted, actual);
    metrics.mse = mse(predicted, actual);
  }

  const bool predicted_positive = is_positive(predicted, threshold);
  const bool actual_positive = is_positive(actual, threshold);
  if (predicted_positive == actual_positive) { // true == well classified
    if (actual_positive) 
      metrics.true_positives = 1; 
    else 
      metrics.true_negatives = 1;

    // N.B: only set distance if well classified
    // N.B: iff dist > 0 predicted is above
    metrics.distanceThreshold = distanceToThreshold(predicted, actual, threshold);
    // N.B: distanceMinThreshold not modified
  } else { // false
    if (actual_positive) 
      metrics.false_negatives = 1;
    else 
      metrics.false_positives = 1;
  }

  return metrics;
}

// N.B: aggregate values for a single output variable
FuzzySystemMetrics FuzzySystemMetricsComputer::computeForOneVariable(const vector<double>& predicted, const vector<double>& actual, double threshold) 
{
  assert(predicted.size() == actual.size());
  FuzzySystemMetrics metrics;

  const int nb = predicted.size();

  double sumDistBelow = 0;
  double sumDistAbove = 0;
  double distMinBelow = INFINITY_DOUBLE;
  double distMinAbove = INFINITY_DOUBLE;

  int actual_nb = 0;
  for (int i = 0; i < nb; i++) {
      if (is_na(predicted[i]) || is_na(actual[i])) continue; // ignore prediced MISSING DATA?
      FuzzySystemMetrics m = computeForOneValue(predicted[i], actual[i], threshold);
      actual_nb++;
      metrics += m;

      const double dist = m.distanceThreshold;

      // N.B: can never be negative since only set when both predicted and actual have the same
      // threshold-sign
      assert(dist >= 0);

      sumDistAbove += distanceMin(dist);
      distMinAbove = min(distMinAbove, dist);

      // if (dist >= 0) { //above
      //   sumDistAbove += distanceMin(dist);
      //   distMinAbove = min(distMinAbove, dist);
      // } else {
      //   sumDistBelow += distanceMin(-dist);
      //   distMinBelow = min(distMinBelow, -dist);
      // }
  }
  if (actual_nb == 0) return metrics;

  int tp = metrics.true_positives, tn = metrics.true_negatives;
  int fp = metrics.false_positives, fn = metrics.false_negatives;
  metrics.sensitivity = sensitivity(tp, fn);
  metrics.specificity = specificity(tn, fp);
  metrics.accuracy = accuracy(tp, tn, fp, fn);
  metrics.ppv = ppv(tp, fp);

  metrics.mse = metrics.mse / actual_nb;
  metrics.rmse = sqrt(metrics.mse);
  metrics.rrse = sqrt(metrics.rrse / actual_nb);
  metrics.rae = metrics.rae / actual_nb;

  metrics.distanceThreshold = distanceToThresholdAggregate(sumDistAbove, sumDistBelow, tp, tn, fp, fn);
  if (distMinAbove == INFINITY_DOUBLE) distMinAbove = 0;
  if (distMinBelow == INFINITY_DOUBLE) distMinBelow = 0;
  metrics.distanceMinThreshold = (distMinAbove + distMinBelow) / 2;

  return metrics;
}

FuzzySystemMetrics FuzzySystemMetricsComputer::compute(const DataFrame& predicted, const DataFrame& actual, const vector<double>& thresholds)
{
  const size_t nb_vars = actual.nbcols();
  assert(predicted.nbrows() == actual.nbrows());
  assert(predicted.nbcols() == actual.nbcols());
  assert(thresholds.size() == nb_vars);

  FuzzySystemMetrics metrics;

  for (size_t var_idx = 0; var_idx < nb_vars; var_idx++) {
    auto m = computeForOneVariable(predicted[var_idx], actual[var_idx], thresholds[var_idx]);
  
    // double distanceMinThreshold = metrics.distanceMinThreshold;
    metrics += m;
    // KArl TODO: must decide on this
    // m.distanceMinThreshold += ''
  }

  // compute mean
  metrics.sensitivity /= nb_vars;
  metrics.specificity /= nb_vars;
  metrics.accuracy /= nb_vars;
  metrics.ppv /= nb_vars;
  metrics.rmse /= nb_vars;
  metrics.rrse /= nb_vars;
  metrics.rae /= nb_vars;
  metrics.mse /= nb_vars;
  metrics.distanceThreshold /= nb_vars;
  metrics.distanceMinThreshold /= nb_vars;

  return metrics;
}
