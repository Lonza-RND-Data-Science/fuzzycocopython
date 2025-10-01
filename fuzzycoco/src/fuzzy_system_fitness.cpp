#include <cmath>
#include "fuzzy_system_fitness.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace logging;


double FuzzySystemFitness::fitness(const FuzzySystemMetrics& metrics, double extra_num , double extra_denum) {
  double fit = 
    metrics.sensitivity
    + metrics.specificity
    + metrics.accuracy
    + metrics.ppv
    + pow(2, -metrics.rmse )
    + pow(2, -metrics.rrse )
    + pow(2, -metrics.rae )
    + pow(2, -metrics.mse )
    + 1.0 / max(1.0, metrics.nb_vars);

  return fit;
}

double FuzzySystemWeightedFitness::weightedFitnessMetricsNumerator(
  const FuzzySystemMetrics& weights, const FuzzySystemMetrics& metrics)
{
  double num = 
    weights.sensitivity * metrics.sensitivity
    + weights.specificity * metrics.specificity
    + weights.accuracy * + metrics.accuracy
    + weights.ppv * metrics.ppv
    + weights.rmse * pow(2, -metrics.rmse )
    + weights.rrse * pow(2, -metrics.rrse )
    + weights.rae * pow(2, -metrics.rae )
    + weights.mse * pow(2, -metrics.mse )
    + weights.nb_vars * (1.0  / max(1.0, metrics.nb_vars));

  return num;
}

double FuzzySystemWeightedFitness::weightedFitnessMetricsDenominator(const FuzzySystemMetrics& weights)
{
  double denum = 
    weights.sensitivity
    + weights.specificity
    + weights.accuracy
    + weights.ppv
    + weights.rmse
    + weights.rrse
    + weights.rae
    + weights.mse
    + weights.nb_vars;

  return denum;
}

double FuzzySystemWeightedFitness::fitness(const FuzzySystemMetrics& metrics, double extra_num , double extra_denum) {
  double num = weightedFitnessMetricsNumerator(_weights, metrics);
  double denum = weightedFitnessMetricsDenominator(_weights);
  
  double fit = (num + extra_num ) / (denum + extra_denum);

  return fit;
}
