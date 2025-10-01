/**
  * @file   fuzzy_system_fitness.h
  * @author Karl Forner <karl.forner@gmail.com>
  * @author Lonza
  * @date   09.2024
  * @class FuzzySystemFitness
  *
  * @brief an asbtract base class for computing the fitness of a Fuzzy System
  */

#ifndef FUZZY_SYSTEM_FITNESS_H
#define FUZZY_SYSTEM_FITNESS_H

#include "fuzzy_system_metrics.h"

namespace fuzzy_coco {

// implement the computation of the fitness of a FuzzySystem based on the FuzzySystemMetrics
class FuzzySystemFitness
{
public:
  FuzzySystemFitness() {}
  virtual ~FuzzySystemFitness() {}

  // INTERFACE
  // N.B: extra_num and extra_denum are actually only used in FuzzySystemWeightedFitness::fitness
  // here it's a NOOP
  virtual double fitness(const FuzzySystemMetrics& metrics, double extra_num = 0, double extra_denum = 0);
};

// a weighted FuzzySystem fitness. weigths are given a priori and are fixed 
class FuzzySystemWeightedFitness : public FuzzySystemFitness {
public:
  FuzzySystemWeightedFitness(const FuzzySystemMetrics& weights) : _weights(weights) {}

  // INTERFACE IMPLEMENTATION
  double fitness(const FuzzySystemMetrics& metrics, double extra_num = 0, double extra_denum = 0) override;

public: // not part of the FuzzySystemFitness interface
  static double weightedFitnessMetricsNumerator(const FuzzySystemMetrics& weights, const FuzzySystemMetrics& metrics);
  static double weightedFitnessMetricsDenominator(const FuzzySystemMetrics& weights);

private:
  FuzzySystemMetrics _weights;
};

}
#endif // FUZZY_SYSTEM_FITNESS_H
