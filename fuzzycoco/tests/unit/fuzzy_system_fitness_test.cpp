#include "tests.h"

#include "fuzzy_system_fitness.h"
#include "fuzzy_system_metrics_computer.h"

using namespace fuzzy_coco;

TEST(FuzzySystemFitness, fitness) {
  DataFrame REF(3, 1);
  REF.fillCol(0, {0, 50, 100});

  DataFrame df(3, 1);
  df.fillCol(0, {1, 47, 105});
  vector<double> thresholds = {50};

  FuzzySystemMetricsComputer cp;
  FuzzySystemFitness fit;
  auto metrics = cp.compute(df, REF, thresholds);
  cerr << metrics << endl;

  metrics = cp.compute(REF, REF, thresholds);
  double self_fitness = fit.fitness(metrics);

  double best_fitness = 0;
  vector<double> best;
  for (double i = 0; i < 150; i+=10) {
    for (double j = 0; j < 150; j++) {
      for (double k = 0; k < 150; k++) {
        df.fillCol(0, {i, j, k});
        metrics = cp.compute(df, REF, thresholds);
        double fitness = fit.fitness(metrics);
        if (fitness > best_fitness) {
          best_fitness = fitness;
          best = df[0];
        }
      }
    }
  }

  EXPECT_DOUBLE_EQ(best_fitness, self_fitness);
  EXPECT_EQ(best, REF[0]);

  // test extra_num and extra_denum
  metrics = cp.compute(REF, REF, thresholds);
  double fit0 = fit.fitness(metrics);
  EXPECT_EQ(fit.fitness(metrics, 0), fit0);
  EXPECT_EQ(fit.fitness(metrics, 0, 0), fit0);
  EXPECT_EQ(fit.fitness(metrics, 1, 2), fit0);
  EXPECT_EQ(fit.fitness(metrics, -1000, 2e7), fit0);
}

TEST(FuzzySystemFitness, weighted_fitness) {
  DataFrame REF(3, 1);
  REF.fillCol(0, {0, 50, 100});

  DataFrame df(3, 1);
  df.fillCol(0, {1, 47, 105});
  vector<double> thresholds = {50};

  FuzzySystemMetricsComputer cp;

  FuzzySystemMetrics weights;
  weights.sensitivity = 1;
  weights.specificity = 2;
  weights.accuracy = 0.5;
  weights.rmse = 1;
  weights.mse = 5;

  FuzzySystemWeightedFitness fit(weights);
  auto metrics = cp.compute(df, REF, thresholds);
  cerr << metrics << endl;

  metrics = cp.compute(REF, REF, thresholds);
  cerr << metrics;
  double self_fitness = fit.fitness(metrics);

  double best_fitness = 0;
  vector<double> best;
  for (double i = 0; i < 150; i+=10) {
    for (double j = 0; j < 150; j++) {
      for (double k = 0; k < 150; k++) {
        df.fillCol(0, {i, j, k});
        metrics = cp.compute(df, REF, thresholds);
        double fitness = fit.fitness(metrics);
        if (fitness > best_fitness) {
          best_fitness = fitness;
          best = df[0];
        }
      }
    }
  }

  EXPECT_DOUBLE_EQ(best_fitness, self_fitness);
  EXPECT_EQ(best, REF[0]);

  // test extra_num and extra_denum
  metrics = cp.compute(REF, REF, thresholds);
  double fit0 = fit.fitness(metrics);
  EXPECT_EQ(fit0, FuzzySystemWeightedFitness::weightedFitnessMetricsNumerator(weights, metrics) / FuzzySystemWeightedFitness::weightedFitnessMetricsDenominator(weights));

  EXPECT_EQ(fit.fitness(metrics, 0), fit0);
  EXPECT_EQ(fit.fitness(metrics, 0, 0), fit0);

  EXPECT_EQ(fit.fitness(metrics, 1, 2), (FuzzySystemWeightedFitness::weightedFitnessMetricsNumerator(weights, metrics) + 1) /
     (FuzzySystemWeightedFitness::weightedFitnessMetricsDenominator(weights) + 2));
  EXPECT_EQ(fit.fitness(metrics, -1000, 2e7), (FuzzySystemWeightedFitness::weightedFitnessMetricsNumerator(weights, metrics) - 1000) /
     (FuzzySystemWeightedFitness::weightedFitnessMetricsDenominator(weights) + 2e7));
}