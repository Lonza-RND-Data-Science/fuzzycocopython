#ifndef FUZZY_COCO_FITNESS_H
#define FUZZY_COCO_FITNESS_H

#include <memory>
#include "fuzzy_coco_params.h"
#include "fuzzy_coco_codec.h"
#include "fuzzy_system_metrics_computer.h"
#include "fuzzy_system_fitness.h"
#include "coevolution_engine.h"

namespace fuzzy_coco {

// a CoopCoevolutionFitnessMethod that knows how to evaluate a co-population (rules, MFs) using a given FuzzySystemFitness instance
// it implements the logic of how to evaluate genomes with cooperators, while the FuzzySystemFitness instance implements the actual
// fitness calculation
// TODO: modify the virtual interface, probably to fitnessImpl(fs)
class FuzzyCocoFitnessMethod : public CoopCoevolutionFitnessMethod {
public:
  // FuzzyCocoFitnessMethod(FuzzySystemFitness& fit_ptr,
  //   const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params);
  // N.B: use default FuzzySystemFitness instance
  FuzzyCocoFitnessMethod(FuzzySystem& fs, FuzzySystemFitness& fitter, const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params);

  static double computeFuzzySystemFitness(FuzzySystem& fs,FuzzySystemFitness& fitter, const FuzzySystemMetrics& metrics,
      double extra_num, double extra_denum);

public: // VIRTUAL INTERFACE
    
    // main method: compute the fitness for a (rules, vars) pair of genomes
  double fitnessImpl(const Genome& rules_genome, const Genome& vars_genome) override;
  // mostly for tests
  virtual string description() const { return "FuzzyCocoFitnessMethod"; }

public: 
  // compute the fitness of the current embedded fuzzy system
  virtual double fitnessImpl();

  void setRulesGenome(const Genome& rules_genome) {
    getFuzzyCocoCodec().setRulesGenome(getFuzzySystem(), rules_genome);
  }

  void setMFsGenome(const Genome& mfs_genome) {
    getFuzzyCocoCodec().setMFsGenome(getFuzzySystem(), mfs_genome);
  }

  // reset the internal fuzzy system to uses those rules and vars
  // return if the fuzzy system is consistent/ok
  bool resetFuzzySystem(const Genome& rules_genome, const Genome& vars_genome);

  FuzzySystemMetrics fitMetrics();
  FuzzySystemMetrics computeMetrics(const DataFrame& predicted, const DataFrame& actual); 
public:
  FuzzyCocoCodec& getFuzzyCocoCodec() { return _codec; }
  const FuzzyCocoCodec& getFuzzyCocoCodec() const { return _codec; }
  FuzzySystem& getFuzzySystem() { return _fuzzy_system; }
  const FuzzySystem& getFuzzySystem() const { return _fuzzy_system; }
  FuzzySystemFitness& getFuzzySystemFitness() { return _fitter; }

  FuzzySystemMetricsComputer& getMetricsComputer() { return _fsmc; }
  const vector<double>& getThresholds() const { return _thresholds; }

  const DataFrame& getInputData() const { return _actual_dfin; }

protected:
  const DataFrame& _actual_dfin;
  const DataFrame& _actual_dfout;

private:
  FuzzySystem& _fuzzy_system;
  FuzzyCocoCodec _codec;
  FuzzySystemMetricsComputer _fsmc;
  FuzzySystemFitness& _fitter;
  const vector<double>& _thresholds;
};


// a specialization of FuzzyCocoFitnessMethod that adds a Features Weights component to the standard FuzzyCocoFitnessMethod
// based on the weigths of features actually used in the FuzzySystem rules
 
class FuzzyCocoFeaturesWeightsFitnessMethod : public FuzzyCocoFitnessMethod {
public:
  FuzzyCocoFeaturesWeightsFitnessMethod(FuzzySystem& fs, FuzzySystemFitness& fitter,
       const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params);
  // FuzzyCocoFeaturesWeightsFitnessMethod(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params);
  // N.B: should add at one point the ctor that accepts a FuzzySystemFitness
public: // INTERFACE
  // main method: compute the fitness for a (rules, vars) pair of genomes
  double fitnessImpl(const Genome& rules_genome, const Genome& mfs_genome) override;
  string description() const override { return "FuzzyCocoFeaturesWeightsFitnessMethod"; }
public: // NON INTERFACE 
  double fitnessImpl() override;

  // compute the sum of used features weigths
  static double sumOfUsedFeaturesWeights(const vector<bool>& used_features, const vector<double>& weights);

  // check if there are some madatory features (whose weight >= 1) that are not used (missing)
  static bool mandatoryFeatureNotUsed(const vector<bool>& used_features, const vector<double>& weights);

  // accessor
  const vector<double>& getFeaturesWeights() const { return _features_weights; }
  double getSumOfFeaturesWeights() const { return _sum_of_weights; }

  // static double sumOfUsedFeaturesWeigths();

// protected:
//   void init(const map<string, double>& features_weights);

private:
    vector<double> _features_weights;
    double _sum_of_weights;
    vector<bool> _used_features_static; // reused to avoid reallocation
};

}
#endif // FUZZY_COCO_FITNESS_H
