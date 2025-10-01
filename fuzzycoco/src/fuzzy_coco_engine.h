#ifndef FUZZY_COCO_ENGINE_H
#define FUZZY_COCO_ENGINE_H

#include "fuzzy_coco_fitness.h"
#include "coevolution_engine.h"

namespace fuzzy_coco {

// implements the FuzzyCoco Algorithm: co-evolve a population of rules and Membership Functions, and use 
// a FuzzySystem together with a FuzzyCocoFitnessMethod to evaluate the fitness
class FuzzyCocoEngine 
{
public:
  // ctor that uses a given FuzzyCocoFitnessMethod
  FuzzyCocoEngine(const DataFrame& dfin, const DataFrame& dfout, FuzzyCocoFitnessMethod& fitter,
      const FuzzyCocoParams& params, RandomGenerator& rng);
  virtual ~FuzzyCocoEngine() {}

  // ================ main interface ======================
  
  // highest level function, Runs everything using the params
  CoevGeneration run(CoevGeneration& gen, int nb, double max_fit);
  CoevGeneration run();

  Genomes buildRulesGenomes(int nb_pop_rules);
  Genomes buildMFsGenomes(int nb_pop_mfs);

  CoevGeneration start() { return start(getParams().rules_params.pop_size, getParams().mfs_params.pop_size); }
  CoevGeneration start(int nb_pop_rules, int nb_pop_mfs);
  CoevGeneration start(const Genomes& rules, const Genomes& mfs);
  void update(CoevGeneration& cogen);

  CoevGeneration next(const CoevGeneration& cogen);
  pair<Genome, Genome> getBest() const { return getCoevolutionEngine().getBest(); }

  NamedList describeBestFuzzySystem();
  FuzzySystem& rebuildBestFuzzySystem();
  // accessors
  
  FuzzyCocoCodec& getFuzzyCocoCodec() { return getFitnessMethod().getFuzzyCocoCodec(); }
  const FuzzyCocoCodec& getFuzzyCocoCodec() const { return getFitnessMethod().getFuzzyCocoCodec(); }

  FuzzySystem& getFuzzySystem() { return getFitnessMethod().getFuzzySystem(); }
  const FuzzySystem& getFuzzySystem() const { return getFitnessMethod().getFuzzySystem(); }

  // this is a bit ugly: we know the FitnessMethod owned by the CoevolutionEngine as a CoopCoevolutionFitnessMethod
  // is actually an instance of CoopCoevolutionFitnessMethod (cf ctor)
  FuzzyCocoFitnessMethod& getFitnessMethod() { return _fitter; }
  const FuzzyCocoFitnessMethod& getFitnessMethod() const { return _fitter; }

  CoevolutionEngine& getCoevolutionEngine() { return _coev_engine; }
  const CoevolutionEngine& getCoevolutionEngine() const {  return _coev_engine;  }

  

  RandomGenerator& getRng() { return _rng; }
  const FuzzyCocoParams& getParams() const { return _params; }
  static vector<Discretizer> createDiscretizersForData(const DataFrame& df, int nb_bits);
public:
    friend ostream& operator<<(ostream& out, const FuzzyCocoEngine& ds);

private:
    FuzzyCocoParams _params;
    RandomGenerator& _rng;

    FuzzyCocoFitnessMethod& _fitter;
    CoevolutionEngine _coev_engine;

    // just used to build _coev_engine, never accessed directly
    EvolutionEngine _rules_engine;
    EvolutionEngine _mfs_engine;
};

}
#endif // FUZZY_COCO_ENGINE_H
