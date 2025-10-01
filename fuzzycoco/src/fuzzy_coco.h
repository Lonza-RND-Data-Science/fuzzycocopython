#ifndef FUZZY_COCO_H
#define FUZZY_COCO_H

#include "fuzzy_coco_engine.h"

namespace fuzzy_coco {

// implements how to use the FuzzyCocoEngine: which fitness to use, how to save and reload the results, how to predict output, how to evaluate the fitne
// ss of a dataset
// this is the high-level interface to use Fuzzycoco
class FuzzyCoco 
{
public:
  // ctor that selects the appropriate fitness methods based on params and defaults
  FuzzyCoco(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params, RandomGenerator& rng);
  FuzzyCoco(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params, const FuzzySystem& fs, RandomGenerator& rng);

  virtual ~FuzzyCoco() {}

  // ================ main static interface ======================
  static NamedList searchBestFuzzySystem(const DataFrame& df, int nb_out_vars, const FuzzyCocoParams& params, int seed);
  // static void searchBestFuzzySystemAndSave(const DataFrame& df, int nb_out_vars, const FuzzyCocoParams& params, int seed, const string& filename);

  DataFrame predict(const DataFrame& dfin) { return getFuzzySystem().smartPredict(dfin); }

  static DataFrame loadAndPredict(const DataFrame& df, const NamedList& saved);
  static DataFrame loadAndPredict(const DataFrame& df, const string& fuzzy_file);

  static void evalAndSave(const DataFrame& df, const string& fuzzy_file, ostream& out = cout);
  static NamedList eval(const DataFrame& df, const FuzzySystem& fs, const FuzzyCocoParams& params);
  // // highest level function, Runs everything using the params

  CoevGeneration start(RandomGenerator& rng, bool influence, double evolving_ratio);
  CoevGeneration run(int nb, double max_fit, RandomGenerator& rng, bool influence = false, double evolving_ratio = 0.8);
  CoevGeneration run(int nb, double max_fit, bool influence = false, double evolving_ratio = 0.8);
  CoevGeneration run() { 
    const auto& p = getParams().global_params; 
    return run(p.max_generations, p.max_fitness, p.influence_rules_initial_population, p.influence_evolving_ratio); 
  }
  CoevGeneration run(int nb, double max_fit, CoevGeneration& from_gen);

  NamedList describeBestFuzzySystem() { return getEngine().describeBestFuzzySystem(); }

  NamedList describe(int nb_generations = MISSING_DATA_INT);
  static NamedList describeFit(FuzzyCocoFitnessMethod& fitter, int nb_generations = MISSING_DATA_INT);

  static FuzzyCoco load(const NamedList desc);

  static NamedList loadFuzzyFile(const string& fuzzy_file);

  // set the shared Fuzzy System with the rules and MFs that achieve the best fitness computed so far
  void selectBestFuzzySystem() { getEngine().rebuildBestFuzzySystem(); }

  double getFitness() { return getFitnessMethod().fitnessImpl(); }

  // accessors
  // get the current Fuzzy System used to compute the fitness. It is used to compute the fitness for all
  // populations so may be in a random state
  FuzzySystem& getFuzzySystem() { return _fuzzy_system; }
  const FuzzySystem& getFuzzySystem() const { return _fuzzy_system; }
  FuzzyCocoFitnessMethod& getFitnessMethod() { return *_fitter_ptr; }
  const FuzzyCocoFitnessMethod& getFitnessMethod() const { return *_fitter_ptr; }

  FuzzySystemFitness& getFuzzySystemFitness() { return *_fuzzy_system_fitter_ptr; }
  FuzzyCocoEngine& getEngine() { return _engine; }
  const FuzzyCocoParams& getParams() const { return _params; }

public:
  friend ostream& operator<<(ostream& out, const FuzzyCoco& ds);

  static void split_dataset(const DataFrame& df, int nb_out_vars, DataFrame& dfin, DataFrame& dfout);
  static void influence_rules_genomes(Genomes& rules_genomes, const vector<double>& features_weights, 
      FuzzyCocoCodec& codec, RandomGenerator& rng, double evolving_ratio = 0.8);

public:
  static unique_ptr<FuzzySystemFitness> selectFuzzySystemFitness(const FuzzyCocoParams& params);

  static unique_ptr<FuzzyCocoFitnessMethod> 
  selectFitnessMethods(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params, FuzzySystem& fs, FuzzySystemFitness& fs_fitter);

private:
  FuzzyCocoParams _params;
  FuzzySystem _fuzzy_system;

  // N.B: only used to construct the FuzzyCocoFitnessMethod _fitter_ptr
  unique_ptr<FuzzySystemFitness> _fuzzy_system_fitter_ptr;

  unique_ptr<FuzzyCocoFitnessMethod> _fitter_ptr;
  FuzzyCocoEngine _engine;
};

}
#endif // FUZZY_COCO_H
