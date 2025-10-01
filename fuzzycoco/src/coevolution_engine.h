#ifndef COEVOLUTIONENGINE_H
#define COEVOLUTIONENGINE_H

#include <memory>
#include "evolution_engine.h"
#include "coevolution_fitness.h"

namespace fuzzy_coco {

// hold the results of one Generation (~ iteration of evolution) of combined populations
struct CoevGeneration {
  CoevGeneration() {}
  CoevGeneration(const Generation& left_gen, const Generation& right_gen) 
  : left_gen(left_gen), right_gen(right_gen) {}
  // CoevGeneration(const Genomes& left_genos, const Genomes& right_genos) 
  //   : left_gen(left_genos), right_gen(right_genos) {}

  Generation left_gen;
  Generation right_gen;
  double fitness = 0;
  int generation_number = 0;
};

inline ostream& operator<<(ostream& out, const CoevGeneration& gen) {
  out 
    << "left generation: " << gen.left_gen
    << "right generation: " << gen.right_gen
    << "fitness=" << gen.fitness << " at generation " << gen.generation_number << endl;
  return out;
}

class CoevolutionEngine 
{
public:
  // init the engine with some params, nothing else
  // need to check that nb_cooperators <= elite_size
  CoevolutionEngine(CoopCoevolutionFitnessMethod& fit, 
    EvolutionEngine& left_engine, 
    EvolutionEngine& right_engine, 
    int nb_cooperators);

public: // main interface
  // main function
  pair<CoevGeneration, vector<double>>  evolve(const Genomes& left_genos, const Genomes& right_genos, int nb_generations, double maxFit);

  CoevGeneration start(const Genomes& left_genos, const Genomes& right_genos);

  void updateGeneration(CoevGeneration& cogen);
  CoevGeneration nextGeneration(const CoevGeneration& cogen);

  // Generation popNextGeneration(const Genomes& genos, EvolutionFitnessMethod& fit);

  static Genomes selectCooperators(const Genomes& elite, int nb_cooperators);

  // N.B: do not depend upon the Generation: return the best pair seen so far
  pair<Genome, Genome> getBest() const { return getFitnessMethod().getBest(); }

public: // accessors
  const CoopCoevolutionFitnessMethod& getFitnessMethod() const { return _fit; }
  CoopCoevolutionFitnessMethod& getFitnessMethod() { return _fit; }
  EvolutionEngine& getLeftEngine() { return _left_engine; }
  EvolutionEngine& getRightEngine() { return _right_engine; }
  int getNbCooperators() const { return _nb_cooperators; }

private:
  CoopCoevolutionFitnessMethod& _fit;
  EvolutionEngine& _left_engine;
  EvolutionEngine& _right_engine; 
  int _nb_cooperators;
};

}
#endif // COEVOLUTIONENGINE_H
