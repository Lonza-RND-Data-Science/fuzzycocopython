#ifndef COEVOLUTION_FITNESS_H
#define COEVOLUTION_FITNESS_H

#include <memory>
#include "evolution_engine.h"
#include "logging_logger.h"

namespace fuzzy_coco {

class CoevolutionFitnessMethod 
{
public:
  CoevolutionFitnessMethod() {}
  virtual ~CoevolutionFitnessMethod() {}

  double fitness(const Genome& left_genome, const Genome& right_genome) {
    double fit = fitnessImpl(left_genome, right_genome);

    if (fit > _best_fitness) {
      _best_fitness = fit;
      _best.first = left_genome;
      _best.second = right_genome;

      logging::logger() << "CoevolutionFitnessMethod::fitness(): " << "_best_fitness=" << _best_fitness  << endl;
    }
    return fit;
  }

  // best so far
  pair<Genome, Genome> getBest() const { return _best; }
  double getBestFitness() const { return _best_fitness; }
  // this is the main method to implement
  virtual double fitnessImpl(const Genome& left_genome, const Genome& right_genome) = 0;

  private:
    double _best_fitness = numeric_limits<double>::lowest();
    pair<Genome, Genome> _best;
};

class CoopCoevolutionFitnessMethod : public CoevolutionFitnessMethod {
public:
  CoopCoevolutionFitnessMethod() {}

  virtual double coopFitness(bool left, const Genome& genome, const Genomes& cooperators);
  // virtual vector<double> coopFitness(bool left, const Genomes& genomes, const Genomes& cooperators);
  // virtual vector<double> fitnesses(const Genome& genome1);
};
class CoopCoevolutionFitnessMethodAdaptor : public EvolutionFitnessMethod {
  public:
    CoopCoevolutionFitnessMethodAdaptor(bool left, CoopCoevolutionFitnessMethod& fit, const Genomes& cooperators) 
      : _left(left), _fit(fit), _cooperators(cooperators) {}
    double fitness(const Genome& genome) override { return _fit.coopFitness(_left, genome, _cooperators);}
  private:
    bool _left;
    CoopCoevolutionFitnessMethod& _fit;
    const Genomes& _cooperators;
};

}
#endif // COEVOLUTION_FITNESS_H
