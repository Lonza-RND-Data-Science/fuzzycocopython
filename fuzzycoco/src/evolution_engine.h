// modified from evolutionengine.h to remove the multithreading and the coupling with higher level classes
//  *     such as ComputeThread/FuzzyCoco. Actually implement an Evolution Base class

#ifndef EVOLUTIONENGINE_H
#define EVOLUTIONENGINE_H


#include "crossover_method.h"
#include "mutation_method.h"
#include "selection_method.h"
#include "evolution_params.h"

namespace fuzzy_coco {

// Evolve a next Generation of genomes:
// - a subgroup (Elite) is selected and passed over to the next generation
// - a subgroup is selected (evolvers) for mutation and crossover --> evolved
// the next generation is the group formed by those two subgroups: the elite + the evolved
// N.B: evolvers and elite can have an intersection

struct GenerationFitness {
    GenerationFitness(int nb) : fitnesses(nb) {}
    vector<double> fitnesses;
    double fitness = 0;
};

struct Generation {
  Generation(int elite_size = 0) : _elite_size(elite_size) {}
  Generation(const Genomes &individuals, int elite_size) 
    : individuals(individuals), _elite_size(elite_size) ,  fitnesses(individuals.size())
  {
    selectElite();
  }

  void selectElite()
  {
    assert((size_t)_elite_size <= individuals.size());
    elite.clear();
    elite.reserve(_elite_size);
    for (int i = 0; i < _elite_size; i++) elite.push_back(individuals[i]);
  }


  Genomes individuals;
  Genomes elite;

  int _elite_size = MISSING_DATA_INT;
  vector<double> fitnesses;
  double fitness = 0;
};

inline ostream& operator<<(ostream& out, const Generation& gen) {
  out << "#individuals=" << gen.individuals.size()
    << ", #elite=" << gen.elite.size() 
    <<", fitness=" << gen.fitness << endl;
  return out;
}

class EvolutionFitnessMethod
{
public:
  EvolutionFitnessMethod() {}
  virtual ~EvolutionFitnessMethod() {}

  virtual double fitness(const Genome& genome) = 0;
  virtual double globalFitness(const vector<double>& fitnesses);
};

class EvolutionEngine
{
public:
    // init the engine with some params, nothing else
    EvolutionEngine(const EvolutionParams& params, RandomGenerator& rng);
    ~EvolutionEngine() {}
    
    const EvolutionParams& params() const { return _params; }

    // main function
    pair<Generation, vector<double>> evolve(const Genomes& genomes, EvolutionFitnessMethod& fitness_method, 
        int nb_generations, double maxFit);

    Generation nextGeneration(const Generation& generation, EvolutionFitnessMethod& fitness_method);

    // initial selection, when no fitnesses yet
    // Genomes selectElite(const Genomes& genomes);
    Genomes selectElite(const Genomes& genomes, const vector<double>& fitnesses);
    Genomes selectEvolvers(int nb, const Genomes& genomes, const vector<double>& fitnesses);

    void updateGeneration(Generation& generation, EvolutionFitnessMethod& fitness_method);

    // static Genomes selectBest(const Generation& generation);
private:
    EvolutionParams _params;
    OnePointCrossoverMethod _crossover_method;
    TogglingMutationMethod _mutation_method;
    ElitismWithRandomMethod _elite_selection_method;
    RankBasedSelectionMethod _individuals_selection_method;
};


}
#endif // EVOLUTIONENGINE_H
