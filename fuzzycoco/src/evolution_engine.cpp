#include "evolution_engine.h"
#include <algorithm>

using namespace fuzzy_coco;

double EvolutionFitnessMethod::globalFitness(const vector<double>& fitnesses) {
    return *max_element(fitnesses.begin(), fitnesses.end());
}

EvolutionEngine::EvolutionEngine(const EvolutionParams& params, RandomGenerator& rng) :
        _params(params),
        _crossover_method(rng, params.cx_prob),
        _mutation_method(rng, params.mut_flip_genome, params.mut_flip_bit),
        _elite_selection_method(rng),
        _individuals_selection_method(rng)
{}

pair<Generation, vector<double>> EvolutionEngine::evolve(const Genomes& genomes, EvolutionFitnessMethod& fitness_method, 
    int nb_generations, double maxFit) 
{
  Generation generation(genomes, params().elite_size);
  updateGeneration(generation, fitness_method);

  vector<double> generation_fitnesses;
  generation_fitnesses.reserve(nb_generations);

  for (int i = 0; i < nb_generations; i++) {
      generation = nextGeneration(generation, fitness_method);
      generation_fitnesses.push_back(generation.fitness);
      if (generation.fitness >= maxFit) break; // early return if we reach the max fitness
  }
  return make_pair(generation, generation_fitnesses);
}

// compute fitness for all individuals and select elite based on that fitnesses
// after that --> generation is up-to-date
void EvolutionEngine::updateGeneration(Generation& generation, EvolutionFitnessMethod& fitness_method)
{
    const auto& genos = generation.individuals;
    const int nb = genos.size();
    generation.fitnesses.resize(nb);

    for (int k = 0; k < nb; k++) generation.fitnesses[k] = fitness_method.fitness(genos[k]);

    generation.fitness = fitness_method.globalFitness(generation.fitnesses);
    generation.elite = selectElite(genos, generation.fitnesses);
}

// performs selection and reproduction
Generation EvolutionEngine::nextGeneration(const Generation& generation, EvolutionFitnessMethod& fitness_method)
{   
    const auto& genomes = generation.individuals;
    const size_t nb = genomes.size();
    const size_t nb_elite = _params.elite_size;
    const size_t nb_evolvers = nb - nb_elite;

    assert(generation.elite.size() == nb_elite);
    assert(nb_elite < nb);
    assert(nb_evolvers <= nb);

    Genomes evolvers = selectEvolvers(nb_evolvers, generation.individuals, generation.fitnesses);

    // N.B: modify in-place
    _crossover_method.reproduceAllPairsOf(evolvers);
    _mutation_method.mutate(evolvers);

    // new generation is elite + evolved
    Genomes indiv;
    const auto& elite = generation.elite;
    indiv.reserve(elite.size() + evolvers.size());
    indiv.insert(indiv.end(), elite.begin(), elite.end());
    indiv.insert(indiv.end(), evolvers.begin(), evolvers.end());

    Generation newgen(indiv, _params.elite_size);
    updateGeneration(newgen, fitness_method);

    return newgen;
}
 
// Genomes EvolutionEngine::selectElite(const Genomes& genomes)
// {
//     const int nb_elite = _params.elite_size;
//     assert(nb_elite < genomes.size());

//     Genomes elite(genomes.begin(), genomes.begin() + nb_elite);
//     return elite;
// }

Genomes EvolutionEngine::selectElite(const Genomes& genomes, const vector<double>& fitnesses)
{

  const int nb_elite = _params.elite_size;
  vector<int> indexes; // TODO: put in instance state
  indexes.reserve(nb_elite);
  _elite_selection_method.selectEntities(nb_elite, fitnesses, indexes);
  Genomes elite;
  elite.reserve(nb_elite);
  for (int idx : indexes) elite.push_back(genomes[idx]);
  return elite;
}

Genomes EvolutionEngine::selectEvolvers(int nb, const Genomes& genomes, const vector<double>& fitnesses)
{
    vector<int> indexes; // TODO: put in instance state 
    indexes.reserve(nb);
    indexes.clear();
    _individuals_selection_method.selectEntities(nb, fitnesses, indexes);
    Genomes evolvers;
    evolvers.reserve(nb);
    for (int idx : indexes) evolvers.push_back(genomes[idx]);
    return evolvers;
}

