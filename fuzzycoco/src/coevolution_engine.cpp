#include "coevolution_engine.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace logging;

CoevolutionEngine:: CoevolutionEngine(
  CoopCoevolutionFitnessMethod& fit, 
  EvolutionEngine& left_engine, 
  EvolutionEngine& right_engine, 
  int nb_cooperators)
    : _fit(fit), 
    _left_engine(left_engine), 
    _right_engine(right_engine), 
    _nb_cooperators(nb_cooperators) 
{}

CoevGeneration CoevolutionEngine::start(const Genomes& left_genos, const Genomes& right_genos)
{
  CoevGeneration cogen({left_genos, getLeftEngine().params().elite_size}, {right_genos, getRightEngine().params().elite_size});
  // Genomes left_coops = selectCooperators(left_genos, getNbCooperators());
  // Genomes right_coops = selectCooperators(right_genos, getNbCooperators()); 
  // auto left_fitnesses = _fit.coopFitness(true, left_genos, right_coops);
  // auto right_fitnesses = _fit.coopFitness(false, right_genos, left_coops);

  // cogen.left_gen.elite = getLeftEngine().selectElite(left_genos);
  // cogen.right_gen.elite = getRightEngine().selectElite(right_genos);

  // only update cogen in-place: set genome fitnesses and elite
  updateGeneration(cogen);

  return cogen;
}//KCOV IGNORE


pair<CoevGeneration, vector<double>> CoevolutionEngine::evolve(const Genomes& left_genos, const Genomes& right_genos, int nb_generations, double maxFit)
{
    // make initial generation: need to select cooperators, compute fitness etc...
    CoevGeneration cogen = start(left_genos, right_genos);
    vector<double> fitnesses;

    vector<double> generation_fitnesses;
    generation_fitnesses.reserve(nb_generations);
    for (int i = 0; i < nb_generations; i++) {
        cogen = nextGeneration(cogen);
        // cerr << cogen.fitness() << ", ";
        generation_fitnesses.push_back(cogen.fitness);
        if (cogen.fitness >= maxFit) break; // early return if we reach the max fitness
    }
    // cerr << endl;

    return make_pair(cogen, generation_fitnesses);
}

// update the fitnesses and elites of both generations of CoevGeneration 
void CoevolutionEngine::updateGeneration(CoevGeneration& cogen)
{
  logger() << "CoevolutionEngine::updateGeneration():,  getNbCooperators()=" <<  getNbCooperators()
    << ", #cogen.right_gen.elite=" << cogen.right_gen.elite.size() << endl;
  Genomes right_coops = selectCooperators(cogen.right_gen.elite, getNbCooperators());
  logger() << "CoevolutionEngine::updateGeneration():" << ", got #right_coops=" << right_coops.size() << endl;
  CoopCoevolutionFitnessMethodAdaptor left_fit(true, getFitnessMethod(), right_coops);
  getLeftEngine().updateGeneration(cogen.left_gen, left_fit);

  Genomes left_coops = selectCooperators(cogen.left_gen.elite, getNbCooperators());
  CoopCoevolutionFitnessMethodAdaptor right_fit(false, getFitnessMethod(), left_coops);
  getRightEngine().updateGeneration(cogen.right_gen, right_fit);
}

// performs selection and reproduction

CoevGeneration CoevolutionEngine::nextGeneration(const CoevGeneration& cogen)
{
  logger() << "CoevolutionEngine::nextGeneration: generation_number=" << cogen.generation_number << endl;
  Genomes right_coops = selectCooperators(cogen.right_gen.elite, getNbCooperators());
  CoopCoevolutionFitnessMethodAdaptor left_fit(true, getFitnessMethod(), right_coops);

  Genomes left_coops = selectCooperators(cogen.left_gen.elite, getNbCooperators());
  CoopCoevolutionFitnessMethodAdaptor right_fit(false, getFitnessMethod(), left_coops);
  
  // making sure getLeftEngine().nextGeneration() is called before getRightEngine().nextGeneration()
  // for portable reproducibility
  auto left_gen_next = getLeftEngine().nextGeneration(cogen.left_gen, left_fit);
  auto right_gen_next = getRightEngine().nextGeneration(cogen.right_gen, right_fit);
  CoevGeneration newcogen(left_gen_next, right_gen_next);
  
  newcogen.generation_number = cogen.generation_number + 1;
  newcogen.fitness = max(newcogen.left_gen.fitness, newcogen.right_gen.fitness);

  return newcogen;
}

Genomes CoevolutionEngine::selectCooperators(const Genomes& elite, int nb_cooperators)
{
  // logger() << "CoevolutionEngine::selectCooperators(): nb_cooperators=" << nb_cooperators << endl;
  if ((size_t)nb_cooperators >= elite.size()) return elite;
  return Genomes(elite.begin(), elite.begin() + nb_cooperators);
}