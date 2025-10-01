#include "coevolution_fitness.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace logging;

double CoopCoevolutionFitnessMethod::coopFitness(bool left, const Genome& genome, const Genomes& cooperators) {
  double max_fitness = numeric_limits<double>::lowest();
  double fit = -1;
  for (const auto& coop : cooperators) {
    if (left)
        fit = fitness(genome, coop);
    else
        fit = fitness(coop, genome);
    max_fitness = max(max_fitness, fit);
  }
  return max_fitness;
}


// vector<double> CoopCoevolutionFitnessMethod::coopFitness(bool left, const Genomes& genomes, const Genomes& cooperators)
// {
//     cerr << "CoopCoevolutionFitnessMethod::coopFitness():" << "left=" << left << " " << genomes[0].size() << "-" << cooperators[0].size() << endl;
//     const int nb = genomes.size();
//     vector<double> fitnesses(nb);
//     for (int i = 0; i < nb; i++)
//         fitnesses[i] = coopFitness(left, genomes[i], cooperators);
//     return fitnesses;
// }
