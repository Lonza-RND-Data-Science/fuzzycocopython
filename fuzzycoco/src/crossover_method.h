#ifndef CROSSOVER_METHOD_H
#define CROSSOVER_METHOD_H

#include "genome.h"
#include "random_generator.h"

namespace fuzzy_coco {


class OnePointCrossoverMethod  {
public:
    OnePointCrossoverMethod(RandomGenerator& rng, double prob) : _rng(rng), _prob(prob) {}
    void reproduceAllPairsOf(vector<Genome>& genomes);
    void reproducePairOf(Genome& gen1, Genome& gen2) ;
private:
    RandomGenerator& _rng;
    double _prob;
};

}
#endif // CROSSOVER_METHOD_H

