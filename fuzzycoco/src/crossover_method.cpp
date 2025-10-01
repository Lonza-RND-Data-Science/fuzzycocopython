#include "crossover_method.h"

using namespace fuzzy_coco;

// cut from ont cut point to the end of the genome minus one
void OnePointCrossoverMethod::reproducePairOf(Genome& gen1, Genome& gen2) {
    const auto nb = gen1.size();
    assert(gen2.size() == nb);
    assert(nb > 2);
    double luck = _rng.randomReal(0, 1);
    if (luck < _prob) {
        // Never exchange the whole genome, must be at least 1bit of the other part.
        // Karl: why ?
        int cutPoint = _rng.random(1, nb - 2);
        for (size_t i = cutPoint; i < nb; i++)
            swap(gen1[i], gen2[i]);
    }
}

void OnePointCrossoverMethod::reproduceAllPairsOf(vector<Genome>& genomes) {
    // process all consecutive pairs. If odd, the last one is not processed
    const int nb_minus_one = genomes.size() - 1;
    for (int i = 0; i < nb_minus_one; i += 2) {
        reproducePairOf(genomes[i], genomes[i + 1]);   
    }
}