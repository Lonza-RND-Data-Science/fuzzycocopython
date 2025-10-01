#include "mutation_method.h"

using namespace fuzzy_coco;

void TogglingMutationMethod::mutate(Genome& genome)
{
    const int nb = genome.size();
    if (_mut_flip_genome == 0) {
        // flip a single random position
        int idx = _rng.random(0, nb - 1);
        genome[idx] = !genome[idx];
        return;
    }
    // batch compute of probs per bit
    vector<double> probs(nb);
    probs.clear();
    _rng.randomReal(0, 1, nb, probs);
    for (int i = 0; i < nb; i++) {
        if (probs[i] < _mut_flip_genome)
             genome[i] = !genome[i];
    
    }
}

void TogglingMutationMethod::mutate(vector<Genome>& genomes) {
    for (auto& gen: genomes) {
        if (_rng.randomReal(0, 1) < _mutFlipInd)
            mutate(gen);
    }
}