#include "selection_method.h"
#include <algorithm>

using namespace fuzzy_coco;

void RankBasedSelectionMethod::selectEntities(int nb, const vector<double>& fitnesses, vector<int>& indexes)
{
    indexes.clear();
    const int nb_entities = fitnesses.size();
    assert(nb <= nb_entities);
    if (nb_entities == 0 || nb == 0) return;
    // karl: N.B: this is a hidden parameter that controls how to sample for best fitnesses
    const int nb_tries = nb_entities / NB_SPLIT;
    vector<int> random_indexes(nb_tries);

    double fitness = 0;
    int idx = 0;
    for (int k = 0; k < nb; k++)
    {
        double bestfitness = numeric_limits<double>::lowest();
        double bestidx = 0;
        random_indexes.clear();
        _rng.random(0, nb_entities - 1, nb_tries, random_indexes); // batch it for speed
        for (int i = 0; i < nb_tries; i++) {
            idx = random_indexes[i];
            fitness = fitnesses[idx];
            if (fitness > bestfitness){
                bestfitness = fitness;
                bestidx = idx;
            }
        }
        indexes.push_back(bestidx);
    }
}

void ElitismWithRandomMethod::selectEntities(int nb, const vector<double>& fitnesses, vector<int>& indexes)
{
    indexes.clear();
    const int nb_entities = fitnesses.size();
    assert(nb <= nb_entities);
    if (nb_entities == 0 || nb == 0) return;

    vector<int> fit_idx(nb_entities);
    for (int i = 0; i < nb_entities; i++) fit_idx[i] = i;


    // sort the fitness indexes in decreasing order
    // improvement: make it a deterministic sort by handling ties with indices
    sort(fit_idx.begin(), fit_idx.end(), [&](int a, int b) { 
      return fitnesses[a] > fitnesses[b] ? true : fitnesses[a] < fitnesses[b] ? false : a < b;
    });

    // only take the first nb-1 elements
    copy_n(fit_idx.begin(), min(nb - 1, nb_entities), back_inserter(indexes));

    // the last element is taken randomly
    int random_fit_idx_idx = _rng.random(fit_idx);

    indexes.push_back(random_fit_idx_idx);
}
