/**
 * Karl: note for myself:
 * a EntitySelection an an abstract class for implementing selection methods: vector<PopEntity *> --> vector<PopEntity *>
 */
#ifndef ENTITY_SELECTION_METHOD_H
#define ENTITY_SELECTION_METHOD_H

#include <vector>
#include "random_generator.h"

namespace fuzzy_coco {

using namespace std;

class SelectionMethod
{
public:
    SelectionMethod() {}
    virtual ~SelectionMethod() {}
    virtual void selectEntities(int nb, const vector<double>& fitnesses, vector<int>& indexes) = 0;
};

class RankBasedSelectionMethod : public SelectionMethod
{
public:
    RankBasedSelectionMethod(RandomGenerator& rng) : _rng(rng) {}
    void selectEntities(int nb, const vector<double>& fitnesses, vector<int>& indexes) override;
    
    // karl: N.B: this is a hidden parameter that controls how to sample for best fitnesses
    static constexpr int NB_SPLIT = 10;  

private:
    RandomGenerator& _rng;
};

class ElitismWithRandomMethod : public SelectionMethod
{
public:
    ElitismWithRandomMethod(RandomGenerator& rng) : _rng(rng) {}
    void selectEntities(int nb, const vector<double>& fitnesses, vector<int>& indexes) override;
private:
    RandomGenerator& _rng;
};

}
#endif // ENTITY_SELECTION_METHOD_H
