#ifndef MUTATION_METHOD_H
#define MUTATION_METHOD_H

#include "genome.h"
#include "random_generator.h"

namespace fuzzy_coco {

// abstract parent class for Mutation implementations
class MutationMethod {
public:
    MutationMethod() {}
    virtual ~MutationMethod() {}
    virtual void mutate(Genome& genome) = 0;
    virtual void mutate(vector<Genome>& genomes) = 0;
};

class TogglingMutationMethod : public MutationMethod
{
public:
    // mutFlipInd: the probability that a genome is a target for a mutation
    // mut_flip_genome: the probability that a bit of a genome is mutated
    TogglingMutationMethod(RandomGenerator& rng, double mutFlipInd, double mut_flip_genome) 
        : _rng(rng), _mutFlipInd(mutFlipInd), _mut_flip_genome(mut_flip_genome) {}

    // N.B: if mutationPerBitProbability == 0 --> flip a single random position
    void mutate(Genome& genome) override;
    void mutate(vector<Genome>& genomes) override;

private:
    RandomGenerator& _rng;
    double _mutFlipInd;
    double _mut_flip_genome;
};

}
#endif // MUTATION_METHOD_H
