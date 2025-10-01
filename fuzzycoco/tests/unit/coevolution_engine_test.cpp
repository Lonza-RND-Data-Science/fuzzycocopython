#include "tests.h"
#include "coevolution_engine.h"

using namespace fuzzy_coco;



class AlignedBitsFitness : public CoopCoevolutionFitnessMethod {
public:
  AlignedBitsFitness() {}
  double fitnessImpl(const Genome& genome1, const Genome& genome2) override {
    int aligned = 0;
    int nb = genome1.size();
    for (int i = 0; i < nb; i++)
      aligned += (genome1[i] == genome2[i]);
    return double(aligned) / nb; 
  }

  // virtual double fitness(const vector<Genome>& genomes) override {
  //   double total = 0;
  //   for (const auto& g : genomes) total += fitness(g);
  //   return total / genomes.size();
  // }
};


TEST(CoEvolutionEngine, evolve) {
  Genome one(100), zero(100);
  fill(one.begin(), one.end(), true);
  RandomGenerator left_rng(123);
  RandomGenerator right_rng(234);

  Genomes left_genomes, right_genomes;
  for (int i =0; i < 10; i++) {
    left_genomes.push_back(one);
  }
  for (int i =0; i < 15; i++) {
    right_genomes.push_back(zero);
  }
  // N.B: inequal pop sizes

  EvolutionParams params;
  params.elite_size = 2;
  params.cx_prob = 1;
  params.mut_flip_genome = 0.25;
  params.mut_flip_bit = 0.2;

  cerr << params << endl;

  EvolutionEngine left_evo(params, left_rng);
  EvolutionEngine right_evo(params, right_rng);
  AlignedBitsFitness fitter;

  CoevolutionEngine coevo(fitter, left_evo, right_evo, 3);

  // ================= evolve ================

  auto [lastgen, generation_fitnesses] = coevo.evolve(left_genomes, right_genomes, 20, 1);
  EXPECT_EQ(lastgen.generation_number, 20);

  // cerr << "fitnesses: " << generation_fitnesses << endl;

  EXPECT_EQ(lastgen.left_gen.individuals.size(), left_genomes.size());
  EXPECT_EQ(lastgen.right_gen.individuals.size(), right_genomes.size());
  EXPECT_EQ(generation_fitnesses.size(), 20);
  EXPECT_TRUE(generation_fitnesses.back() > 2*generation_fitnesses.front());
  // EXPECT_TRUE(generation_fitnesses.back() > 3*generation_fitnesses.front());

  // ========= selectBest =========
  auto [left_best, right_best] = coevo.getBest();
  int nbbest = left_best.size();
  EXPECT_TRUE(nbbest > 0);
  EXPECT_EQ(right_best.size(), nbbest);
  EXPECT_TRUE(fitter.fitnessImpl(left_best, right_best) >= generation_fitnesses.back());
  cerr << endl << "best pair: " << endl;
  cerr << left_best << endl << right_best << endl;
}

TEST(CoEvolutionEngine, iterator) {
  Genome one(100), zero(100);
  fill(one.begin(), one.end(), true);
  RandomGenerator left_rng(123);
  RandomGenerator right_rng(234);

  Genomes left_genomes, right_genomes;
  for (int i =0; i < 10; i++) {
    left_genomes.push_back(one);
  }
  for (int i =0; i < 15; i++) {
    right_genomes.push_back(zero);
  }

  EvolutionParams params;
  params.elite_size = 2;
  params.cx_prob = 1;
  params.mut_flip_genome = 0.25;
  params.mut_flip_bit = 0.2;

  EvolutionEngine left_evo(params, left_rng);
  EvolutionEngine right_evo(params, right_rng);
  AlignedBitsFitness fitter;
  CoevolutionEngine coevo(fitter, left_evo, right_evo, 3);

  auto gen = coevo.start(left_genomes, right_genomes);
  EXPECT_EQ(gen.generation_number, 0);
  vector<double> generation_fitnesses;
  for (int i = 0; i < 20; i++) {
    gen = coevo.nextGeneration(gen);
    generation_fitnesses.push_back(gen.fitness);
  }
  EXPECT_EQ(gen.generation_number, 20);

  // ========= selectBest =========
  auto [left_best, right_best] = coevo.getBest();
  int nbbest = left_best.size();
  EXPECT_TRUE(nbbest > 0);
  EXPECT_EQ(right_best.size(), nbbest);
  EXPECT_TRUE(fitter.fitnessImpl(left_best, right_best) >= generation_fitnesses.back());
  cerr << endl << "best pair: " << endl;
  cerr << "fitness: " << fitter.fitnessImpl(left_best, right_best) << endl;
  cerr << left_best << endl << right_best << endl;
}