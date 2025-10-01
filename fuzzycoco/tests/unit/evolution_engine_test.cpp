#include "tests.h"
#include <algorithm>
#include "evolution_engine.h"

using namespace fuzzy_coco;



class SumBitFitness0 : public EvolutionFitnessMethod {
public:
  SumBitFitness0() : EvolutionFitnessMethod() {}
  double fitness(const Genome& genome) override {return double(sum(genome)) / genome.size(); }

};

TEST(EvolutionEngine, evolveFromScratch) {
  Genome zero(100);
  RandomGenerator rng(666);
  vector<Genome> genomes;
  for (int i =0; i < 20; i++) {
    genomes.push_back(zero);
  }

  int maxGen = 10;
  double maxFit = 1;
  EvolutionParams params;
  params.elite_size = 2;
  params.cx_prob = 1;
  params.mut_flip_genome = 0.25;
  params.mut_flip_bit = 0.2;
  cerr << params;
  SumBitFitness0 fit;
  EvolutionEngine evo(params, rng);

  // ================= evolve ================
  auto [lastgen, generation_fitnesses] = evo.evolve(genomes, fit, maxGen, maxFit);

  // difficult to test the values. at least test if that makes sense
  EXPECT_EQ(lastgen.individuals.size(), genomes.size());
  EXPECT_EQ(generation_fitnesses.size(), maxGen);
  EXPECT_DOUBLE_EQ(fit.globalFitness(lastgen.fitnesses), generation_fitnesses.back());
  EXPECT_TRUE(generation_fitnesses.back() > 2*generation_fitnesses.front());

  cerr << endl << "last generation:" << endl;
  for (auto& g : lastgen.individuals)
    cerr << g << endl;
  int nb_gen = generation_fitnesses.size();
  for (int i = 0; i < nb_gen; i++)
     cerr << "generation " << i << ", fitness=" << generation_fitnesses[i] << endl;

  // // select best
  // Genomes best = EvolutionEngine::selectBest(lastgen);
  // EXPECT_TRUE(best.size() > 0);
  // for (auto& g : best)
  //   EXPECT_DOUBLE_EQ(fit.fitness(g), lastgen.fitness);
  // cerr << endl << "best results:" << endl;
  // for (auto& g : best) cerr << g << endl;
  
  // ====== selectElite ============
  vector<Genome> elite = evo.selectElite(genomes, lastgen.fitnesses);
  EXPECT_EQ(elite.size(), params.elite_size);

  // ====== selectEvolvers ============
  vector<Genome> evolvers = evo.selectEvolvers(5, genomes, lastgen.fitnesses);
  EXPECT_EQ(evolvers.size(), 5);

  // ========== updateGeneration ===========
  
  Genome one(50);
  fill(one.begin(), one.end(), true);
  Genomes ones(10, one);

  Generation gen(ones, params.elite_size);

  EXPECT_EQ(gen.elite.size(), params.elite_size);
  EXPECT_TRUE(all(gen.fitnesses, [](auto x) { return x == 0; }));

  evo.updateGeneration(gen, fit);

  EXPECT_EQ(gen.elite.size(), params.elite_size);
  EXPECT_TRUE(all(gen.fitnesses, [](auto x) { return x == 1; }));

  // ========== nextGeneration ===========

  auto newgen = evo.nextGeneration(gen, fit);
  EXPECT_EQ(newgen.elite.size(), elite.size());
  // stable size across generations
  EXPECT_EQ(newgen.individuals.size(), ones.size());
}
