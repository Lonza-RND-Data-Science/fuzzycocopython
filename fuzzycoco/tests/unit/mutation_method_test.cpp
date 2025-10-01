#include "tests.h"
#include "mutation_method.h"

using namespace fuzzy_coco;


TEST(TogglingMutationMethod, mutate) {
  Genome gen(100);

  RandomGenerator rng(123);

  auto sum = [](auto v) { return accumulate(v.begin(), v.end(), 0); };
  // very small pr mut.mutate(gen, 0.5);ob (but not 0) ==> we do not expect any change
  TogglingMutationMethod(rng, 1, 1e-9).mutate(gen);
  EXPECT_EQ(sum(gen), 0);

  // 0 prob --> one single mutation
  gen.clear(); gen.resize(100);
  TogglingMutationMethod(rng, 1, 0).mutate(gen);
  EXPECT_EQ(sum(gen), 1);

  // usual case
  gen.clear(); gen.resize(100);
  TogglingMutationMethod(rng, 1, 0.5).mutate(gen);

  EXPECT_TRUE(fabs(50 - sum(gen)) < 10);

  // prob == 1 --> all flipped
  gen.clear(); gen.resize(100);
  TogglingMutationMethod(rng, 1, 1).mutate(gen);
  EXPECT_EQ(sum(gen), 100);

}

TEST(TogglingMutationMethod, mutate_genomes) {
  RandomGenerator rng(123);
  Genome gen0(100);
  vector<Genome> genomes;
  for (int i =0; i < 10; i++) 
    genomes.push_back(gen0);
  
  vector<Genome> gens;

  // no individuals should be mutated
  gens = genomes;
  TogglingMutationMethod(rng, 0, 1).mutate(gens);
  for (const auto& g : gens) {
    EXPECT_EQ(sum(g), 0);
  }

  // all individuals should be mutated
  gens = genomes;
  TogglingMutationMethod(rng, 1, 1).mutate(gens);
  for (const auto& g : gens) {
    EXPECT_EQ(sum(g), 100);
  }

  // all individuals should be mutated
  gens = genomes;
  TogglingMutationMethod(rng, 1, 0.5).mutate(gens);
  for (const auto& g : gens) {
    auto x = sum(g);
    EXPECT_TRUE(x > 10 && x < 90);
  }

}

