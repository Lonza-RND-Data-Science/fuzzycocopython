#include "tests.h"
#include "crossover_method.h"

using namespace fuzzy_coco;



TEST(TogglingMutationMethod, reproducePairOf) {
  Genome gen1(100), gen2(100);
  fill(gen1.begin(), gen1.end(), true);
  RandomGenerator rng(1234);

  OnePointCrossoverMethod cross0(rng, 0), cross1(rng, 1);

  // prob == 0 ==> no change
  cross0.reproducePairOf(gen1, gen2);
  EXPECT_EQ(sum(gen1), 100);
  EXPECT_EQ(sum(gen2), 0);

  // prob = 1 ==> certainly a change
  cross1.reproducePairOf(gen1, gen2);
  int diff = sum(gen2);
  EXPECT_TRUE(diff > 0);
  EXPECT_EQ(sum(gen1) + diff, 100);


  cerr << gen1 << endl << gen2 << endl;
}


TEST(TogglingMutationMethod, reproduceAllPairsOf) {
  Genome gen1(100), gen2(100);
  fill(gen1.begin(), gen1.end(), true);
  RandomGenerator rng(666);

  OnePointCrossoverMethod cross0(rng, 0), cross1(rng, 1);

  vector<Genome> genomes;
  for (int i =0; i < 10; i++) {
    genomes.push_back(gen1);
    genomes.push_back(gen2);
  }

  // prob == 0
  vector<Genome> gs1 = genomes;
  cross0.reproduceAllPairsOf(gs1);
  for (auto i = 0U; i < gs1.size(); i++) {
    EXPECT_EQ(sum(gs1[i]), i%2 == 0 ? 100 : 0);
  }
  EXPECT_EQ(gs1, genomes);

  // prob == 1
  gs1 = genomes;
  cross1.reproduceAllPairsOf(gs1);
  for (const auto& g : gs1) {
    EXPECT_NE(sum(g), 100);
  }

  // odd number of genos
  gs1 = genomes;
  gs1.push_back(gen1);
  cross1.reproduceAllPairsOf(gs1);
  // last unchanged
  EXPECT_EQ(gs1.back(), gen1);
}
