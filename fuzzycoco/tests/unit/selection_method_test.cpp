#include "tests.h"
#include <algorithm>
#include "selection_method.h"
using namespace fuzzy_coco;


TEST(RankBasedSelectionMethod, selectEntities) {
  vector<double> fitnesses;

  RandomGenerator rng(123);
  RankBasedSelectionMethod selector(rng);

  // empty
  vector<int> indexes;
  selector.selectEntities(0, fitnesses, indexes);
  EXPECT_EQ(indexes.size(), 0);
  
  // one element
  fitnesses.push_back(0.2);
  indexes.clear();

  selector.selectEntities(1, fitnesses, indexes);

  EXPECT_EQ(indexes.size(), 1);
  EXPECT_EQ(indexes[0], 0);

  // standard case
  fitnesses.clear();
  int nb = 100;
  for (int i = 0; i < nb; i++) {
    fitnesses.push_back(i);
  }

  selector.selectEntities(10, fitnesses, indexes);

  EXPECT_EQ(indexes.size(), 10);
  EXPECT_TRUE(all(indexes, [](auto x) { return x > 70; }));
  // cerr << indexes;
}

TEST(ElitismWithRandomMethod, selectEntities) {
  vector<double> fitnesses;

  RandomGenerator rng(123);
  ElitismWithRandomMethod selector(rng);

  // empty
  vector<int> indexes;
  selector.selectEntities(0, fitnesses, indexes);
  EXPECT_EQ(indexes.size(), 0);
  
  // one element
  fitnesses.push_back(0.2);
  indexes.clear();

  selector.selectEntities(1, fitnesses, indexes);

  EXPECT_EQ(indexes.size(), 1);
  EXPECT_EQ(indexes[0], 0);

  // standard case
  fitnesses.clear();
  int nb = 100;
  for (int i = 0; i < nb; i++) {
    fitnesses.push_back(i);
  }

  selector.selectEntities(10, fitnesses, indexes);

  EXPECT_EQ(indexes.size(), 10);
  // sorted except for the last one
  for (int i = 0; i < 9; i++) 
    EXPECT_EQ(indexes[i], 99 - i);
  EXPECT_NE(indexes[9], 99 -9);
  // cerr << indexes << endl;;


}