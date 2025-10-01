#include "tests.h"
#include <algorithm>
#include <cstring>
#include "random_generator.h"
#include "digest.h"

using namespace fuzzy_coco;
using namespace Digest;

template<typename T>
ostream& operator<<(ostream& out, const vector<T>& v) {
  out << "{";
  const int nb = v.size() - 1;
  for (int i = 0; i < nb; ++i) {
      out << v[i] << ", "; 
  }
  if (nb >= 0) out << v[nb];
  out << "}";
  return out;
}

TEST(RandomGenerator, random) {
  RandomGenerator rng(123);

  EXPECT_EQ(rng.random(0, 0), 0);
  EXPECT_EQ(rng.random(10, 10), 10);
  EXPECT_EQ(rng.random(-2, -2), -2);

  vector<int> r1;
  rng.random(1, 10, 1000, r1);
  EXPECT_EQ(r1.size(), 1000);
  EXPECT_EQ(*min_element(r1.begin(), r1.end()), 1);
  EXPECT_EQ(*max_element(r1.begin(), r1.end()), 10);
}

TEST(RandomGenerator, randomReal) {
  RandomGenerator rng(456);

  EXPECT_DOUBLE_EQ(rng.randomReal(0, 0), 0);
  EXPECT_EQ(rng.randomReal(0.21, 0.21), 0.21);
  EXPECT_EQ(rng.randomReal(-2.01, -2.01), -2.01);

  vector<double> r1;
  rng.randomReal(-1, 2, 1000, r1);
  EXPECT_EQ(r1.size(), 1000);
  EXPECT_TRUE(*min_element(r1.begin(), r1.end()) >= -1);
  EXPECT_TRUE(*max_element(r1.begin(), r1.end()) <= 2);
}

TEST(RandomGenerator, reproducibility) {
  RandomGenerator rng1(456);
  RandomGenerator rng2(456);
  RandomGenerator rng3(456);

  vector<int> ints1, ints2;
  rng1.random(-1, 3, 100, ints1);
  rng2.random(-1, 3, 100, ints2);
  EXPECT_EQ(ints1, ints2);

  bool all_equal = true;
  for (int i = 0; i < 100; i++)
    if (rng3.random(-1, 3) != ints1[i]) {
      all_equal = false;
      break;
    }
  EXPECT_TRUE(all_equal);  

  vector<double> reals1, reals2;
  rng1.randomReal(-1000.1, -200.04, 100, reals1);
  rng2.randomReal(-1000.1, -200.04, 100, reals2);
  EXPECT_EQ(reals1, reals2);
}

TEST(RandomGenerator, seed) {
  RandomGenerator rng1(123);
  RandomGenerator rng2(456);

  vector<int> ints1, ints2;
  rng1.random(-1, 3, 100, ints1);
  rng2.random(-1, 3, 100, ints2);
  EXPECT_NE(ints1, ints2);
}

TEST(RandomGenerator, vector) {
  RandomGenerator rng1(123);
  RandomGenerator rng2(123);

  int nb = 10;
  vector<double> v(nb);
  for (int i = 0; i < nb; ++i) {
    v[i] = i / 11.2;
  }
  EXPECT_DOUBLE_EQ(rng1.random(v), v[rng2.random(0, nb - 1)]);

}

TEST(mt19937, portability) {
    std::mt19937 mt(666);
    uint_fast32_t expected[10] = {3008354540,440739714,3625754029,907667358,2905606974,553951302,3126126537,3222645150,4086480804,1442973117};

    for (int i = 0; i < 10; i++) {
      EXPECT_EQ(mt(), expected[i]);
    }

    
    {

      int r = mt19937::max();
      double norm1 = double(r) / double(mt19937::max() + 1.0);
      double norm2 = std::ldexp(r, -32);
      cout  << setprecision(numeric_limits<double>::max_digits10 * 2)  << r << ":" << norm1 << "," << norm2 << endl;

    }


    // random
    {
      RandomGenerator rng(123);
      int expected[] = {-1283394130,1572590647,-1408460518,2045083368,1182622138,-1864563734,1697250018,-1459027229,1100910199,-2067864206,-1498199443,848784249,-1599039828,-1834388561,-1758575032,-1472022046,-1615485263,-1953253780,988267000,-410378308};
      for (int i = 0; i < 20; i++) {
        int r1 = rng.random();
        int r2 = rng.random();
        int k = rng.random(min(r1, r2), max(r1, r2));
        // cout << k << ",";
        EXPECT_EQ(k, expected[i]);
      }

    }

    // macOs bug
    {
// -944773575, 2109339754, 1817228411, 2045083367.675199031829833984375, 41de795fb9eb3676
      int x = -944773575;
      int r1 = 1817228411;
      int r2 = 2109339754;
      EXPECT_EQ(double_to_hex(RandomGenerator::scale_int_to_double_strict(x, r1, r2)), "41de795fb9eb3676");
    }
}