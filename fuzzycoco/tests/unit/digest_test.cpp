#include "tests.h"
#include <algorithm>
#include <cstring>
#include "digest.h"

using namespace fuzzy_coco;
using namespace Digest;

TEST(RandomGenerator, double_to_hex) {
  double d = 22.0/7.0;
  auto hex = double_to_hex(d);
  cerr << hex << endl;

  EXPECT_EQ(hex_to_double(hex), d);
}

TEST(RandomGenerator, hash_string) {
  string s = "hash_me";
  auto hash = hash_string(s);
  cerr << uint64_to_hex(hash) << endl;

  EXPECT_EQ(hash_string(s), hash);
  EXPECT_NE(hash_string("hash_you"), hash);
}

TEST(RandomGenerator, digest) {
  vector<string> v = { "un", "deux", "trois" };
  vector<string> v2 = { "deux", "un", "trois" };
  vector<string> v3 = { "un", "deuX", "trois" };
  
  auto hash = digest(v);
  cerr << hash << endl;

  EXPECT_EQ(digest(v), hash);
  EXPECT_NE(digest(v2), hash);
  EXPECT_NE(digest(v3), hash);
}

TEST(RandomGenerator, digest_double) {
  vector<double> v = { 1, 2, 3 };
  vector<double> v2 = { 3, 2, 1};
  vector<double> v3 = { 1, 2.00001, 3};
  
  auto hash = digest(v);
  cerr << hash << endl;

  EXPECT_EQ(digest(v), hash);
  EXPECT_NE(digest(v2), hash);
  EXPECT_NE(digest(v3), hash);
}
