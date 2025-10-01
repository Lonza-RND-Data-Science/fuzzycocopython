#include "tests.h"
#include "genome_codec.h"

using namespace fuzzy_coco;



TEST(IntCodec, basic) {
  int nb_bits = 32;
  int nb_ints = 10;
  BitArray bits(nb_ints * nb_bits, false);
  IntCodec codec(nb_bits);

  EXPECT_EQ(codec.size(), nb_bits);

  int i = 0;
  for (auto it = bits.begin(); it != bits.end();) {
    codec.encode(i++, it);
  }

  i = 0;
  for (auto it = bits.cbegin(); it != bits.cend();) {
    EXPECT_EQ(codec.decode(it), i++);
  }
}

TEST(IntVectorCodec, basic) {
  int nb_bits = 4;
  int nb = 10;

  IntVectorCodec codec(nb, nb_bits);
  BitArray bits(codec.size(), false);

  EXPECT_EQ(codec.size(), nb * nb_bits);
  EXPECT_EQ(codec.getNumberOfElements(), nb);

  vector<int> ints;
  for (int i = 0; i < nb; i++) ints.push_back(i);

  auto it = bits.begin();
  codec.encode(ints, it);

  vector<int> res;
  auto cit = bits.cbegin();
  codec.decode(cit, res);

  EXPECT_EQ(res, ints);
}


TEST(ConditionIndexCodec, basic) {
  int var_idx_nb_bits = 4;
  int set_idx_nb_bits = 2;
  int nb = 10;
  BitArray bits((var_idx_nb_bits + set_idx_nb_bits) * nb, false);
  ConditionIndexCodec codec(var_idx_nb_bits, set_idx_nb_bits);

  EXPECT_EQ(codec.size(), var_idx_nb_bits + set_idx_nb_bits);

  auto it = bits.begin();
  ConditionIndex ci(0, 0);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      codec.encode(ConditionIndex{i, j}, it);

  auto cit = bits.cbegin();
  EXPECT_EQ(codec.decode(cit), ConditionIndex(0, 0));
  EXPECT_EQ(codec.decode(cit), ConditionIndex(0, 1));
  EXPECT_EQ(codec.decode(cit), ConditionIndex(1, 0));
  EXPECT_EQ(codec.decode(cit), ConditionIndex(1, 1));

  // ConditionIndexes
  ConditionIndexes cis;
  cis.reserve(10);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      cis.push_back({i, j});

  it = bits.begin();
  codec.encode(cis, it);

  ConditionIndexes res;
  cit = bits.cbegin();
  codec.decode(cit, res, 6);
  EXPECT_EQ(res, cis);
}

TEST(ConditionIndexesCodec, basic) {
  IntPairParams params(15, 5, 3);
  ConditionIndexesCodec codec(params);

  EXPECT_EQ(codec.size(), (params.nb_bits1 + params.nb_bits2) * params.nb);
  EXPECT_EQ(codec.getNumberOfConditions(), params.nb);

  BitArray bits(codec.size(), false);

  ConditionIndexes cis;
  cis.reserve(10);
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 3; j++)
      cis.push_back({i, j});

  auto it = bits.begin();
  codec.encode(cis, it);

  ConditionIndexes res;
  auto cit = bits.cbegin();
  codec.decode(cit, res);
  EXPECT_EQ(res, cis);

  cerr << cis;
}


// TEST(ConditionIndexesVectorCodec, basic) {
//   int nb = 4;
//   IntPairParams params(5, 4, 3);
//   ConditionIndexesVectorCodec codec(nb, params);

//   EXPECT_EQ(codec.size(), (params.nb_bits1 + params.nb_bits2) * params.nb * nb);

//   BitArray bits(codec.size(), false);

//   vector<ConditionIndexes> cond_lst;
//   cond_lst.reserve(nb);
//   for (int k = 0; k < nb; k++) {
//       ConditionIndexes cis;
//       cis.reserve(params.nb);
//       for (int i = 0; i < params.nb; i++)
//         cis.push_back({(i+8)%16, (i + 3) % 8});
//       cond_lst.push_back(cis);
//   }
 
//   auto it = bits.begin();
//   codec.encode(cond_lst, it);

//   vector<ConditionIndexes> res;
//   auto cit = bits.cbegin();
//   codec.decode(cit, res);
//   EXPECT_EQ(res, cond_lst);
// }

TEST(RuleCodec, basic) {
  IntPairParams params1(3, 5, 3);
  IntPairParams params2(2, 3, 2);

  RuleCodec codec(params1, params2);
  cerr << codec;

  EXPECT_EQ(codec.size(), (3*(5 + 3) + 2 * (3 + 2)));

  BitArray bits(codec.size(), false);

  ConditionIndexes cis_in = {{ 1, 1}, {1, 2}, {2, 0}};
  ConditionIndexes cis_out = {{0, 2}, {2, 2}};

  auto it = bits.begin();
  codec.encode(cis_in, cis_out, it);

  ConditionIndexes res1, res2;
  auto cit = bits.cbegin();
  codec.decode(cit, res1, res2);
  EXPECT_EQ(res1, cis_in);
  EXPECT_EQ(res2, cis_out);
}


TEST(RulesCodec, basic) {
  IntPairParams params1(3, 5, 3);
  IntPairParams params2(2, 3, 2);
  int nb_rules = 4;
  RulesCodec codec(4, params1, params2);
  cerr << codec;

  EXPECT_EQ(codec.size(), (3*(5 + 3) + 2 * (3 + 2)) * nb_rules + 2*2);

  BitArray bits(codec.size(), false);

  vector<ConditionIndexes> rules_in = {
    {{ 1, 1}, {1, 2}, {2, 0}},
    {{ 0, 0}, {0, 1}, {0, 2}},
    {{2, 1}, {2, 0}, {2, 0}},
    {{0, 2}, {2, 2}, {1,2 }},
  };

  vector<ConditionIndexes> rules_out = {
    {{ 1, 1}, {1, 2}},
    {{0, 2}, {2, 2}},
    {{2, 1}, {2, 0}},
    {{ 0, 0}, {0, 1}},
  };

  vector<int> default_rules;
  for (int i = 0; i < 2; i++)
    default_rules.push_back((i + 7) % 2);

  auto it = bits.begin();
  codec.encode(rules_in, rules_out, default_rules, it);

  vector<ConditionIndexes> res1, res2;
  vector<int> res3;
  auto cit = bits.cbegin();
  codec.decode(cit, res1, res2, res3);
  EXPECT_EQ(res1, rules_in);
  EXPECT_EQ(res2, rules_out);
  EXPECT_EQ(res3, default_rules);
}

TEST(DiscretizedFuzzySystemSetPositionsCodec, basic) {

  PosParams params1(3, 5, 3); // 3 vars, 5 sets, 3 bits per set
  PosParams params2(2, 3, 2);// 2 vars, 3 sets, 2 bits

  // positions for input sets
  vector<Discretizer> disc_in = { 
    {4, 0, 100},
    {2, -5, -1},
    {8, 100, 10000}
  };
  // positions for output sets
  vector<Discretizer> disc_out = { 
    {1, 0, 1},
    {3, 5, 9}
  };
  
  DiscretizedFuzzySystemSetPositionsCodec codec(params1, params2, disc_in, disc_out);
  cerr << codec << endl;

  EXPECT_EQ(codec.size(), (3* 5 * 3 + 2 * 3 * 2));

  BitArray bits(codec.size(), false);

  Matrix<int> pos_in = {
    {1, 2, 3, 4, 5},
    {5, 1, 2, 4, 5},
    {5, 1, 0, 4, 1},
  };

  Matrix<int> pos_out = {
    {0, 1, 2},
    {2, 1, 0},
  };

  Matrix<double> pos_in2(pos_in.nbrows(), pos_in.nbcols());
  for (int i = 0; i < pos_in.nbrows(); i++)
    for (int j = 0; j < pos_in.nbcols(); j++)
      pos_in2[i][j] = disc_in[i].undiscretize(pos_in[i][j]);
  // cerr << pos_in << pos_in2;

  Matrix<double> pos_out2(pos_out.nbrows(), pos_out.nbcols());
  for (int i = 0; i < pos_out.nbrows(); i++)
    for (int j = 0; j < pos_out.nbcols(); j++)
      pos_out2[i][j] = disc_out[i].undiscretize(pos_out[i][j]);
  // cerr << pos_out << pos_out2;

  auto it = bits.begin();
  codec.encode(pos_in2, pos_out2, it);

  // Matrix<double> res1(pos_in.nbrows(), pos_in.nbcols());
  // Matrix<double> res2(pos_out.nbrows(), pos_out.nbcols());
  Matrix<double> res1, res2;

  auto cit = bits.cbegin();
  codec.decode(cit, res1, res2);

  cerr << res1 << res2;

  EXPECT_EQ(res1, pos_in2);
  EXPECT_EQ(res2, pos_out2);
}
