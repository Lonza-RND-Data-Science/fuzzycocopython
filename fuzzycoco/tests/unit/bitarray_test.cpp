
#include "tests.h"
#include "bitarray.h"

using namespace fuzzy_coco;


using namespace BitArrayUtils;
TEST(bitarray, decode_number) {
  BitArray bits(8*100, false);
  
  for (int i = 0; i < 256; i++) {
    BitArrayUtils::encode_number(i, bits.begin(), 8);
    EXPECT_EQ(BitArrayUtils::decode_number(bits.cbegin(), 8), i);
  }

  // vary nb_bits
  BitArrayUtils::encode_number(255, bits.begin(), 8);
  for (int i = 0; i <= 8; i++) {
    EXPECT_EQ(BitArrayUtils::decode_number(bits.cbegin(), i), 255 >> (8 - i));
  }

  // encode many numbers
  {
    auto it = bits.begin();
    for (int i = 0; i < 100; i++) {
      BitArrayUtils::encode_number(i, it, 8);
      it += 8;
    }
  }
  {
    auto it = bits.cbegin();
    for (int i = 0; i < 100; i++) {
      EXPECT_EQ(BitArrayUtils::decode_number(it, 8), i);
      it += 8;
    }
  }
}

TEST(bitarray, randomize) {
  BitArray bits(100);
  EXPECT_EQ(sum(bits), 0);

  RandomGenerator rng(666);

  randomize(bits, rng);
  EXPECT_TRUE(sum(bits) > 0);
  EXPECT_TRUE(sum(bits) < int(bits.size()));

  // reproducible
  BitArray bits2(100);
  RandomGenerator rng2(666);
  randomize(bits2, rng2);
  EXPECT_EQ(bits2, bits);

}