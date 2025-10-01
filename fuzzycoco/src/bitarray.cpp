#include "bitarray.h"
#include <ios>

using namespace fuzzy_coco;

int BitArrayUtils::decode_number(BitArray::const_iterator it, int nb_bits) {
    int decoded = 0;
    for (int i = 0; i < nb_bits; i++) {
        decoded += (*it) << i;
        it++;
    }
    return decoded;
}

void BitArrayUtils::encode_number(int number, BitArray::iterator it, int nb_bits) {
    int remainder = number;
    for (int i = 0; i < nb_bits; i++) {
        *it++ = remainder & 1;
        remainder >>= 1;
    }
}

void BitArrayUtils::randomize(BitArray& bits, RandomGenerator& rng) {
    const size_t nb = bits.size();
    vector<int> probs(nb);
    probs.resize(0);
    rng.random(0, 1, int(nb), probs); // batch

    for (size_t i = 0; i < nb; i++) {
      bits[i] = probs[i];
    }
}

namespace fuzzy_coco {
  ostream& operator<<(ostream& out, const BitArray& bits) {
    for (const auto& bit: bits)
      out << noboolalpha << bit;
    return out;
  }
}//KCOV IGNORE
