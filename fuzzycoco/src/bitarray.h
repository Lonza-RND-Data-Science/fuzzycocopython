#ifndef BITARRAY_H
#define BITARRAY_H

#include <vector>
#include <iostream>
#include "random_generator.h"

namespace fuzzy_coco {

using namespace std;
using BitArray = std::vector<bool>;

// decode a number encoded as bits (actually on bools)
namespace BitArrayUtils {
    
    int decode_number(BitArray::const_iterator bits, int nb_bits);
    void encode_number(int number, BitArray::iterator bits, int nb_bits);
    void randomize(BitArray& bits, RandomGenerator& rng);
};

ostream& operator<<(ostream& out, const BitArray& bits);

}
#endif // BITARRAY_H
