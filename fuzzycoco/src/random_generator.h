
#ifndef RANDOM_GENERATOR_H
#define RANDOM_GENERATOR_H

#include <random>
#include <cassert>
#include <cstring>

namespace fuzzy_coco {

using namespace std;
class RandomGenerator
{
public:
  RandomGenerator(int seed = random_device{}()) : _rng(seed) {}

  uint32_t random() { 
    return _rng(); 
  }

  int random(int min, int max) {
    // uniform_int_distribution<> distrib(min, max);
    // return distrib(_rng);
    return scale_int_strict(random(), min, max);
  }

  // N.B: append. no reserve
  void random(int min, int max, int nb, vector<int>& stack) {
    // uniform_int_distribution<> distrib(min, max);
    for (int i = 0 ; i < nb; i++)
      stack.push_back(random(min, max));
  }

  template <typename T>
  const T& random(const vector<T>& v) {
    int nb = v.size();
    assert(nb > 0);
    return v[random(0, nb - 1)];
  }

  double randomReal(double min, double max) {
    // uniform_real_distribution<double> distrib(min, max);
    // return distrib(_rng);
    return scale_int_to_double_strict(random(), min, max);
    // auto x = _rng();
    // double norm = double(x) / double(_rng.max() + 1.0);
    // return min + (max - min) * norm;
  }

  // N.B: append. no reserve
  void randomReal(double min, double max, int nb, vector<double>& stack) {
    // uniform_real_distribution<double> distrib(min, max);
    for (int i = 0 ; i < nb; i++)
      // stack.push_back(distrib(_rng));
      stack.push_back(randomReal(min, max));
  }

  static int scale_int_strict(uint32_t x, int min, int max) {
    double norm = force_round(static_cast<double>(x) / 4294967296.0);
    double diff = force_round(max - min + 1);
    double prod = force_round(diff * norm);
    int res  = min + int(prod);
    return res;
  }

  static double force_round(double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(v));
    std::memcpy(&v, &bits, sizeof(v));
    return v;
  };

  static double scale_int_to_double_strict(uint32_t x, double min, double max) {
    double norm = force_round(static_cast<double>(x) / 4294967296.0);
    double diff = force_round(max - min);
    double prod = force_round(diff * norm);
    double res  = force_round(min + prod);
    return res;
  }

private:
  mt19937 _rng;

};

}
#endif // RANDOM_GENERATOR_H
