// a simple and fast portable digest implementation based on FNV-1a (https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function)
// intended for testing

#ifndef DIGEST_H
#define DIGEST_H

#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <iomanip>
#include "coevolution_engine.h"

namespace fuzzy_coco {
  using namespace std;
  namespace Digest {

    constexpr uint64_t FNV1A_64_OFFSET_BASIS = 14695981039346656037ULL;
    constexpr uint64_t FNV1A_64_PRIME        = 1099511628211ULL;

    // cf boost::hash_combine
    inline uint64_t hash_combine(uint64_t seed, uint64_t hash)
    {
      seed ^= hash + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      return seed;
    }

    inline string uint64_to_hex(uint64_t value) {
        ostringstream oss;
        oss << hex << setfill('0') << setw(16) << value;
        return oss.str();
    }

    inline uint64_t double_to_uint64(double value) {
      uint64_t bits;
      memcpy(&bits, &value, sizeof(value));
      return bits;
    }

    inline double uint64_to_double(uint64_t bits) {
      double value;
      memcpy(&value, &bits, sizeof(value));
      return value;
    }

    inline string double_to_hex(double value) {
      return uint64_to_hex(double_to_uint64(value));
    }

    // Convert hex → double
    inline uint64_t hex_to_uint64(const string& hex) {
      uint64_t bits = 0;
      istringstream iss(hex);
      iss >> std::hex >> bits;
      return bits;
    }

    // Convert hex → double
    inline double hex_to_double(const string& hex) {
      return uint64_to_double(hex_to_uint64(hex));
    }

    // FNV-1a 64-bit hash for arbitrary data
    inline uint64_t fnv1a64(const uint8_t* data, size_t size) {
      uint64_t hash = FNV1A_64_OFFSET_BASIS;
      for (size_t i = 0; i < size; ++i) {
        hash ^= data[i]; // Bitwise XOR 
        hash *= FNV1A_64_PRIME; // * by FNV prime
      }
      return hash;
    }

    inline uint64_t hash_string(const string& s) {
      return fnv1a64(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }

    // Combine hashes of all strings in the vector
    inline uint64_t hash_vector(const vector<string>& v) {
      uint64_t vector_hash = FNV1A_64_OFFSET_BASIS;
      for (const auto& s : v) {
          uint64_t hash = hash_string(s);
          // Mix into cumulative hash
          vector_hash = hash_combine(vector_hash, hash);
      }
      return vector_hash;
    }

    inline uint64_t hash_vector(const vector<double>& v) {
      uint64_t vector_hash = FNV1A_64_OFFSET_BASIS;
      for (const auto& s : v) {
          uint64_t hash = hash_string(double_to_hex(s));
          // Mix into cumulative hash
          vector_hash = hash_combine(vector_hash, hash);
      }
      return vector_hash;
    }
    
    inline string digest(const string& s) {
      return uint64_to_hex(hash_string(s));
    }

    inline string digest(const vector<string>& v) {
      return uint64_to_hex(hash_vector(v));
    }

    inline string digest(double x) {
      return double_to_hex(x);
    }

    inline string digest(const vector<double>& v) {
      return uint64_to_hex(hash_vector(v));
    }


    inline string digest(const Genome& genome) {
      size_t n = genome.size();
      size_t byte_count = (n + 7) / 8; // round up to full bytes

      ostringstream oss;
      for (size_t byte_idx = 0; byte_idx < byte_count; ++byte_idx) {
        uint8_t byte = 0;
        for (size_t bit = 0; bit < 8; ++bit) {
          size_t idx = byte_idx * 8 + bit;
          if (idx < n && genome[idx]) {
              byte |= (1u << bit); // pack bit (LSB first)
          }
        }
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
      }
      return oss.str();
    }

    inline string digest(const Genomes& genomes) {
      vector<string> hexs;
      hexs.resize(genomes.size());
      for (size_t i = 0; i < genomes.size(); i++)
        hexs[i] = digest(genomes[i]);
      return digest(hexs);
    }

    inline string digest(const Generation& gen) {
      vector<string> hashes;
      hashes.reserve(3);
      hashes.push_back(digest(gen.individuals));
      hashes.push_back(digest(gen.elite));
      hashes.push_back(uint64_to_hex(hash_string(double_to_hex(gen.fitness))));

      return digest(hashes);
    }

    inline string digest(const CoevGeneration& cogen) {
      vector<string> hashes;

      hashes.reserve(3);
      hashes.push_back(digest(cogen.left_gen));
      hashes.push_back(digest(cogen.right_gen));
      hashes.push_back(uint64_to_hex(hash_string(double_to_hex(cogen.fitness))));

      return digest(hashes);
    }

  }
}


#endif // DIGEST_H