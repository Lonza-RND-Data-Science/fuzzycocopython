#ifndef DISCRETIZER_H
#define DISCRETIZER_H

#include <cmath>
using namespace std;

#include "types.h"

namespace fuzzy_coco {

class Discretizer {
public:
    // N.B: support for MISSING DATA
    Discretizer(int nb_bits, const vector<double>& data) {
        const int nb = data.size();
        double maxv = numeric_limits<double>::lowest();
        double minv = numeric_limits<double>::max();
        bool all_missing = true;
        double v = -1;
        for (int i = 0; i < nb; i++) {
            v = data[i];
            if (!is_na(v)) {
                all_missing = false;
                maxv = max(maxv, v);
                minv = min(minv, v);
            }
        }
        if (all_missing) throw runtime_error("empty or all missing data in Discretizer()!");
        reset(nb_bits, minv, maxv);
    }

    Discretizer(int nb_bits, double min, double max) {
        reset(nb_bits, min, max);
    }

    void reset(int nb_bits, double min, double max) {
        _nb_bits = nb_bits;
        _min = min;
        _max = max;
        _step = (max - min) / ( (1 << nb_bits) - 1);
    }

    int discretize(double value) const {
        if (_step == 0) return 0; // constant
        return lround((value - _min) / _step); 
    }

    double undiscretize(int discret) const {
        return _min + discret * _step;
    }

    double getStep() const { return _step; }

    bool operator==(const Discretizer& ds) const { 
        return _min == ds._min && _max == ds._max && _nb_bits == ds._nb_bits && _step == ds._step; }

    inline friend ostream& operator<<(ostream& out, const Discretizer& ds) {
        out << "Discretizer: [" << ds._min << "-" << ds._max << "] by " 
        << ds._step << " on " << ds._nb_bits;
    return out;
    }

private:
    double _min;
    double _max;
    int _nb_bits;
    double _step;
};

}
#endif // DISCRETIZER_H
