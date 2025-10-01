#ifndef GENOME_CODEC_H
#define GENOME_CODEC_H

#include "bitarray.h"
#include "genome.h"
#include "fuzzy_rule.h"
#include "discretizer.h"

namespace fuzzy_coco {

using namespace BitArrayUtils;

class IntCodec {
public:
    IntCodec(int nb_bits) : _nb_bits(nb_bits) {}
    int decode(BitArray::const_iterator& bits) const { 
        int decoded = decode_number(bits, _nb_bits);
        bits += _nb_bits;
        return decoded; 
    }
    void encode(int number, BitArray::iterator& bits) const { 
        encode_number(number, bits, _nb_bits); 
        bits += _nb_bits;
    }

    int size() const { return _nb_bits; }

    int getNumberOfBits() const { return _nb_bits; }

protected:
    int _nb_bits;
};

class IntVectorCodec {
public:
    IntVectorCodec(int nb, int nb_bits) : _codec(nb_bits), _nb(nb) {}
    void decode(BitArray::const_iterator& bits, vector<int>& res) const { 
        res.resize(_nb);
        for (size_t i = 0; i < _nb; i++)
            res[i] = _codec.decode(bits);
    }
    void encode(const vector<int>& res, BitArray::iterator& bits) const {
        assert(res.size() == _nb);
        for (int v : res) _codec.encode(v, bits);
    }

    int getNumberOfBits() const { return _codec.getNumberOfBits(); }
    int getNumberOfElements() const { return _nb; }

    int size() const { return _codec.size() * _nb; }
protected:
    IntCodec _codec;
    size_t _nb;
};

// codec for a list of (i1 ,i2) pairs, encoded each with its own number of bits
class IntPairCodec {
public:
    IntPairCodec(int nb_bits1, int nb_bits2) 
        : _codec1(nb_bits1), _codec2(nb_bits2) {}
    void encode(int i1, int i2, BitArray::iterator& bits) const {
        _codec1.encode(i1, bits);
        _codec2.encode(i2, bits);
    }

    void decode(BitArray::const_iterator& bits, int& i1, int& i2) const { 
        i1 = _codec1.decode(bits);
        i2 = _codec2.decode(bits);
    }

    int getNumberOfBits1() const { return _codec1.getNumberOfBits(); }
    int getNumberOfBits2() const { return _codec2.getNumberOfBits(); }

    int size() const { return _codec1.size() + _codec2.size(); }

protected:
    IntCodec _codec1;
    IntCodec _codec2;  
};

class ConditionIndexCodec {
public:
    ConditionIndexCodec(int var_idx_nb_bits, int set_idx_nb_bits) 
        : _codec(var_idx_nb_bits, set_idx_nb_bits) {}

    void encode(const ConditionIndex& ci, BitArray::iterator& bits) const {
        _codec.encode(ci.var_idx, ci.set_idx, bits);
    }

    void encode(const ConditionIndexes& cis, BitArray::iterator& bits) const {
        for (const auto& ci: cis)
            encode(ci, bits);
    }

    void decode(BitArray::const_iterator& bits, ConditionIndex& res) const { 
        _codec.decode(bits, res.var_idx, res.set_idx);
    }

    ConditionIndex decode(BitArray::const_iterator& bits) const {
        ConditionIndex ci;
        decode(bits, ci);
        return ci;
    }

    void decode(BitArray::const_iterator& bits, ConditionIndexes& res, int nb) const {
        res.resize(nb);
        for (int i = 0; i < nb; i++) {
            decode(bits, res[i]);
        }
    }

    int getNumberofVarIndexBits() const { return _codec.getNumberOfBits1(); }
    int getNumberofSetIndexBits() const { return _codec.getNumberOfBits2(); }

    int size() const { return _codec.size();}

protected:
    IntPairCodec _codec;
};


struct IntPairParams {
    int nb;
    int nb_bits1;
    int nb_bits2;

    IntPairParams(int nb, int nb_bits1, int nb_bits2) {
        this->nb = nb;
        this->nb_bits1 = nb_bits1;
        this->nb_bits2 = nb_bits2;
    }
};

class ConditionIndexesCodec {
public:
    ConditionIndexesCodec(const IntPairParams& params) 
        : _codec(params.nb_bits1, params.nb_bits2), _nb(params.nb) {}

    void encode(const ConditionIndexes& cis, BitArray::iterator& bits) const {
        assert(cis.size() == _nb);
        _codec.encode(cis, bits);
    }

    void decode(BitArray::const_iterator& bits, ConditionIndexes& res) const {
        _codec.decode(bits, res, _nb);
    }

    int getNumberofVarIndexBits() const { return _codec.getNumberofVarIndexBits(); }
    int getNumberofSetIndexBits() const { return _codec.getNumberofSetIndexBits(); }
    int getNumberOfConditions() const { return _nb; }

    int size() const { return _nb * _codec.size(); }

protected:
    ConditionIndexCodec _codec;
    size_t _nb;
};



class RuleCodec {
public:
    RuleCodec(const IntPairParams& params_input, const IntPairParams& params_output) 
        : _codec_in(params_input), _codec_out(params_output) {}

    void decode(BitArray::const_iterator& bits, ConditionIndexes& cis_in, ConditionIndexes& cis_out) const 
    {
        _codec_in.decode(bits, cis_in);
        _codec_out.decode(bits, cis_out);
    }

    void encode(const ConditionIndexes& cis_in, const ConditionIndexes& cis_out, BitArray::iterator& bits) const 
    {
        _codec_in.encode(cis_in, bits);
        _codec_out.encode(cis_out, bits);
    }

    int size() const { return _codec_in.size() + _codec_out.size(); }

    inline friend ostream& operator<<(ostream& out, const RuleCodec& codec) {
        out << "nb_input_vars=" << codec._codec_in.getNumberOfConditions()
        << ", " << "nb_output_vars=" << codec._codec_out.getNumberOfConditions();
        return out;
    }

private:
    ConditionIndexesCodec _codec_in;
    ConditionIndexesCodec _codec_out;
};

class RulesCodec {
public:
    RulesCodec(int nb_rules, const IntPairParams& params_input, const IntPairParams& params_output) 
        : _codec(params_input, params_output), _codec_default_rules(params_output.nb, params_output.nb_bits2), _nb_rules(nb_rules) {}

    int getNbRules() const { return _nb_rules; }

    // convenience methods
    void decode(const BitArray& bits, vector<ConditionIndexes>& rules_in, vector<ConditionIndexes>& rules_out, vector<int>& default_rules) const {
        auto it = bits.cbegin();
        decode(it, rules_in, rules_out, default_rules);
    }
    void encode(const vector<ConditionIndexes>& rules_in, const vector<ConditionIndexes>& rules_out, const vector<int>& default_rules,
         BitArray& bits) const 
    {
        auto it = bits.begin();
        encode(rules_in, rules_out, default_rules, it);
    }

    void decode(BitArray::const_iterator& bits, vector<ConditionIndexes>& rules_in, vector<ConditionIndexes>& rules_out, vector<int>& default_rules) const;

    void encode(const vector<ConditionIndexes>& rules_in, const vector<ConditionIndexes>& rules_out, const vector<int>& default_rules, 
        BitArray::iterator& bits) const;

    int size() const { return _codec.size() * getNbRules() + _codec_default_rules.size(); }

    inline friend ostream& operator<<(ostream& out, const RulesCodec& codec) {
        out << "RulesCodec: [" 
        << "nb_rules=" << codec._nb_rules 
        << ", " << codec._codec 
        << ", " << "nb default rules=" << codec._codec_default_rules.getNumberOfElements();
        return out;
    }

private:
    RuleCodec _codec;
    IntVectorCodec _codec_default_rules;
    int _nb_rules;
};

struct PosParams {
    int nb_vars;
    int nb_sets;
    int nb_bits;

    PosParams(int nb_vars, int nb_sets, int nb_bits) {
        this->nb_vars = nb_vars;
        this->nb_sets = nb_sets;
        this->nb_bits = nb_bits;
    }
};

// class FuzzySystemSetPositionsCodec {
// public:
//     FuzzySystemSetPositionsCodec(const PosParams& input, const PosParams& output)
//         : _codec_in(input.nb_sets, input.nb_bits), 
//         _codec_out(output.nb_sets, output.nb_bits),
//         _nb_input_vars(input.nb_vars), _nb_output_vars(output.nb_vars)
//          {}

//     int getNbInputVars() const { return _nb_input_vars; }
//     int getNbOutputVars() const { return _nb_output_vars; }



//     void decode(BitArray::const_iterator& bits, Matrix<int>& pos_in, Matrix<int>& pos_out) const;
//     void encode(const Matrix<int>& pos_in, const Matrix<int>& pos_out, BitArray::iterator& bits) const;

//     int size() const { return _codec_in.size() * getNbInputVars() + _codec_out.size() * getNbOutputVars(); }

// private:
//     int _nb_input_vars;
//     int _nb_output_vars;
//     IntVectorCodec _codec_in;
//     IntVectorCodec _codec_out;
// };


class DiscretizedFuzzySystemSetPositionsCodec {
public:
    DiscretizedFuzzySystemSetPositionsCodec(const PosParams& input, const PosParams& output, 
        const vector<Discretizer>& disc_in, const vector<Discretizer>& disc_out);

    int getNbInputVars() const { return _input_params.nb_vars; }
    int getNbOutputVars() const { return _output_params.nb_vars; }

    // convenience methods
    void decode(const BitArray& bits, Matrix<double>& pos_in, Matrix<double>& pos_out) const {
        auto it = bits.cbegin();
        decode(it, pos_in, pos_out);
    }
    void encode(const Matrix<double>& pos_in, const Matrix<double>& pos_out, BitArray& bits) const {
        auto it = bits.begin();
        encode(pos_in, pos_out, it);
    }

    void decode(BitArray::const_iterator& it, Matrix<double>& pos_in, Matrix<double>& pos_out) const;

    void encode(const Matrix<double>& pos_in, const Matrix<double>& pos_out, BitArray::iterator& it) const;

    int size() const { return _codec_in.size() * getNbInputVars() * _input_params.nb_sets + _codec_out.size() * getNbOutputVars() * _output_params.nb_sets; }

    inline friend ostream& operator<<(ostream& out, const DiscretizedFuzzySystemSetPositionsCodec& codec) {
        out << "DiscretizedFuzzySystemSetPositionsCodec: [" 
        << "getNbInputVars=" << codec.getNbInputVars() 
        << ", " << "getNbOutputVars=" << codec.getNbOutputVars() 
        << "]";
        return out;
    }

protected:
    static void encode(const Matrix<double>& pos, BitArray::iterator& bits, const vector<Discretizer>& discs, const IntCodec& codec);
    static void decode(BitArray::const_iterator& bits, Matrix<double>& pos,  const vector<Discretizer>& discs, const IntCodec& codec);

private:
    PosParams _input_params;
    PosParams _output_params;
    IntCodec _codec_in;
    IntCodec _codec_out;
    vector<Discretizer> _disc_in;
    vector<Discretizer> _disc_out;
};

}
#endif // GENOME_CODEC_H