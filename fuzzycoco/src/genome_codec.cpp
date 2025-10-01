#include "genome_codec.h"
using namespace fuzzy_coco;

void RulesCodec::decode(BitArray::const_iterator& bits, vector<ConditionIndexes>& rules_in, vector<ConditionIndexes>& rules_out, vector<int>& default_rules) const 
{
    rules_in.resize(_nb_rules);
    rules_out.resize(_nb_rules);
    for (int i = 0; i < _nb_rules; i++) _codec.decode(bits, rules_in[i], rules_out[i]);
    _codec_default_rules.decode(bits, default_rules);
}

void RulesCodec::encode(const vector<ConditionIndexes>& rules_in, const vector<ConditionIndexes>& rules_out, const vector<int>& default_rules, 
    BitArray::iterator& bits) const 
{
    for (int i = 0; i < _nb_rules; i++) _codec.encode(rules_in[i], rules_out[i], bits);
    _codec_default_rules.encode(default_rules, bits);
}

// void FuzzySystemSetPositionsCodec::decode(BitArray::const_iterator& bits, Matrix<int>& pos_in, Matrix<int>& pos_out) const 
// {
//     pos_in.resize(_nb_input_vars);
//     for (int i = 0; i < _nb_input_vars; i++)
//         _codec_in.decode(bits, pos_in[i]);

//     pos_out.resize(_nb_output_vars);
//     for (int i = 0; i < _nb_output_vars; i++)
//         _codec_out.decode(bits, pos_out[i]);
// }

// void FuzzySystemSetPositionsCodec::encode(const Matrix<int>& pos_in, const Matrix<int>& pos_out, BitArray::iterator& bits) const 
// {
//     assert(pos_in.size() == _nb_input_vars);
//     assert(pos_out.size() == _nb_output_vars);
//     for (int i = 0; i < _nb_input_vars; i++)
//         _codec_in.encode(pos_in[i], bits);

//     for (int i = 0; i < _nb_output_vars; i++)
//         _codec_out.encode(pos_out[i], bits);        
// }

DiscretizedFuzzySystemSetPositionsCodec::DiscretizedFuzzySystemSetPositionsCodec(
    const PosParams& input, const PosParams& output, 
    const vector<Discretizer>& disc_in, const vector<Discretizer>& disc_out)
    :   
        _input_params(input),
        _output_params(output),
        _codec_in(input.nb_bits), 
        _codec_out(output.nb_bits),
        _disc_in(disc_in), 
        _disc_out(disc_out)
{}


void DiscretizedFuzzySystemSetPositionsCodec::decode(BitArray::const_iterator& bits, Matrix<double>& pos_in, Matrix<double>& pos_out) const 
{
    pos_in.redim(getNbInputVars(), _input_params.nb_sets);
    decode(bits, pos_in, _disc_in, _codec_in);

    pos_out.redim(getNbOutputVars(), _output_params.nb_sets);
    decode(bits, pos_out, _disc_out, _codec_out);
}
 
void DiscretizedFuzzySystemSetPositionsCodec::decode(BitArray::const_iterator& bits, Matrix<double>& pos,  const vector<Discretizer>& discs, const IntCodec& codec)
{
    const int nbrows = pos.nbrows();
    const int nbcols = pos.nbcols();
    for (int i = 0; i < nbrows; i++) {
        auto& row = pos[i];
        const auto& discretizer = discs[i];
        for (int j = 0; j < nbcols; j++)
            row[j] = discretizer.undiscretize(codec.decode(bits));
    }   
}

void DiscretizedFuzzySystemSetPositionsCodec::encode(const Matrix<double>& pos, BitArray::iterator& bits, const vector<Discretizer>& discs, const IntCodec& codec) {
    const int nbrows = pos.nbrows();
    const int nbcols = pos.nbcols();
    for (int i = 0; i < nbrows; i++) {
        const auto& row = pos[i];
        const auto& discretizer = discs[i];
        for (int j = 0; j < nbcols; j++) 
            codec.encode(discretizer.discretize(row[j]), bits); 
    }
}

void DiscretizedFuzzySystemSetPositionsCodec::encode(const Matrix<double>& pos_in, const Matrix<double>& pos_out, BitArray::iterator& bits) const 
{
    assert(pos_in.nbrows() == getNbInputVars());
    assert(pos_in.nbcols() == _input_params.nb_sets);
    assert(pos_out.nbrows() == getNbOutputVars());
    assert(pos_out.nbcols() == _output_params.nb_sets);

    encode(pos_in, bits, _disc_in, _codec_in);
    encode(pos_out, bits, _disc_out, _codec_out);
}