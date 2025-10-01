#ifndef FUZZY_COCO_SYSTEM_H
#define FUZZY_COCO_SYSTEM_H

#include "fuzzy_coco_params.h"
#include "fuzzy_system.h"
#include "genome_codec.h"
#include "discretizer.h"

namespace fuzzy_coco {
// extension of FuzzySystem to deal with the Rules and Genomes encoding
// will be used by FuzzyCocoFitnessMethod to compute the fitness of (rule, mfs) genomes pair
class FuzzyCocoCodec {
public:
  // complete ctor
  FuzzyCocoCodec(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params);

public: // main FuzzyCocoCodec interface
  Genome buildRulesGenome() { return Genome(getRulesCodec().size()); }
  Genome buildMFsGenome() { return Genome(getMFsCodec().size()); }

  void setRulesGenome(FuzzySystem& fs, const Genome& rules_genome);
  void setMFsGenome(FuzzySystem& fs, const Genome& mfs_genome);

public:
  // accessors

  // get the number of antecedents in a rule
  int getNbRuleAntedecents() const { return _nb_rules_antecedents; }
  int getNbRules() const { return getRulesCodec().getNbRules(); };

  RulesCodec& getRulesCodec() { return _rules_codec; }
  const RulesCodec& getRulesCodec() const { return _rules_codec; }
  DiscretizedFuzzySystemSetPositionsCodec& getMFsCodec() { return *_vars_codec_ptr; }
  const DiscretizedFuzzySystemSetPositionsCodec& getMFsCodec() const { return *_vars_codec_ptr; }

  static vector<Discretizer> createDiscretizersForData(const DataFrame& df, int nb_bits);

public: // genome / population influence related to features weight
  // modify in-place the rule #rule_idx so that its antecedent #ant_idx is set to var_idx
  void modifyRuleAntecedent(Genome& rules_genome, int rule_idx, int ant_idx, int var_idx);
  
public: // utils, mainly for testing and debugging
  Genome encode(const vector<FuzzyRule>& rules, const vector<int>& default_rules);
  Genome encode(const vector<FuzzyRule>& rules, const vector<FuzzyDefaultRule>& default_rules);
  Genome encode(const vector<FuzzyRule>& rules, const FuzzyDefaultRule& default_rule);
  void encode(const vector<ConditionIndexes>& in_cis, const vector<ConditionIndexes>& out_cis, 
    const vector<int>& default_rules, Genome& rules_geno);
  Genome encode(const Matrix<double>& pos_in, const Matrix<double>& pos_out);
  void decode(const Genome& mfs_genome, Matrix<double>& pos_in, Matrix<double>& pos_out);
  void decode(const Genome& rules_genome, vector<ConditionIndexes>& rules_in, vector<ConditionIndexes>& rules_out,
    vector<int>& default_rules);
  // void decode(const Genome& rules_genome, const FuzzyVariablesDB& db, vector<FuzzyRule>& rules, vector<FuzzyDefaultRule>& default_rules);
private:
  void init(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params);

private:
  int _nb_rules_antecedents;
  // the genome codecs
  RulesCodec _rules_codec;
  unique_ptr<DiscretizedFuzzySystemSetPositionsCodec> _vars_codec_ptr;

  // internal state - to avoid reallocation
  vector<ConditionIndexes> _rules_in;
  vector<ConditionIndexes> _rules_out;
  vector<int> _default_rules;
  Matrix<double> _pos_in, _pos_out;
};

}
#endif // FUZZY_COCO_SYSTEM_H
