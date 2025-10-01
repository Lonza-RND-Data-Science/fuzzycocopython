#include <numeric>

#include "fuzzy_coco.h"
#include "file_utils.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace logging;

FuzzyCoco::FuzzyCoco(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params, RandomGenerator& rng) 
  :  _params(params),
    _fuzzy_system(dfin.colnames(), dfout.colnames(), params.input_vars_params.nb_sets, params.output_vars_params.nb_sets),
    _fuzzy_system_fitter_ptr(selectFuzzySystemFitness(params)),
    _fitter_ptr(selectFitnessMethods(dfin, dfout, params, _fuzzy_system, *_fuzzy_system_fitter_ptr)),
    _engine(dfin, dfout, *_fitter_ptr, params, rng)
{}

FuzzyCoco::FuzzyCoco(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params, const FuzzySystem& fs, RandomGenerator& rng) 
  :  _params(params),
    _fuzzy_system(fs),
    _fuzzy_system_fitter_ptr(selectFuzzySystemFitness(params)),
    _fitter_ptr(selectFitnessMethods(dfin, dfout, params, _fuzzy_system, *_fuzzy_system_fitter_ptr)),
    _engine(dfin, dfout, *_fitter_ptr, params, rng)
{}

unique_ptr<FuzzySystemFitness> 
FuzzyCoco::selectFuzzySystemFitness(const FuzzyCocoParams& params)
{
  return make_unique<FuzzySystemWeightedFitness>(params.fitness_params.metrics_weights);
}

unique_ptr<FuzzyCocoFitnessMethod> 
FuzzyCoco::selectFitnessMethods(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params, FuzzySystem& fs, 
  FuzzySystemFitness& fs_fitter)
{
  auto fitter =  params.fitness_params.features_weights.empty() ? 
    make_unique<FuzzyCocoFitnessMethod>(fs, fs_fitter, dfin, dfout, params) :
    make_unique<FuzzyCocoFeaturesWeightsFitnessMethod>(fs, fs_fitter, dfin, dfout, params);
  return fitter;
}

void FuzzyCoco::split_dataset(const DataFrame& df, int nb_out_vars, DataFrame& dfin, DataFrame& dfout)
{
  if (nb_out_vars < 1) throw runtime_error("nb_out_vars must be >= 1 !!"); 

  dfin = df.subsetColumns(0, df.nbcols() - nb_out_vars - 1);
  dfout = df.subsetColumns(df.nbcols() - nb_out_vars, df.nbcols() - 1);
}

CoevGeneration FuzzyCoco::start(RandomGenerator& rng, bool influence, double evolving_ratio)
{
  logger() << "FuzzyCoco::start(): influence=" << influence 
    << ", evolving_ratio=" << evolving_ratio << endl;
  Genomes rules = getEngine().buildRulesGenomes(getParams().rules_params.pop_size);
  Genomes mfs = getEngine().buildMFsGenomes(getParams().mfs_params.pop_size);

  if (influence) {
    influence_rules_genomes(
      rules, 
      getParams().fitness_params.asFeaturesWeightsByIdx(getFitnessMethod().getInputData().colnames()),
      getEngine().getFuzzyCocoCodec(),
      rng,
      evolving_ratio
    );
  }

  CoevGeneration cogen(
    {rules, getParams().rules_params.elite_size}, 
    {mfs, getParams().mfs_params.elite_size});
  getEngine().update(cogen);

  return cogen;
}

CoevGeneration FuzzyCoco::run(int nb, double max_fit, bool influence, double evolving_ratio) {
  return run(nb, max_fit, getEngine().getRng(), influence, evolving_ratio);
}

CoevGeneration FuzzyCoco::run(int nb, double max_fit, RandomGenerator& rng, bool influence, double evolving_ratio) {
  auto gen = start(rng, influence, evolving_ratio);
  return getEngine().run(gen, nb, max_fit);
}

CoevGeneration FuzzyCoco::run(int nb, double max_fit, CoevGeneration& from_gen) {
  return getEngine().run(from_gen, nb, max_fit);
}

NamedList FuzzyCoco::searchBestFuzzySystem(const DataFrame& df, int nb_out_vars, const FuzzyCocoParams& params, int seed)
{
  logger() << L_time << "FuzzyCoco::searchBestFuzzySystem()...\n";
  DataFrame dfin, dfout;
  split_dataset(df, nb_out_vars, dfin, dfout);

  FuzzyCocoParams fixed_params = params;
  fixed_params.fitness_params.fix_output_thresholds(nb_out_vars);

  RandomGenerator rng(seed);
  FuzzyCoco coco(dfin, dfout, fixed_params, rng);

  auto gen = coco.run();
  
  if (coco.getFitnessMethod().getBestFitness() <= 0) {
    // nothing found --> return empty results
    logger() << L_time << "FuzzyCoco::searchBestFuzzySystem(): No Fuzzy system found!!!\n";
    return NamedList();
  }
  
  coco.selectBestFuzzySystem();

  logger() << L_time << "Exiting FuzzyCoco::searchBestFuzzySystem()\n";
  return coco.describe(gen.generation_number);
}


NamedList FuzzyCoco::loadFuzzyFile(const string& fuzzy_file)
{
  ifstream in(fuzzy_file);
  return NamedList::parse(in);
}

DataFrame FuzzyCoco::loadAndPredict(const DataFrame& df, const NamedList& saved)
{
  FuzzySystem fs = FuzzySystem::load(saved["fuzzy_system"]);
  return fs.smartPredict(df);
}

DataFrame FuzzyCoco::loadAndPredict(const DataFrame& df, const string& fuzzy_file)
{
  return loadAndPredict(df, loadFuzzyFile(fuzzy_file));
}


NamedList FuzzyCoco::eval(const DataFrame& df, const FuzzySystem& fs, const FuzzyCocoParams& params)
{
  logger() << L_time << "FuzzyCoco::eval()...\n";

  RandomGenerator rng(0); // N.B: not used
  DataFrame dfin, dfout;
  split_dataset(df, fs.getDB().getNbOutputVars(), dfin, dfout);

  FuzzyCoco coco(dfin, dfout, params, fs, rng);
  return describeFit(coco.getFitnessMethod());
}

void FuzzyCoco::evalAndSave(const DataFrame& df, const string& fuzzy_file, ostream& out)
{
  NamedList desc = loadFuzzyFile(fuzzy_file);
  FuzzySystem fs = FuzzySystem::load(desc["fuzzy_system"]);
  auto params = FuzzyCocoParams(desc["params"]);
  
  auto res_lst = eval(df, fs, params);
  out << res_lst;
}

namespace fuzzy_coco {
  ostream& operator<<(ostream& out, const FuzzyCoco& coco) {
    out << "FuzzyCoco:" << endl;
    out << "--------------------------------------------------------" << endl;
    out << "# PARAMS" << endl;
    out << coco._params << endl;
    out << "# Rules Codec " << coco.getFitnessMethod().getFuzzyCocoCodec().getRulesCodec() << endl;
    out << "# MFs Codec " << coco.getFitnessMethod().getFuzzyCocoCodec().getMFsCodec() << endl;
    return out;
  }
}

NamedList FuzzyCoco::describeFit(FuzzyCocoFitnessMethod& fitter, int nb_generations) {
  auto metrics = fitter.fitMetrics();
  double fitness = fitter.fitnessImpl();

  NamedList fit;
  fit.add("fitness", fitness);
  fit.add("metrics", metrics.describe());

  if (!is_na(nb_generations))
    fit.add("generations", nb_generations);

  return fit;
}

NamedList FuzzyCoco::describe(int nb_generations) 
{
  NamedList desc;
  desc.add("fit", describeFit(getFitnessMethod(), nb_generations));
  desc.add("fuzzy_system", getFuzzySystem().describe());
  desc.add("params", getParams().describe());

  return desc;
}
void FuzzyCoco::influence_rules_genomes(
  Genomes& rules_genomes, 
  const vector<double>& features_weights, 
  FuzzyCocoCodec& codec,
  RandomGenerator& rng, 
  double evolving_ratio)
{
  logger() << L_time << "FuzzyCoco::influence_rules_genomes()" << endl;
  logger().stream() << "features_weights=" << features_weights << endl;
  auto sum = [](auto v) { return std::accumulate(v.begin(), v.end(), 0.0); };
  double sum_of_weights = sum(features_weights);
  // logger() << "sum_of_weights=" << sum_of_weights << endl;
  if (sum_of_weights <= 0) return;

  const int nb_genomes = rules_genomes.size();
  const int nb_rules = codec.getNbRules();
  const int nb_antecedents = codec.getNbRuleAntedecents();

  // process variable per variable
  for (size_t var_idx = 0; var_idx < features_weights.size(); var_idx++) {
    double weight = features_weights[var_idx];
    if (weight > 0) {
      // modify genomes proportional to the variable weight
      double prob = (weight / sum_of_weights) *  evolving_ratio;
      for (int i = 0; i < nb_genomes; i++) {
        if (rng.randomReal(0, 1) <= prob) {
          int rule_idx = rng.random(0, nb_rules - 1);
          int ant_idx = rng.random(0, nb_antecedents - 1);
          // force that antecedent ant_idx of that rule rule_idx of that genome i to use our var var_idx
          logger() << "FuzzyCoco::influence_rules_genomes(): setting var_idx=" << var_idx 
             << " to ant_idx " << ant_idx 
             << " of rule_idx " << rule_idx
             << " of genome " << i << endl;
          codec.modifyRuleAntecedent(rules_genomes[i], rule_idx, ant_idx, var_idx);
        }
      }
    }
  }
}

