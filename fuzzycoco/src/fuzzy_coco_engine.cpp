#include "fuzzy_coco_engine.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace logging;


FuzzyCocoEngine::FuzzyCocoEngine(const DataFrame& dfin, const DataFrame& dfout, FuzzyCocoFitnessMethod& fitter,
  const FuzzyCocoParams& params, RandomGenerator& rng) 
  :  _params(params), 
  _rng(rng),
  _fitter(fitter),
  _coev_engine(_fitter,  _rules_engine, _mfs_engine, params.global_params.nb_cooperators),
  _rules_engine(params.rules_params, rng),
  _mfs_engine(params.mfs_params, rng)
{
  if (params.has_missing()) throw runtime_error("ERROR: some parameters in params are missing!");
}//KCOV IGNORE


namespace fuzzy_coco {
  ostream& operator<<(ostream& out, const FuzzyCocoEngine& cocoe) {
    out << "FuzzyCoco:" << endl;
    out << "--------------------------------------------------------" << endl;
    out << "# PARAMS" << endl;
    out << cocoe._params << endl;
    out << "# Rules Codec " << cocoe.getFuzzyCocoCodec().getRulesCodec() << endl;
    out << "# MFs Codec " << cocoe.getFuzzyCocoCodec().getMFsCodec() << endl;
    return out;
  }
}

FuzzySystem& FuzzyCocoEngine::rebuildBestFuzzySystem() 
{
  logger() << "FuzzyCocoEngine::rebuildBestFuzzySystem(), best fitness=" << getFitnessMethod().getBestFitness() << endl;
  auto [best_rule, best_mf] = getBest();
  getFitnessMethod().setRulesGenome(best_rule);
  getFitnessMethod().setMFsGenome(best_mf);

  return getFitnessMethod().getFuzzySystem();
}

NamedList FuzzyCocoEngine::describeBestFuzzySystem() 
{
  auto& fs = rebuildBestFuzzySystem();
  auto [best_rule, best_mf] = getBest();
  double fitness = getFitnessMethod().fitnessImpl(best_rule, best_mf);
  NamedList desc;
  desc.add("fitness", fitness);
  auto mw = getParams().fitness_params.metrics_weights.describe();
  desc.add("fitness_metrics_weights", mw);

  const auto& thresholds = getParams().fitness_params.output_vars_defuzz_thresholds;
  NamedList thresh;
  const auto& db = fs.getDB();
  const int nb_out_vars = db.getNbOutputVars();
  assert((size_t)nb_out_vars == thresholds.size());
  for (int i = 0; i < nb_out_vars; i++)
    thresh.add(db.getOutputVariable(i).getName(), thresholds[i]);
  
  desc.add("fuzzy_system", fs.describe());
  desc.add("defuzz_thresholds", thresh);

  return desc;
}

Genomes FuzzyCocoEngine::buildRulesGenomes(int nb_pop_rules) {
  Genomes rules;
  rules.reserve(nb_pop_rules);
  for (int i = 0; i < nb_pop_rules; i++) {
    Genome rules_geno = getFuzzyCocoCodec().buildRulesGenome();
    randomize(rules_geno, _rng);

    rules.push_back(rules_geno);
  }
  return rules;
}//KCOV IGNORE

Genomes FuzzyCocoEngine::buildMFsGenomes(int nb_pop_mfs) {
  Genomes mfs;
  mfs.reserve(nb_pop_mfs);
  for (int i = 0; i < nb_pop_mfs; i++) {
    Genome mf = getFuzzyCocoCodec().buildMFsGenome();
    randomize(mf, _rng);
    mfs.push_back(mf);
  }

  return mfs;
}//KCOV IGNORE

CoevGeneration FuzzyCocoEngine::start(int nb_pop_rules, int nb_pop_mfs) {
  Genomes rules = buildRulesGenomes(nb_pop_rules);
  Genomes mfs = buildMFsGenomes(nb_pop_mfs);
  return start(rules, mfs);
}

CoevGeneration FuzzyCocoEngine::start(const Genomes& rules, const Genomes& mfs) {
  CoevGeneration cogen(
    {rules, getParams().rules_params.elite_size}, 
    {mfs, getParams().mfs_params.elite_size});
  update(cogen);
  return cogen;
}//KCOV IGNORE

void FuzzyCocoEngine::update(CoevGeneration& cogen) {
  getCoevolutionEngine().updateGeneration(cogen);
}

CoevGeneration FuzzyCocoEngine::next(const CoevGeneration& cogen) {
  return getCoevolutionEngine().nextGeneration(cogen);
}

CoevGeneration FuzzyCocoEngine::run() { 
  auto gen = start();
  return run(
    gen,
    getParams().global_params.max_generations, 
    getParams().global_params.max_fitness);
}

CoevGeneration FuzzyCocoEngine::run(CoevGeneration& gen, int nb, double max_fit) {
  // auto gen = start();

  rebuildBestFuzzySystem(); // TODO: remove that
  logger() << "initial rules: " << FuzzyRule::describeRules(getFuzzySystem().getRules());

  logger() << L_time << "FuzzyCocoEngine::run() " << nb << " generations" << ", max_fit=" << max_fit << endl;
  for (int i = 0; i < nb; i++) {
    gen = next(gen);
    logger() << L_time << "generation " << (i+1) << ": fitness=" << gen.fitness << endl;
    // logger() << describeBestFuzzySystem();
    if (gen.fitness >= max_fit) break;
  }
  logger() << endl;
  return gen;
}

