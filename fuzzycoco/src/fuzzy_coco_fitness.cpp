#include <numeric>

#include "fuzzy_coco_fitness.h"
#include "fuzzy_coco.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace logging;

// FuzzyCocoFitnessMethod::FuzzyCocoFitnessMethod(const DataFrame &dfin, const DataFrame &dfout, const FuzzyCocoParams& params)
//     : FuzzyCocoFitnessMethod(
//         make_unique<FuzzySystemWeightedFitness>(params.fitness_params.metrics_weights), 
//         dfin, dfout, params)
// {}

FuzzyCocoFitnessMethod::FuzzyCocoFitnessMethod(FuzzySystem& fs,FuzzySystemFitness& fitter,
                                               const DataFrame &dfin, const DataFrame &dfout, const FuzzyCocoParams& params)
  : CoopCoevolutionFitnessMethod(),  
      _actual_dfin(dfin),_actual_dfout(dfout), 
      _fuzzy_system(fs),
      _codec(dfin, dfout, params),
      _fsmc(), 
      _fitter(fitter), 
      _thresholds(params.fitness_params.output_vars_defuzz_thresholds)
{
  assert(_thresholds.size() == (size_t)dfout.nbcols());
}//KCOV IGNORE

bool FuzzyCocoFitnessMethod::resetFuzzySystem(const Genome& rules_genome, const Genome& vars_genome)
{

  setRulesGenome(rules_genome);
  setMFsGenome(vars_genome);

  bool status = getFuzzySystem().ok();
  // logger() << "FuzzyCocoFitnessMethod::resetFuzzySystem() : status=" << status << endl;
  return status;
}

double FuzzyCocoFitnessMethod::computeFuzzySystemFitness(FuzzySystem& fs, FuzzySystemFitness& fitter, const FuzzySystemMetrics& metrics, double extra_num, double extra_denum) 
{ 
  if (!fs.ok()) return 0;
  return fitter.fitness(metrics, extra_num, extra_denum);
}

double FuzzyCocoFitnessMethod::fitnessImpl(const Genome& rules_genome, const Genome& mfs_genome) 
{
  if (!resetFuzzySystem(rules_genome, mfs_genome)) 
    return 0; 
  return fitnessImpl();
}

double FuzzyCocoFitnessMethod::fitnessImpl() {
  return computeFuzzySystemFitness(getFuzzySystem(), getFuzzySystemFitness(), fitMetrics(), 0, 0.0);
}

FuzzySystemMetrics FuzzyCocoFitnessMethod::fitMetrics() 
{
  auto predicted_output = getFuzzySystem().predict(_actual_dfin);
  return computeMetrics(predicted_output, _actual_dfout);
}


FuzzySystemMetrics FuzzyCocoFitnessMethod::computeMetrics(const DataFrame& predicted, const DataFrame& actual) 
{
  auto metrics = _fsmc.compute(predicted, actual, _thresholds);
    
  // VERY IMPORTANT FOR NOW: need to add the number of variables used in the rules
  metrics.nb_vars = getFuzzySystem().computeTotalInputVarsUsedInRules();

  return metrics;
}


FuzzyCocoFeaturesWeightsFitnessMethod::FuzzyCocoFeaturesWeightsFitnessMethod(
  FuzzySystem& fs,
  FuzzySystemFitness& fitter,
  const DataFrame& dfin, 
  const DataFrame& dfout, 
  const FuzzyCocoParams& params)
: FuzzyCocoFitnessMethod(fs, fitter, dfin, dfout, params),
  _features_weights(params.fitness_params.asFeaturesWeightsByIdx(dfin.colnames()))
{
  _sum_of_weights = std::accumulate(_features_weights.begin(), _features_weights.end(), double(0));
}

double FuzzyCocoFeaturesWeightsFitnessMethod::fitnessImpl(const Genome& rules_genome, const Genome& mfs_genome) 
{
  if (!resetFuzzySystem(rules_genome, mfs_genome)) return 0.0;
  return fitnessImpl();
}

double FuzzyCocoFeaturesWeightsFitnessMethod::fitnessImpl() {
  // fill _used_features_static with the input variables usage
  getFuzzySystem().fetchInputVariablesUsage(_used_features_static);
  
  // IMPORTANT: if a mandatory input variable is not used by the system --> 0
  if (mandatoryFeatureNotUsed(_used_features_static, getFeaturesWeights())) return 0.0;
  return computeFuzzySystemFitness(getFuzzySystem(), getFuzzySystemFitness(), fitMetrics(),
    sumOfUsedFeaturesWeights(_used_features_static, getFeaturesWeights()), getSumOfFeaturesWeights());
}


double FuzzyCocoFeaturesWeightsFitnessMethod::sumOfUsedFeaturesWeights(const vector<bool>& used_features, const vector<double>& weights)
{
  const int nb = weights.size();
  double sumw = 0;
  for (int i = 0; i < nb; i++)
    if (used_features[i])
    sumw += weights[i];
  
  return sumw;
}

bool FuzzyCocoFeaturesWeightsFitnessMethod::mandatoryFeatureNotUsed(const vector<bool>& used_features, const vector<double>& weights)
{
  const int nb = weights.size();
  for (int i = 0; i < nb; i++)
    if (!used_features[i] && weights[i] >= 1)
      return true;
  
  return false;
}
