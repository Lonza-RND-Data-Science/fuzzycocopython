#include "fuzzy_system_metrics.h"
using namespace fuzzy_coco;

FuzzySystemMetrics::FuzzySystemMetrics(const NamedList& desc) : FuzzySystemMetrics() {
  setValues(desc);
}

void FuzzySystemMetrics::setValues(const NamedList& desc)
{
  sensitivity = desc.get_double("sensitivity", sensitivity);
  specificity = desc.get_double("specificity", specificity);
  accuracy = desc.get_double("accuracy", accuracy);
  ppv = desc.get_double("ppv", ppv);
  rmse = desc.get_double("rmse", rmse);
  rrse = desc.get_double("rrse", rrse);
  rae = desc.get_double("rae", rae);
  mse = desc.get_double("mse", mse);
  distanceThreshold = desc.get_double("distanceThreshold", distanceThreshold);
  distanceMinThreshold = desc.get_double("distanceMinThreshold", distanceMinThreshold);
  nb_vars = desc.get_double("nb_vars", nb_vars);
  overLearn = desc.get_double("overLearn", overLearn);
  true_positives = desc.get_double("true_positives", true_positives);
  false_positives = desc.get_double("false_positives", false_positives);
  true_negatives = desc.get_double("true_negatives", true_negatives);
  false_negatives = desc.get_double("false_negatives", false_negatives);
}


void FuzzySystemMetrics::reset() {
  true_positives = false_positives = true_negatives = false_negatives = 0;
  sensitivity = 0;
  specificity = 0;
  accuracy = 0;
  ppv = 0;
  rmse = 0;
  rrse = 0;
  rae = 0;
  mse = 0;
  distanceThreshold = 0;
  distanceMinThreshold = 0;
  nb_vars = 0;
  overLearn = 0;
}

bool FuzzySystemMetrics::operator==(const FuzzySystemMetrics& p) const {
  return 
      sensitivity == p.sensitivity && 
      specificity == p.specificity && 
      accuracy == p.accuracy && 
      ppv == p.ppv && 
      rmse == p.rmse && 
      rrse == p.rrse && 
      rae == p.rae && 
      mse == p.mse && 
      distanceThreshold == p.distanceThreshold && 
      distanceMinThreshold == p.distanceMinThreshold && 
      nb_vars == p.nb_vars && 
      overLearn == p.overLearn &&
      true_positives == p.true_positives &&
      false_positives == p.false_positives &&
      true_negatives == p.true_negatives &&
      false_negatives == p.false_negatives;
}


void FuzzySystemMetrics::operator+=(const FuzzySystemMetrics& m) {
  double v = 0;
  sensitivity += !is_na(v = m.sensitivity) ? v : 0;
  specificity += !is_na(v = m.specificity) ? v : 0;
  // sensitivity += m.sensitivity;
  // specificity += m.specificity;
  accuracy += !is_na(v = m.accuracy) ? v : 0;
  ppv += !is_na(v = m.ppv) ? v : 0;
  rmse += !is_na(v = m.rmse) ? v : 0;
  rrse += !is_na(v = m.rrse) ? v : 0;
  rae += !is_na(v = m.rae) ? v : 0;
  mse += !is_na(v = m.mse) ? v : 0;
  distanceThreshold += !is_na(v = m.distanceThreshold) ? v : 0;
  distanceMinThreshold += !is_na(v = m.distanceMinThreshold) ? v : 0;
  // nb_vars += !is_na(v = m.nb_vars) ? v : 0;
  overLearn += !is_na(v = m.overLearn) ? v : 0;
  true_positives += !is_na(v = m.true_positives) ? v : 0;
  true_negatives += !is_na(v = m.true_negatives) ? v : 0;
  false_positives += !is_na(v = m.false_positives) ? v : 0;
  false_negatives += !is_na(v = m.false_negatives) ? v : 0;
}

NamedList FuzzySystemMetrics::describe() const {
  NamedList desc;
  desc.add("sensitivity", sensitivity);
  desc.add("specificity", specificity);
  desc.add("accuracy", accuracy);
  desc.add("ppv", ppv);
  desc.add("rmse", rmse);
  desc.add("rrse", rrse);
  desc.add("rae", rae);
  desc.add("mse", mse);
  desc.add("distanceThreshold", distanceThreshold);
  desc.add("distanceMinThreshold", distanceMinThreshold);
  desc.add("nb_vars", nb_vars);
  desc.add("overLearn", overLearn);
  desc.add("true_positives", true_positives);
  desc.add("false_positives", false_positives);
  desc.add("true_negatives", true_negatives);
  desc.add("false_negatives", false_negatives);

  return desc;
}

namespace fuzzy_coco {
  ostream& operator<<(ostream& out, const FuzzySystemMetrics& p) {
    out 
    << "sensitivity=" << p.sensitivity << ", " 
    << "specificity=" << p.specificity << ", " 
    << "accuracy=" << p.accuracy << ", " 
    << "ppv=" << p.ppv << ", "
    << "rmse=" << p.rmse << ", " 
    << "rrse=" << p.rrse << ", " 
    << "rae=" << p.rae << ", "  
    << "mse=" << p.mse << ", "
    << "distanceThreshold=" << p.distanceThreshold << ", " 
    << "distanceMinThreshold=" << p.distanceMinThreshold << ", " 
    << "nb_vars=" << p.nb_vars << ", "  
    << "overLearn=" << p.overLearn  << ", " 
    << "TP=" << p.true_positives << ", " 
    << "FP=" << p.false_positives << ", " 
    << "TN=" << p.true_negatives << ", " 
    << "FN=" << p.false_negatives;

    return out;
  }
}//KCOV IGNORE

