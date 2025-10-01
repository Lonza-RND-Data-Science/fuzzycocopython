- [fuzzy\_system\_metrics\_computer.h](#fuzzy_system_metrics_computerh)
  - [class FuzzySystemPreferenceComputer](#class-fuzzysystempreferencecomputer)
- [fuzzy\_system.h](#fuzzy_systemh)
- [string\_utils.h](#string_utilsh)
- [fuzzy\_coco.h](#fuzzy_cocoh)
  - [FuzzyCocoFitnessMethod::evaluateSystemFitness()](#fuzzycocofitnessmethodevaluatesystemfitness)
  - [FuzzycocoPreferenceFitnessMethod::fitnessImpl](#fuzzycocopreferencefitnessmethodfitnessimpl)
- [fuzzy\_coco\_executable.cpp](#fuzzy_coco_executablecpp)
  - [initFitnessMethod](#initfitnessmethod)
  - [loadFuzzySystem](#loadfuzzysystem)
  - [launch](#launch)


# fuzzy_system_metrics_computer.h


## class FuzzySystemPreferenceComputer

```
class FuzzySystemPreferenceComputer {
  public:
    FuzzySystemPreferenceComputer() {};
    // those methods could be made static

    // just sum the weigths in the vector ??
    double compute_total_weight(const vector<tuple<string, double>>& variable_weights) const;

    // given a list of variable names, compute the sum of weights. If at least one variable with weight 1 is not present, return nullopt
    std::optional<double> compute_weighted_preference(const vector<string>& presence, const vector<tuple<string, double>>& variable_weights) const;
};
```

- should not use names, just variable indices --> vector<double> instead of vector<tuple<string, double>>

- should be integrated in FuzzycocoPreferenceFitnessMethod


# fuzzy_system.h

- added vector<string> getUsedInputVariableNames() const.

# string_utils.h

```
inline std::string vectorToString(const std::vector<std::tuple<std::string, double>>& vec) {
```

should instead use the describe()/printDescription trait./


# fuzzy_coco.h

## FuzzyCocoFitnessMethod::evaluateSystemFitness()

```
double evaluateSystemFitness(FuzzySystemMetrics weights, NamedList thresh_desc, const DataFrame& df, FuzzySystem& fs);
```
no. FuzzyCocoFitnessMethod should not be aware of that data. Should probably go elsewhere, higher level

## FuzzycocoPreferenceFitnessMethod::fitnessImpl
```
double FuzzycocoPreferenceFitnessMethod::fitnessImpl(const Genome& rules_genome, const Genome& mfs_genome)
```
call FuzzySystemWeightedFitness::fitness(fitMetrics(), extra_num.value(), extra_denum);


# fuzzy_coco_executable.cpp

## initFitnessMethod
```
unique_ptr<FuzzyCocoFitnessMethod> initFitnessMethod(const DataFrame& df,const ScriptParams& params) 
```

build the appropriate fitnessmethod given the input and params: a FuzzyCocoFitnessMethod or  FuzzycocoPreferenceFitnessMethod

## loadFuzzySystem

ok... the Fuzzy System is duplicated but that should not really be a problem.
Would rather add a constructor or factory method...

## launch

now calling `(*fitmethod).evaluateSystemFitness` instead of fitter.fitness

launch() --> FuzzyCocoFitnessMethod::evaluateSystemFitness()