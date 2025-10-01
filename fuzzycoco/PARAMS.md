Fuzzy Coco Parameters <!-- omit from toc -->
=====================

- [global\_params](#global_params)
  - [nb\_rules](#nb_rules)
  - [nb\_max\_var\_per\_rule](#nb_max_var_per_rule)
  - [max\_generations](#max_generations)
  - [max\_fitness](#max_fitness)
  - [nb\_cooperators](#nb_cooperators)
  - [influence\_rules\_initial\_population](#influence_rules_initial_population)
  - [influence\_evolving\_ratio](#influence_evolving_ratio)
- [input\_vars\_params](#input_vars_params)
  - [nb\_sets](#nb_sets)
  - [nb\_bits\_vars](#nb_bits_vars)
  - [nb\_bits\_sets](#nb_bits_sets)
  - [nb\_bits\_pos](#nb_bits_pos)
- [output\_vars\_params](#output_vars_params)
  - [nb\_sets](#nb_sets-1)
  - [nb\_bits\_vars](#nb_bits_vars-1)
  - [nb\_bits\_sets](#nb_bits_sets-1)
  - [nb\_bits\_pos](#nb_bits_pos-1)
- [rules\_params](#rules_params)
  - [pop\_size](#pop_size)
  - [elite\_size](#elite_size)
  - [cx\_prob](#cx_prob)
  - [mut\_flip\_genome](#mut_flip_genome)
  - [mut\_flip\_bit](#mut_flip_bit)
- [mfs\_params](#mfs_params)
  - [pop\_size](#pop_size-1)
  - [elite\_size](#elite_size-1)
  - [cx\_prob](#cx_prob-1)
  - [mut\_flip\_genome](#mut_flip_genome-1)
  - [mut\_flip\_bit](#mut_flip_bit-1)
- [fitness\_params](#fitness_params)
  - [output\_vars\_defuzz\_thresholds](#output_vars_defuzz_thresholds)
  - [features\_weights](#features_weights)
  - [metrics\_weights](#metrics_weights)
    - [sensitivity](#sensitivity)
    - [specificity](#specificity)
    - [accuracy](#accuracy)
    - [ppv](#ppv)
    - [rmse](#rmse)
    - [rrse](#rrse)
    - [rae](#rae)
    - [mse](#mse)
    - [distanceThreshold](#distancethreshold)
    - [distanceMinThreshold](#distanceminthreshold)
    - [nb\_vars](#nb_vars)


There are quite a lot of parameters controlling the behaviour of Fuzzy Coco.

## global_params

These are high-level structural parameters and parameters controlling the number of iterations of the algorithm.

### nb_rules

- The number of rules of the fuzzy system to fit. 
- **Mandatory**
- Note that due to **wildcards** the resulting fuzzy system may have less rules that this parameter.

### nb_max_var_per_rule

- The maximum number of antecedents (input variables) to use in each rule. 
- - **Mandatory**
- Note that due to **wildcards** some rules may use less input variables than this parameter.

### max_generations

- The maximum number of iterations of the algorithm. Each iteration produces a new generation of the rules and membership functions populations.
- default: 100

### max_fitness

- a stop condition: the iterations stop as soon as a generated fuzzy system reaches that threshold.
- default: 1.0 (no early stop)

### nb_cooperators

  - The number of cooperators to use in the coevolution algorithm.
  - default: 2

### influence_rules_initial_population

  - whether to influence the initial genome rules population with the features weights (if any)
  - default: false (use random initialization)
  - cf the **features importance** new feature (TODO)

### influence_evolving_ratio

  -  the evolving ratio to use to influence the initial genome rules population, cf [influence_rules_initial_population](#influence_rules_initial_population)
  -  default: 0.8


## input_vars_params

Parameters related to the INPUT variables.

### nb_sets

  - the number of fuzzy sets to use for the Membership function associated with the input variables
  - default: 3
  - structural parameter

### nb_bits_vars

  - the number of bits to encode the INPUT variable indexes. 
  - default: **automatic** to fit the number of INPUT variables + 1 bit for the DONTCARE/wildcard.
  - N.B: at least one extra bit must be added to account for the DONTCARE
  
### nb_bits_sets

  - the number of bits to encode the set index. 
  - default: **automatic** to fit `nb_sets` 

### nb_bits_pos

  - the number of bits to encode the INPUT variable fuzzy set position/value. 
  - **Mandatory**
  - N.B: a low value means that the variable range is heavily discretized (can take only few values)
  
## output_vars_params

Parameters related to the OUTPUT variables.

### nb_sets

  - the number of fuzzy sets to use for the Membership function associated with the OUTPUT variables
  - default: 3
  - structural parameter

### nb_bits_vars

  - the number of bits to encode the OUTPUT variable indexes. 
  - default: **automatic** to fit the number of OUTPUT variables + 1 bit for the DONTCARE/wildcard.
  - N.B: at least one extra bit must be added to account for the DONTCARE
  
### nb_bits_sets

  - the number of bits to encode the set index. 
  - default: **automatic** to fit `nb_sets` 

### nb_bits_pos

  - the number of bits to encode the OUTPUT variable fuzzy set position/value. 
  - **Mandatory**
  - N.B: a low value means that the variable range is heavily discretized (can take only few values)
  
## rules_params

Parameters controlling the genetic evolution of the Fuzzy Rules population.

### pop_size

  - the size (number of genomes/individuals) of the population to evolve
  - **Mandatory**

### elite_size

  -  the size of the elite sub-population: the number of elite genomes/individuals
  -  default: 5

### cx_prob

  - the crossover probability
  - default: 0.5

### mut_flip_genome

  - the probability that a particular genome/individual is to be mutated
  - default: 0.5

### mut_flip_bit

  - the probability to mutate a particular bit of a genome
  - default: 0.025
  
## mfs_params

Parameters controlling the genetic evolution of the Membership functions (fuzzy sets)  population.

### pop_size

  - the size (number of genomes/individuals) of the population to evolve
  - **Mandatory**

### elite_size

  -  the size of the elite sub-population: the number of elite genomes/individuals
  -  default: 5

### cx_prob

  - the crossover probability
  - default: 0.5

### mut_flip_genome

  - the probability that a particular genome/individual is to be mutated
  - default: 0.5

### mut_flip_bit

  - the probability to mutate a particular bit of a genome
  - default: 0.025
  
## fitness_params

Parameters controllling how the fitness of fuzzy systems is computed.

### output_vars_defuzz_thresholds

  - a output variable threshold is the threshold determining if an outcome is POSITIVE or NEGATIVE. If the fuzzyfied
    value is greater or equal to that threshold, it is defuzzyfied as POSITIVE, otherwise NEGATIVE.
  - **Mandatory**
  - it is an array, there should be one value per OUTPUT variable.
  - Usually, it is set in the middle of variable range.
  - N.B: in case there are multiple OUTPUT variables, but only one threshold is set, this threshold is *automatically* applied to all OUTPUT variables for convenience. 

### features_weights

  - the weigths to apply to the fitness depending on which input variables are effectively used in the rules of a fuzzy system.
  - the fitness is computed as `(fitness_numerator + sumOfWeightsActuallyUsed) / (fitness_denominator + sumOfWeights)`
    where `sumOfWeightsActuallyUsed` is the sum of weights of variables actually used by the rules and `sumOfWeights` is 
    the sum of all features weights set.
  - if a variable has a weight = 1, and it is not used by the rules that the fitness is set to 0 ==> the variable is **mandatory**
  - default: none. a JSON dictionary associating weigths to some INPUT variables. 
  - N.B: this is a relatively new feature

### metrics_weights

There are the weights to apply for each type of fitness metric to compute the fitness.
We note: 
  - P: number of positives
  - N: number of negatives
  - TP: number of true positives
  - TN: number of true negatives
  - FP: number of false positives
  - FN: number of false negatives

N.B: positives/negatives are computed using the [output_vars_defuzz_thresholds](#output_vars_defuzz_thresholds)

#### sensitivity

  - the weight for the sensitivity (true positive rate): `TP / (TP + FN)`
  - default: 1.0
  
#### specificity

  - the weight for the specificity (true negative rate): `TN / (TN + FP)`
  - default: 0.8
  
#### accuracy

  - the weight for the accuracy: `(TP + TN) / (TP + TN + FP + FN)`
  - default: 0

#### ppv

  - the weight for the positive predictive value (precision): `TP / (TP + FP))`
  - default: 0

#### rmse

  - the weight for the root mean square error
  - default: 0
  
#### rrse

  - the weight for the root relative squared error
  - default: 0

#### rae

  - the weight for the relative absolute error
  - default: 0
  
#### mse

  - the weight for the mean squared error
  - default: 0

#### distanceThreshold

  - the weight for the distanceThreshold metric
  - the distanceThreshold metric is the ratio of the distance to the output threshold of the predicted and actual values.
  - default: 0
  
#### distanceMinThreshold

  - the weight for the distanceMinThreshold metric
  - the distanceMinThreshold metric is: TODO
  - default: 0

#### nb_vars

  - the weight for the `nb_vars` metric
  - default: 0
  - the `nb_vars` metric is: `1/nb_vars` where `nb_vars` is the actual number of variables used by the set of rules of a fuzzy system
  - it is used to penalize huge systems.

