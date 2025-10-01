Parameters Guide
================

This page describes the full set of parameters used by the FuzzyCoco engine and
how to configure them from Python. The Python wrappers pass a
``FuzzyCocoParams`` object to the C++ core. You can:

- let the wrapper build sane defaults automatically (and override via
  scikit-learn style constructor kwargs like ``nb_rules=10``), or
- use the convenience helper ``make_fuzzy_params`` with a nested dict to tune a subset.

High‑level structure
--------------------

``FuzzyCocoParams`` is a container composed of sub-structures:

- ``global_params``: global search and rule topology limits.
- ``input_vars_params`` and ``output_vars_params``: encoding of variables and sets.
- ``rules_params`` and ``mfs_params``: evolutionary algorithm knobs for the rule genome
  and the membership function positions.
- ``fitness_params``: scoring and feature weighting.

GlobalParams
------------

Fields (defaults in parentheses):

- ``nb_rules`` (required in C++; default 5 in Python helper): number of rules
  to infer. Note: "DontCare" may effectively reduce the final rule count.
- ``nb_max_var_per_rule`` (auto): number of input variable slots in rule antecedents.
  The wrapper sets this to the number of features in ``X`` if not provided.
- ``max_generations`` (100): maximum coevolution generations.
- ``max_fitness`` (1.0): early-stop fitness threshold; values > 1 disable early stopping.
- ``nb_cooperators`` (2): number of cooperators during fitness evaluation.
- ``influence_rules_initial_population`` (False): bias initial population using feature weights.
- ``influence_evolving_ratio`` (0.8): evolving ratio when the above is enabled.

VarsParams (input/output)
-------------------------

Controls the encoding of variables and their fuzzy sets. One instance is used for
inputs and one for outputs.

- ``nb_sets`` (2 by default in Python helper): number of fuzzy sets per variable.
- ``nb_bits_vars`` (auto): bits to encode the variable index. For inputs, an extra
  bit is used to encode "DontCare". Internally computed as
  ``ceil(log2(nb_vars)) + 1`` when missing.
- ``nb_bits_sets`` (auto): bits to encode the set index. Computed as ``ceil(log2(nb_sets))``.
- ``nb_bits_pos`` (8): bits to encode discretized positions of membership functions.

EvolutionParams (rules_params / mfs_params)
-------------------------------------------

Evolutionary algorithm parameters for both the rules population and the membership
function positions population. Same fields for both:

- ``pop_size`` (50): population size.
- ``elite_size`` (5): number of elites preserved between generations.
- ``cx_prob`` (0.5): crossover probability.
- ``mut_flip_genome`` (0.5): probability a genome is chosen for mutation.
- ``mut_flip_bit`` (0.025): probability a bit of the genome is flipped.

FitnessParams
-------------

Scoring configuration and feature-specific weights.

- ``output_vars_defuzz_thresholds``: list of thresholds used for defuzzification.
  The wrapper sets a single value (default 0.5). During ``fit``, this is expanded
  to match the number of outputs via ``fix_output_thresholds``.
- ``metrics_weights``: weights applied to system metrics when optimizing. Available
  fields include: ``sensitivity``, ``specificity``, ``accuracy``, ``ppv``, ``rmse``,
  ``rrse``, ``rae``, ``mse``, and more (see ``fuzzy_system_metrics.h``). The Python
  helper defaults are ``sensitivity=1.0`` and ``specificity=0.8``; others default to 0.
- ``features_weights``: optional mapping ``{feature_name: weight}`` with weights in [0, 1].
  Unknown names raise at runtime.

Automatic defaults applied by the wrapper
----------------------------------------

When you do not pass a ``params`` object:

- ``nb_max_var_per_rule`` defaults to ``3``.
- Bit widths follow the C++ logic: ``nb_bits_vars = ceil(log2(nb_vars)) + 1``
  and ``nb_bits_sets = ceil(log2(nb_sets))``.
- ``output_vars_defuzz_thresholds`` are adapted to the number of outputs.

How to configure from Python
----------------------------

1) Minimal: rely on defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fuzzycocopython import FuzzyCocoClassifier
   clf = FuzzyCocoClassifier(random_state=0)
   clf.fit(X, y)

2) Moderate: override some top-level kwargs

You can also pass flat constructor kwargs in sklearn style:

.. code-block:: python

   # direct globals
   clf = FuzzyCocoClassifier(nb_rules=8, nb_max_var_per_rule=3)

   # or nested via double underscore
   clf = FuzzyCocoClassifier(
       global_params__nb_rules=8,
       input_vars_params__nb_sets=3,
       rules_params__pop_size=100,
   )

3) Tuning with ``make_fuzzy_params``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``make_fuzzy_params`` accepts a nested dict (recommended) and some convenient
flat keywords for common options.

.. code-block:: python

   from fuzzycocopython.utils import make_fuzzy_params

   params = make_fuzzy_params({
       "global_params": {
           "nb_rules": 10,
           "nb_max_var_per_rule": 4,
           "max_generations": 150,
       },
       "input_vars_params": {
           "nb_sets": 3,
           "nb_bits_pos": 6,
       },
       "output_vars_params": {
           "nb_sets": 3,
       },
       "rules_params": {  # evolutionary hyperparameters for rules
           "pop_size": 100,
           "elite_size": 10,
       },
       "mfs_params": {    # evolutionary hyperparameters for MF positions
           "pop_size": 80,
       },
       "fitness_params": {
           "threshold": 0.5,  # single value replicated if multi-output
           "metrics_weights": {"sensitivity": 1.0, "specificity": 1.0, "accuracy": 1.0},
           "features_weights": {"A": 1.0, "B": 0.2},
       },
   })

   clf = FuzzyCocoClassifier(params=params, random_state=0)
   clf.fit(X, y)

Notes and tips
--------------

- Bit defaults are aligned with the engine: ``nb_bits_sets = ceil(log2(nb_sets))``
  and ``nb_bits_vars = ceil(log2(nb_vars)) + 1``.
- ``nb_bits_pos`` controls discretization of MF positions. Smaller values constrain
  the search and can speed up runs at the cost of granularity.
- ``metrics_weights`` act as a linear scalarization over the internal metrics; set
  only the metrics you want to optimize explicitly.
- Use ``features_weights`` to encourage or discourage specific input variables in
  the genome encoding and selection. Unknown feature names raise an error.
