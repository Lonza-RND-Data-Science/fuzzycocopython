Parameters Guide
================

FuzzyCoco estimators now expose an explicit, scikit-learn compatible set of
constructor keyword arguments. The wrappers translate those keyword arguments
into a ``FuzzyCocoParams`` instance that is passed to the C++ optimisation
engine. This page documents the structure of that parameter object and the
defaults used by the Python interface.

High-level structure
--------------------

``FuzzyCocoParams`` is composed of the following sub-sections:

- ``global_params``: topology and cross-population settings.
- ``input_vars_params`` and ``output_vars_params``: encoding of variables and fuzzy sets.
- ``rules_params`` and ``mfs_params``: evolutionary algorithm knobs for rules and
  membership functions respectively.
- ``fitness_params``: scoring configuration and optional feature weights.

GlobalParams
------------

Constructor keywords mapping to this section:

- ``nb_rules`` (default ``5``): number of candidate rules evolved each generation.
- ``nb_max_var_per_rule`` (default ``3``): maximum antecedents per rule.
- ``max_generations`` (default ``100``): total number of evolution iterations.
- ``max_fitness`` (default ``1.0``): early-stopping fitness target (values > 1.0 disable it).
- ``nb_cooperators`` (default ``2``): number of cooperating agents when estimating fitness.
- ``influence_rules_initial_population`` (default ``False``): seed the initial population
  using influence heuristics.
- ``influence_evolving_ratio`` (default ``0.8``): ratio controlling the strength of the
  above influence during evolution.

VarsParams (input/output)
-------------------------

Both input and output variables share the same structure. Corresponding constructor
arguments are suffixed with ``_in`` or ``_out``.

- ``nb_sets_in`` / ``nb_sets_out`` (default ``2``): number of fuzzy sets per variable.
- ``nb_bits_vars_in`` / ``nb_bits_vars_out`` (default ``auto``): bit width used to encode
  variable indices. When left to auto, the wrapper computes
  ``ceil(log2(nb_vars)) + 1`` where ``nb_vars`` is the number of input/output variables
  observed during ``fit``.
- ``nb_bits_sets_in`` / ``nb_bits_sets_out`` (default ``auto``): bit width used to encode
  set indices. The automatic rule is ``ceil(log2(nb_sets))``.
- ``nb_bits_pos_in`` / ``nb_bits_pos_out`` (default ``8``): discretisation used to encode
  membership-function positions.

EvolutionParams (rules_params / mfs_params)
-------------------------------------------

Both the rule population and the membership-function population expose the same set
of hyper-parameters:

- ``pop_size_rules`` / ``pop_size_mfs`` (default ``200``): population size.
- ``elite_size_rules`` / ``elite_size_mfs`` (default ``5``): elite survivors per generation.
- ``cx_prob_rules`` / ``cx_prob_mfs`` (defaults ``0.6`` and ``0.9`` respectively): crossover probability.
- ``mut_flip_genome_rules`` / ``mut_flip_genome_mfs`` (defaults ``0.4`` and ``0.2``): probability that a genome is selected for mutation.
- ``mut_flip_bit_rules`` / ``mut_flip_bit_mfs`` (both default ``0.01``): probability that a
  bit within a selected genome is flipped.

FitnessParams
-------------

- ``threshold`` (default ``0.5``): singleton defuzzification threshold. During ``fit`` the
  value is expanded to match the number of outputs.
- ``metrics_weights``: mapping of fitness metrics to weights. When omitted the classifier
  defaults to ``{"accuracy": 1.0}`` and the regressor defaults to ``{"rmse": 1.0}``.
- ``features_weights`` (default ``None``): optional per-feature weights used by the fitness
  function.

Automatic defaults applied during ``fit``
----------------------------------------

- Bit widths fall back to the automatic rules described above when their constructor
  arguments are left as ``None``.
- The threshold list is replicated to match the observed number of outputs.
- Feature names and number of outputs are inferred from the data passed to ``fit``.

Configuring estimators from Python
----------------------------------

Minimal configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fuzzycocopython import FuzzyCocoClassifier

   clf = FuzzyCocoClassifier(random_state=0)
   clf.fit(X, y)

Override selected hyper-parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   clf = FuzzyCocoClassifier(
       nb_rules=12,
       nb_sets_in=3,
       pop_size_rules=150,
       random_state=42,
   )
   clf.fit(X, y)

Advanced: build ``FuzzyCocoParams`` directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For specialised pipelines you may still construct the parameter object yourself using
the ``fuzzycocopython.utils.build_fuzzycoco_params`` helper. This mirrors the logic used
inside ``fit`` and accepts the same keyword arguments plus dataset dimensions:

.. code-block:: python

   from fuzzycocopython.utils import build_fuzzycoco_params

   params = build_fuzzycoco_params(
       nb_features=X.shape[1],
       n_outputs=1,
       nb_rules=10,
       nb_sets_in=3,
       nb_sets_out=2,
       threshold=0.4,
       metrics_weights={"accuracy": 1.0, "sensitivity": 0.5},
       features_weights={"A": 1.0},
       pop_size_rules=100,
       pop_size_mfs=80,
   )

   # The helper is primarily intended for direct interaction with the low-level bindings.

Notes and tips
--------------

- Lower ``nb_bits_pos`` values restrict the search space for membership-function
  positions and may speed up optimisation at the cost of precision.
- ``metrics_weights`` act as a linear scalarisation of the internal engine metrics.
  Only specify the ones you care about; unspecified metrics default to zero weight.
- ``features_weights`` expects feature names as seen by pandas DataFrames or the
  ``feature_names`` argument passed to ``fit``. Unknown names raise an error inside
  the engine.
