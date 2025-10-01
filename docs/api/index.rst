API Reference
=============

.. note::
   This API section focuses on the public Python wrappers. The internal
   compiled core is not documented here to avoid confusion.

Estimators
----------

.. autoclass:: fuzzycocopython.fuzzycoco_base.FuzzyCocoClassifier
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: set_score_request, set_fit_request, set_predict_request, set_transform_request

.. autoclass:: fuzzycocopython.fuzzycoco_base.FuzzyCocoRegressor
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: set_score_request, set_fit_request, set_predict_request, set_transform_request

Utilities
---------

.. autofunction:: fuzzycocopython.fuzzycoco_base.save_model

.. autofunction:: fuzzycocopython.fuzzycoco_base.load_model

Plotting
--------

.. autoclass:: fuzzycocopython.fuzzycoco_plot_mixin.FuzzyCocoPlotMixin
   :members:
   :undoc-members:
