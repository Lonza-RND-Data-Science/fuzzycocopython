API Reference
=============

.. note::
   This API section focuses on the public Python wrappers. The internal
   compiled core is not documented here to avoid confusion.

Quick example
-------------

.. code-block:: python

   from fuzzycocopython import FuzzyCocoClassifier

   clf = FuzzyCocoClassifier(random_state=0)
   clf.fit(X, y)
   preds = clf.predict(X)
   print(clf.score(X, y))

Estimators
----------

.. autoclass:: fuzzycocopython.fuzzycoco_base.FuzzyCocoClassifier
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: set_score_request, set_fit_request, set_predict_request, set_transform_request

.. autoclass:: fuzzycocopython.fuzzycoco_base.FuzzyCocoRegressor
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: set_score_request, set_fit_request, set_predict_request, set_transform_request

Utilities
---------

.. autofunction:: fuzzycocopython.fuzzycoco_base.save_model

.. autofunction:: fuzzycocopython.fuzzycoco_base.load_model

.. code-block:: python

   from fuzzycocopython import FuzzyCocoClassifier
   from fuzzycocopython.fuzzycoco_base import save_model, load_model

   clf = FuzzyCocoClassifier().fit(X, y)
   save_model(clf, "clf.joblib")

   restored = load_model("clf.joblib")
   print(restored.predict(X[:5]))

Plotting
--------

.. autoclass:: fuzzycocopython.fuzzycoco_plot_mixin.FuzzyCocoPlotMixin
   :members:
   :undoc-members:
