Quickstart
==========

Install and import::

   pip install -e .

Train a classifier on Iris::

   from sklearn.datasets import load_iris
   from fuzzycocopython import FuzzyCocoClassifier

   data = load_iris(as_frame=True)
   clf = FuzzyCocoClassifier(random_state=0)
   clf.fit(data.data, data.target)
   print("Accuracy:", clf.score(data.data, data.target))
   print(clf.rules_df_.head())

Train a regressor::

   from sklearn.datasets import load_diabetes
   from fuzzycocopython import FuzzyCocoRegressor

   diabetes = load_diabetes(as_frame=True)
   reg = FuzzyCocoRegressor(random_state=0)
   reg.fit(diabetes.data, diabetes.target)
   print("R2:", reg.score(diabetes.data, diabetes.target))

See ``demo.ipynb`` in the repository for a compact walkthrough.
