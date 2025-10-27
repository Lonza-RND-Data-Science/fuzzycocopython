from fuzzycocopython import FuzzyCocoRegressor
import numpy as np

model = FuzzyCocoRegressor(nb_rules=5, random_state=42)
X = np.random.rand(10, 3)
y = np.random.rand(10)

model.fit(X,y)


#print(model.rules_stat_activations(X))
print("----")
model.set_target_names("new_target")
print(model.predict(X[:5]))

print(model.rules_stat_activations(X))
