"""
Quick helper script to train a tiny fuzzy model and display rule activation stats.

Usage:
    python test.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fuzzycocopython import FuzzyCocoClassifier


def main():
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.random((12, 3)), columns=["feat_1", "feat_2", "feat_3"])
    y = pd.Series(rng.integers(0, 2, size=len(X)), name="target")

    clf = FuzzyCocoClassifier(
        nb_rules=3,
        nb_sets_in=2,
        nb_sets_out=2,
        max_generations=5,
        random_state=42,
    )
    clf.fit(X, y)

    stats, matrix = clf.rules_stat_activations(X, return_matrix=True, sort_by_impact=False)

    print("Model rules:")
    print(clf.rules_view_)
    print("Model default rules:")
    print(clf.default_rules_view_)
    print()
    print("=== Rule statistics ===")
    print(stats)
    print()

    print("=== Activation matrix (first 5 rows) ===")
    print(pd.DataFrame(matrix[:5], columns=stats.index))
    print()

    default_rules = getattr(matrix, "default_rules", None)
    if default_rules is None:
        print("No default-rule activations recorded.")
    else:
        print("=== Default rule activations per sample ===")
        for idx, payload in enumerate(default_rules[:5]):
            print(f"Sample {idx}: {payload}")


if __name__ == "__main__":
    main()
