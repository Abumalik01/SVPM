"""Phase D: diagnostic baselines (approved U12). These are NOT candidate models.

1. Majority-class dummy: always predicts the most frequent training label.
2. Label-shuffle diagnostic: permute TRAINING labels only (same label counts),
   train the usual model, and evaluate on the UNTOUCHED validation labels. If a real
   model scores about the same as its shuffled version, the features carry no usable signal.
Shuffling is row-level and seeded; repeated N_SHUFFLES times per fold and model.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier

N_SHUFFLES = 10
SEED = 42


def majority_dummy() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def shuffle_labels(y_train, seed: int) -> np.ndarray:
    """Return a permutation of y_train (label distribution preserved). Never use on validation/test labels."""
    return np.random.default_rng(seed).permutation(np.asarray(y_train))


def shuffle_seeds(fold: int, base_seed: int = SEED, n: int = N_SHUFFLES) -> list[int]:
    return [base_seed * 1000 + fold * 100 + i for i in range(n)]
