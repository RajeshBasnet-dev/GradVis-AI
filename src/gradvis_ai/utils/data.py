"""Dataset utilities for generated and uploaded data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_linear_data(n_samples: int = 120, noise: float = 1.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5, 5, size=(n_samples, 1))
    y = 3.2 * X[:, 0] + 1.7 + rng.normal(0, noise, size=n_samples)
    return X, y


def make_logistic_data(n_samples: int = 250, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.2, size=(n_samples, 2))
    logits = 1.5 * X[:, 0] - 2.2 * X[:, 1] + 0.2
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    return X, y


def load_csv(file) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(file)
    numeric_df = df.select_dtypes(include=["number"]) 
    if numeric_df.shape[1] < 2:
        raise ValueError("CSV must contain at least two numeric columns.")

    X = numeric_df.iloc[:, :-1].values
    y = numeric_df.iloc[:, -1].values
    return X, y, list(numeric_df.columns)
