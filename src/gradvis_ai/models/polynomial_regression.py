"""Polynomial regression built on manual linear regression gradients."""

from __future__ import annotations

import numpy as np

from .linear_regression import LinearRegressionGD


class PolynomialRegressionGD:
    def __init__(
        self,
        degree: int = 2,
        learning_rate: float = 0.01,
        iterations: int = 800,
        regularization: str = "none",
        lambda_: float = 0.0,
    ):
        self.degree = degree
        self.model = LinearRegressionGD(learning_rate, iterations, regularization, lambda_)

    def _poly_features(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != 1:
            raise ValueError("Polynomial regression visualization expects a single feature.")
        x = X[:, 0]
        features = [x**p for p in range(1, self.degree + 1)]
        return np.column_stack(features)

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self.model.fit(self._poly_features(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._poly_features(X))

    @property
    def weights(self):
        return self.model.weights

    @property
    def bias(self):
        return self.model.bias

    @property
    def history(self):
        return self.model.history
