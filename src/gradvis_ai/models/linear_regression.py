"""Linear Regression implemented from scratch with gradient descent."""

from __future__ import annotations

import numpy as np

from .base import TrainingHistory


class LinearRegressionGD:
    """Single/multi-feature linear regression using manual gradients."""

    def __init__(self, learning_rate: float = 0.01, iterations: int = 500, regularization: str = "none", lambda_: float = 0.0):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.history = TrainingHistory()

    def _regularization_term(self, n_samples: int) -> tuple[float, np.ndarray]:
        if self.regularization == "l2":
            loss_reg = (self.lambda_ / (2 * n_samples)) * np.sum(self.weights**2)
            grad_reg = (self.lambda_ / n_samples) * self.weights
        elif self.regularization == "l1":
            loss_reg = (self.lambda_ / n_samples) * np.sum(np.abs(self.weights))
            grad_reg = (self.lambda_ / n_samples) * np.sign(self.weights)
        else:
            loss_reg = 0.0
            grad_reg = np.zeros_like(self.weights)
        return loss_reg, grad_reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingHistory:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = TrainingHistory()

        for _ in range(self.iterations):
            y_pred = X @ self.weights + self.bias
            errors = y_pred - y

            mse_loss = np.mean(errors**2)
            reg_loss, reg_grad = self._regularization_term(n_samples)
            loss = mse_loss + reg_loss

            dw = (2 / n_samples) * (X.T @ errors) + reg_grad
            db = (2 / n_samples) * np.sum(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.history.append(loss, self.weights, self.bias)

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias
