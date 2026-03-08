"""Logistic Regression from scratch with gradient descent and cross-entropy."""

from __future__ import annotations

import numpy as np

from .base import TrainingHistory


class LogisticRegressionGD:
    def __init__(self, learning_rate: float = 0.1, iterations: int = 700, regularization: str = "none", lambda_: float = 0.0):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.history = TrainingHistory()

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

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

        eps = 1e-12
        for _ in range(self.iterations):
            linear = X @ self.weights + self.bias
            probs = self._sigmoid(linear)

            ce = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
            reg_loss, reg_grad = self._regularization_term(n_samples)
            loss = ce + reg_loss

            dw = (1 / n_samples) * (X.T @ (probs - y)) + reg_grad
            db = (1 / n_samples) * np.sum(probs - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.history.append(loss, self.weights, self.bias)

        return self.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
