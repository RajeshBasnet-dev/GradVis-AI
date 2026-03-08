import numpy as np

from src.gradvis_ai.models import LinearRegressionGD, LogisticRegressionGD, PolynomialRegressionGD


def test_linear_regression_converges():
    X = np.linspace(-2, 2, 80).reshape(-1, 1)
    y = 4 * X[:, 0] + 1
    model = LinearRegressionGD(learning_rate=0.05, iterations=600)
    history = model.fit(X, y)
    assert history.losses[-1] < history.losses[0]
    assert np.isclose(model.weights[0], 4, atol=0.2)


def test_logistic_regression_learns_boundary():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegressionGD(learning_rate=0.1, iterations=600)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.mean(preds == y) > 0.9


def test_polynomial_regression_fits_curve():
    X = np.linspace(-2, 2, 120).reshape(-1, 1)
    y = 1.2 * X[:, 0] ** 2 - 0.7 * X[:, 0] + 2
    model = PolynomialRegressionGD(degree=2, learning_rate=0.02, iterations=1300)
    history = model.fit(X, y)
    assert history.losses[-1] < 0.05
