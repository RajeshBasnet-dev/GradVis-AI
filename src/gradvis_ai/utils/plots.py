"""Plot helpers for optimization traces and model visualizations."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_loss_curve(losses: list[float]):
    fig = px.line(x=np.arange(len(losses)), y=losses, labels={"x": "Iteration", "y": "Loss"}, title="Loss over iterations")
    return fig


def plot_regression_fit(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, title: str = "Regression Fit"):
    order = np.argsort(X[:, 0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:, 0], y=y, mode="markers", name="Data"))
    fig.add_trace(go.Scatter(x=X[order, 0], y=y_pred[order], mode="lines", name="Prediction", line=dict(color="firebrick", width=3)))
    fig.update_layout(title=title, xaxis_title="Feature", yaxis_title="Target")
    return fig


def plot_logistic_boundary(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float):
    fig = go.Figure()
    for cls in [0, 1]:
        mask = y == cls
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name=f"Class {cls}",
            )
        )

    if len(weights) == 2 and abs(weights[1]) > 1e-8:
        x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 120)
        y_vals = -(weights[0] * x_vals + bias) / weights[1]
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Decision boundary", line=dict(color="black", width=3)))

    fig.update_layout(title="Logistic Decision Boundary", xaxis_title="x1", yaxis_title="x2")
    return fig


def plot_weight_path(weights: list[np.ndarray], max_points: int = 250):
    if not weights:
        return go.Figure()

    sampled = weights[:: max(1, len(weights) // max_points)]
    w1 = [w[0] if len(w) > 0 else 0.0 for w in sampled]
    w2 = [w[1] if len(w) > 1 else 0.0 for w in sampled]

    fig = go.Figure(data=[go.Scatter(x=w1, y=w2, mode="lines+markers", marker=dict(size=5))])
    fig.update_layout(title="Weight trajectory during gradient descent", xaxis_title="w1", yaxis_title="w2 (or 0 if 1D)")
    return fig
