from __future__ import annotations

import time

import numpy as np
import streamlit as st

from src.gradvis_ai.models import LinearRegressionGD, LogisticRegressionGD, PolynomialRegressionGD
from src.gradvis_ai.utils.data import load_csv, make_linear_data, make_logistic_data
from src.gradvis_ai.utils.explanations import GRADIENT_DESCENT_CALCULUS, STATISTICS_AND_OPTIMIZATION
from src.gradvis_ai.utils.plots import (
    plot_logistic_boundary,
    plot_loss_curve,
    plot_regression_fit,
    plot_weight_path,
)

st.set_page_config(page_title="GradVis AI", layout="wide")
st.title("📉 GradVis AI – Gradient Descent & ML Visualizer")
st.caption("Learn linear and logistic models from scratch through calculus, statistics, and optimization.")

with st.sidebar:
    st.header("Controls")
    algo = st.selectbox("Algorithm", ["Linear Regression", "Logistic Regression", "Polynomial Regression"])
    data_source = st.radio("Dataset", ["Synthetic", "CSV Upload"])
    lr = st.slider("Learning rate", min_value=0.0001, max_value=1.0, value=0.05, step=0.0001, format="%.4f")
    iterations = st.slider("Iterations", min_value=50, max_value=3000, value=500, step=50)
    regularization = st.selectbox("Regularization", ["none", "l1", "l2"])
    lambda_ = st.slider("Regularization factor λ", 0.0, 2.0, 0.0, 0.05)
    degree = 2
    if algo == "Polynomial Regression":
        degree = st.slider("Polynomial degree", 2, 8, 3)

if data_source == "CSV Upload":
    uploaded = st.file_uploader("Upload CSV (last numeric column treated as target)", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to proceed.")
        st.stop()
    X, y, columns = load_csv(uploaded)
    st.write(f"Using columns: {columns}")
else:
    if algo == "Logistic Regression":
        X, y = make_logistic_data()
    else:
        X, y = make_linear_data(noise=1.0)

if algo == "Linear Regression":
    model = LinearRegressionGD(lr, iterations, regularization, lambda_)
elif algo == "Logistic Regression":
    model = LogisticRegressionGD(lr, iterations, regularization, lambda_)
else:
    model = PolynomialRegressionGD(degree, lr, iterations, regularization, lambda_)

if algo in {"Linear Regression", "Polynomial Regression"} and X.shape[1] != 1:
    st.warning("For regression plotting clarity, using first feature only.")
    X = X[:, :1]

train = st.button("Train model")

if train:
    history = model.fit(X, y)

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(plot_loss_curve(history.losses), use_container_width=True)

        if algo == "Logistic Regression":
            preds = model.predict(X)
            acc = float(np.mean(preds == y))
            st.metric("Training accuracy", f"{acc:.3f}")
            st.plotly_chart(plot_logistic_boundary(X, y, model.weights, model.bias), use_container_width=True)
        else:
            y_pred = model.predict(X)
            mse = float(np.mean((y_pred - y) ** 2))
            st.metric("Training MSE", f"{mse:.3f}")
            st.plotly_chart(plot_regression_fit(X, y, y_pred, title=f"{algo} fit"), use_container_width=True)

    with right:
        st.plotly_chart(plot_weight_path(history.weights), use_container_width=True)

        st.subheader("Animate optimization checkpoints")
        animate = st.checkbox("Play iteration snapshots")
        if animate:
            placeholder = st.empty()
            step = max(1, len(history.losses) // 50)
            for i in range(0, len(history.losses), step):
                w = history.weights[i]
                b = history.biases[i]
                placeholder.write({"iteration": i, "loss": history.losses[i], "weights": w.tolist(), "bias": b})
                time.sleep(0.04)

st.markdown("---")
with st.expander("📘 Calculus behind gradient descent", expanded=True):
    st.markdown(GRADIENT_DESCENT_CALCULUS)

with st.expander("📊 Statistics, bias/variance, and optimization", expanded=True):
    st.markdown(STATISTICS_AND_OPTIMIZATION)
