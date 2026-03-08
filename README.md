# GradVis AI – Gradient Descent & ML Visualizer

GradVis AI is a hands-on educational Python project that teaches machine learning foundations through **manual implementations** of optimization algorithms.

## What it includes

- Linear Regression using gradient descent and MSE
- Logistic Regression using gradient descent and cross-entropy
- Polynomial Regression (optional non-linear extension)
- Optional L1/L2 regularization for optimization intuition
- Interactive Streamlit dashboard with:
  - algorithm switching
  - synthetic or CSV-based datasets
  - hyperparameter controls
  - loss curves, parameter trajectories, and fitted boundaries/lines
  - calculus and statistics explanations

## Why this is portfolio-ready

This project emphasizes:

- Calculus: explicit partial derivatives + update rules
- Statistics: loss, variance/bias intuition, and model diagnostics
- Optimization: convergence behavior with learning-rate and regularization controls

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data format for CSV upload

- Include at least two numeric columns
- All numeric columns except the last are treated as features (`X`)
- The final numeric column is treated as target (`y`)

## Project structure

```
app.py
src/gradvis_ai/models/
src/gradvis_ai/utils/
tests/
```

## Notes

- No scikit-learn or high-level ML model libraries are used for training algorithms.
- NumPy is used for vectorized math and gradient calculations.
