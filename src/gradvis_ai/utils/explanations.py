"""Educational text blocks displayed in the Streamlit app."""

GRADIENT_DESCENT_CALCULUS = r"""
### Calculus Core: Partial Derivatives and Update Rules
For a model prediction \(\hat{y} = w^T x + b\):

- **Linear Regression Loss (MSE):**
  \[
  J(w,b)=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2
  \]
  \[
  \frac{\partial J}{\partial w} = \frac{2}{n}X^T(\hat{y} - y), \quad
  \frac{\partial J}{\partial b} = \frac{2}{n}\sum(\hat{y}-y)
  \]

- **Logistic Regression Hypothesis:**
  \[
  p = \sigma(z)=\frac{1}{1+e^{-z}},\ z=w^Tx+b
  \]

- **Cross-Entropy Loss:**
  \[
  J(w,b)= -\frac{1}{n}\sum\left[y\log(p)+(1-y)\log(1-p)\right]
  \]
  \[
  \frac{\partial J}{\partial w}=\frac{1}{n}X^T(p-y), \quad
  \frac{\partial J}{\partial b}=\frac{1}{n}\sum(p-y)
  \]

Gradient descent update:
\[
w \leftarrow w - \alpha \frac{\partial J}{\partial w},\quad b \leftarrow b - \alpha \frac{\partial J}{\partial b}
\]
where \(\alpha\) is the learning rate.
"""

STATISTICS_AND_OPTIMIZATION = """
### Statistics & Optimization Intuition
- **Variance**: High variance means the model tracks noise and overfits.
- **Bias**: High bias means underfitting (model too simple for pattern).
- **Loss curves** let you diagnose convergence:
  - smooth decrease -> stable learning,
  - oscillation -> learning rate likely too high,
  - flat high loss -> learning rate too low or poor feature scaling.
- **Regularization** controls complexity:
  - L1 encourages sparse parameters,
  - L2 shrinks large weights and stabilizes optimization.
"""
