# Logistic Regression

## Introduction

Logistic regression is the standard starting point for binary classification. It models the relationship between a binary outcome $y \in \{0, 1\}$ and a set of predictors $X$ by estimating the probability that $y = 1$ given $X$. Despite its name, it is a classification model — but one grounded in probability, not arbitrary decision boundaries.

The goal is to estimate the conditional probability $P(y = 1 \mid X)$, quantify uncertainty around those estimates, and make valid inferences about which predictors drive the outcome and by how much. This requires a model that respects the bounded nature of probabilities, a fitting procedure suited to binary outcomes, and assumptions that are distinct from — and in some ways more demanding than — those of linear regression.

---

## 1. The Model

### Why Not Linear Regression?

Applying OLS to a binary outcome produces the **Linear Probability Model** (LPM): $P(y = 1 \mid X) = X\beta$. It is unbiased under exogeneity and sometimes useful as a quick approximation, but it has a fundamental structural defect — it can generate predicted probabilities outside $[0, 1]$. For inference on probabilities, a model that respects the unit interval is required.

### The Logistic Function

The solution is to pass the linear index $\eta = X\beta$ through the **logistic (sigmoid) function**:

$$\sigma(\eta) = \frac{1}{1 + e^{-\eta}} = \frac{e^\eta}{1 + e^\eta}$$

This maps any real-valued input to $(0, 1)$, which is exactly the range of a probability. The logistic regression model is therefore:

$$P(y_i = 1 \mid x_i) = \sigma(x_i^\top \beta) = \frac{1}{1 + e^{-x_i^\top \beta}}$$

Written in terms of the linear index $\eta_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}$:

$$P(y_i = 1 \mid x_i) = \frac{1}{1 + e^{-\eta_i}}$$

The model is linear in $\eta$, not in $P(y = 1 \mid X)$. This distinction drives every interpretation that follows.

### The Log-Odds (Logit) Representation

Rearranging gives the equivalent **logit** form. Define the **odds** as $P(y=1\mid x) / P(y=0\mid x)$. Then:

$$\log\left(\frac{P(y_i = 1 \mid x_i)}{1 - P(y_i = 1 \mid x_i)}\right) = x_i^\top \beta = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}$$

The log-odds (logit) is linear in $X$ and $\beta$. The model is therefore linear on the logit scale, nonlinear on the probability scale.

### Assumptions

| Assumption | Statement |
|---|---|
| **Binary outcome** | $y_i \in \{0, 1\}$, each drawn from a Bernoulli distribution |
| **Correct link function** | The logit of $P(y=1\mid X)$ is linear in $X$ |
| **Independence** | Observations are independent given $X$ |
| **No perfect separation** | No linear combination of predictors perfectly predicts $y$ |
| **No perfect multicollinearity** | $\mathrm{rank}(X) = p + 1$ |

Logistic regression does **not** require normally distributed errors, homoscedasticity, or linearity between $y$ and $X$. It does not have an error term $\varepsilon$ in the same sense as OLS — the stochasticity comes entirely from the Bernoulli distribution of $y$.

---

## 2. Maximum Likelihood Estimation

### The Likelihood

Since $y_i \mid x_i \sim \mathrm{Bernoulli}(\pi_i)$ where $\pi_i = P(y_i = 1 \mid x_i)$, the probability of observing a single outcome is:

$$P(y_i \mid x_i) = \pi_i^{y_i}(1 - \pi_i)^{1 - y_i}$$

Assuming independence across observations, the joint likelihood is:

$$\mathcal{L}(\beta) = \prod_{i=1}^n \pi_i^{y_i}(1 - \pi_i)^{1 - y_i}$$

### The Log-Likelihood Objective

Taking logs (which preserves the same maximizer and is numerically more tractable):

$$\ell(\beta) = \sum_{i=1}^n \left[ y_i \log \pi_i + (1 - y_i) \log(1 - \pi_i) \right]$$

Substituting $\pi_i = \sigma(x_i^\top \beta)$ and simplifying using $\log(1 - \sigma(\eta)) = -\log(1 + e^\eta)$:

$$\ell(\beta) = \sum_{i=1}^n \left[ y_i x_i^\top \beta - \log(1 + e^{x_i^\top \beta}) \right]$$

MLE finds:

$$\hat{\beta} = \underset{\beta}{\arg\max} \; \ell(\beta)$$

Unlike OLS, there is **no closed-form solution**. The log-likelihood is strictly concave in $\beta$ (guaranteeing a unique global maximum), and the solution is found iteratively.

### Newton-Raphson / IRLS

The standard algorithm is **Newton-Raphson**, which iterates:

$$\beta^{(t+1)} = \beta^{(t)} - \left[\nabla^2 \ell(\beta^{(t)})\right]^{-1} \nabla \ell(\beta^{(t)})$$

The gradient (score) and Hessian are:

$$\nabla \ell(\beta) = X^\top (y - \pi), \qquad \nabla^2 \ell(\beta) = -X^\top W X$$

where $W = \mathrm{diag}(\pi_i(1 - \pi_i))$ is a diagonal weight matrix. Substituting, the update becomes:

$$\beta^{(t+1)} = (X^\top W X)^{-1} X^\top W z$$

where $z = X\beta^{(t)} + W^{-1}(y - \pi)$ is the **working response**. This is **Iteratively Reweighted Least Squares (IRLS)** — at each step, a weighted OLS problem is solved. The weights $\pi_i(1-\pi_i)$ are maximized when $\pi_i = 0.5$ and shrink toward zero as predictions become more extreme.

---

## 3. Properties of the MLE

### Consistency

Under correct model specification and standard regularity conditions, $\hat{\beta} \xrightarrow{p} \beta$ as $n \to \infty$. The MLE is consistent.

### Asymptotic Normality

The MLE is asymptotically normal:

$$\sqrt{n}(\hat{\beta} - \beta) \xrightarrow{d} \mathcal{N}(0, \mathcal{I}(\beta)^{-1})$$

where $\mathcal{I}(\beta) = X^\top W X / n$ is the Fisher information matrix. In practice, the asymptotic covariance matrix of $\hat{\beta}$ is estimated as:

$$\widehat{\mathrm{Var}}(\hat{\beta}) = (X^\top \hat{W} X)^{-1}$$

where $\hat{W} = \mathrm{diag}(\hat{\pi}_i(1 - \hat{\pi}_i))$ uses fitted probabilities. The square roots of the diagonal elements are the **standard errors** reported in logistic regression output.

### Efficiency

The MLE is **asymptotically efficient** — it achieves the Cramér-Rao lower bound, meaning no consistent estimator has a smaller asymptotic variance. This is the analogue of Gauss-Markov for the MLE.

### No Finite-Sample Guarantee

Unlike OLS, logistic regression has no exact finite-sample unbiasedness result. Properties are asymptotic. In small samples, MLE can be substantially biased, particularly when the outcome is rare or predictors are many.

---

## 4. Interpretation of Coefficients

### Log-Odds Scale

Since the logit is linear in $\beta$, each coefficient has a direct interpretation on the log-odds scale: a one-unit increase in $x_j$ increases the log-odds of $y = 1$ by $\beta_j$, holding all other predictors constant.

$$\Delta \log\left(\frac{\pi}{1-\pi}\right) = \beta_j \quad \text{for a one-unit increase in } x_j$$

### Odds Ratios

Exponentiating gives the **odds ratio**:

$$\mathrm{OR}_j = e^{\beta_j}$$

A one-unit increase in $x_j$ multiplies the odds of $y = 1$ by $e^{\beta_j}$, all else equal. This is the most common reporting convention in applied work.

| $e^{\hat{\beta}_j}$ | Interpretation |
|---|---|
| $> 1$ | $x_j$ increases the odds of $y = 1$ |
| $= 1$ | $x_j$ has no effect on the odds |
| $< 1$ | $x_j$ decreases the odds of $y = 1$ |

### Marginal Effects on Probability

The effect of $x_j$ on the **probability** $\pi$ is not constant — it depends on the current level of $\pi$:

$$\frac{\partial \pi_i}{\partial x_{ij}} = \beta_j \cdot \pi_i(1 - \pi_i)$$

The factor $\pi_i(1 - \pi_i)$ is maximized at $\pi_i = 0.5$ and shrinks at the tails. This means the same coefficient $\beta_j$ implies larger probability changes for observations near the decision boundary and smaller changes for observations near 0 or 1.

Two conventions for reporting probability-scale effects:

**Average Marginal Effect (AME)**: Average the individual marginal effects across all observations:

$$\mathrm{AME}_j = \frac{1}{n}\sum_{i=1}^n \hat{\beta}_j \cdot \hat{\pi}_i(1 - \hat{\pi}_i)$$

**Marginal Effect at the Mean (MEM)**: Evaluate at the sample mean of all predictors:

$$\mathrm{MEM}_j = \hat{\beta}_j \cdot \bar{\pi}(1 - \bar{\pi})$$

The AME is generally preferred because it reflects the actual distribution of observations rather than a hypothetical average observation.

---

## 5. Inference

### Wald Tests for Individual Coefficients

Under $H_0: \beta_j = 0$, the Wald statistic is:

$$z_j = \frac{\hat{\beta}_j}{\mathrm{SE}(\hat{\beta}_j)} \xrightarrow{d} \mathcal{N}(0, 1)$$

This is the asymptotic analogue of the t-test in OLS. In finite samples, the approximation degrades — particularly when the outcome is rare or the estimated probability is extreme.

### Likelihood Ratio Test

A more reliable alternative is the **Likelihood Ratio (LR) test**. For a restriction $H_0$, compare the unrestricted log-likelihood $\ell_U$ to the restricted log-likelihood $\ell_R$:

$$\mathrm{LR} = 2(\ell_U - \ell_R) \xrightarrow{d} \chi^2_q$$

where $q$ is the number of restrictions. The LR test is generally preferred over Wald tests in logistic regression because it is invariant to reparameterization and has better finite-sample behavior.

### Pseudo-$R^2$

There is no direct analogue of $R^2$ in logistic regression. Several pseudo- $R^2$ measures have been proposed; the most common are:

**McFadden's $R^2$**:

$$R^2_{\mathrm{McF}} = 1 - \frac{\ell(\hat{\beta})}{\ell(\hat{\beta}_0)}$$

where $\ell(\hat{\beta}_0)$ is the log-likelihood of the intercept-only (null) model. Values above 0.2 are generally considered good fit; above 0.4 is excellent. McFadden's $R^2$ is not comparable to OLS $R^2$ and should not be interpreted as variance explained.

**Cox-Snell $R^2$** and **Nagelkerke $R^2$** are also reported by most software, but McFadden's is the most theoretically grounded.

---

## 6. Separation and Rare Events

### Perfect Separation

**Perfect separation** occurs when a linear combination of predictors perfectly predicts the outcome: all $y_i = 1$ fall on one side of a hyperplane and all $y_i = 0$ on the other. In this case, the MLE does not exist — the log-likelihood has no maximum because $\hat{\beta}$ diverges to infinity.

Symptoms: extremely large coefficient estimates with enormous standard errors, and warnings from the fitting routine. The solution is either to drop the offending predictor, merge categories, or use a penalized estimator such as Firth's logistic regression.

**Quasi-separation** (also called near-separation) produces inflated but finite estimates and similarly unreliable inference.

### Rare Events

When $P(y = 1)$ is small, standard logistic regression systematically **underestimates** the probability of the rare event. In small samples this bias is substantial. Rare-events corrections (e.g., King & Zeng 2001) or resampling-based approaches are recommended when the positive class comprises less than roughly 5% of the data.

---

## 7. Multicollinearity

### The Problem

The same issue that affects OLS affects logistic regression: when predictors are highly correlated, the Fisher information matrix $X^\top W X$ approaches singularity, coefficient variances inflate, and estimates become unstable.

### Variance Inflation Factor

VIF is still applicable, computed from auxiliary OLS regressions of each predictor on the rest — exactly as in linear regression. The same thresholds apply.

| VIF | Interpretation |
|---|---|
| 1 | No collinearity |
| 1–5 | Mild, generally acceptable |
| 5–10 | Moderate concern |
| > 10 | Severe — coefficient estimates unreliable |

### Remedies

Variable removal is the simplest fix. **Ridge-penalized logistic regression** (L2 regularization) is the direct analogue of Ridge regression:

$$\hat{\beta}_{\mathrm{ridge}} = \underset{\beta}{\arg\max} \; \ell(\beta) - \lambda \|\beta\|^2$$

Adding the penalty $\lambda \|\beta\|^2$ regularizes the solution, shrinks coefficients toward zero, and stabilizes estimation under collinearity — at the cost of introducing bias.

---

## 8. Model Evaluation

### Confusion Matrix and Derived Metrics

Given a decision threshold $\tau$ (default 0.5), each observation is classified as $\hat{y} = \mathbf{1}[\hat{\pi} \geq \tau]$. The **confusion matrix** cross-tabulates true vs. predicted class:

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN | FP |
| **Actual 1** | FN | TP |

From this, the primary metrics are:

$$\text{Accuracy} = \frac{TP + TN}{n}, \quad \text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Accuracy is misleading when classes are imbalanced. In those cases, precision, recall, and $F_1$ are more informative. The choice of $\tau$ should reflect the relative cost of false positives vs. false negatives — there is no universal reason to use 0.5.

### ROC Curve and AUC

The **Receiver Operating Characteristic (ROC) curve** plots the True Positive Rate (Recall) against the False Positive Rate across all possible thresholds $\tau$. The **Area Under the Curve (AUC)** summarizes overall discriminative performance:

| AUC | Interpretation |
|---|---|
| $0.5$ | No discrimination (random classifier) |
| $0.5$–$0.7$ | Poor |
| $0.7$–$0.8$ | Acceptable |
| $0.8$–$0.9$ | Excellent |
| $> 0.9$ | Outstanding |

AUC is threshold-independent and invariant to class imbalance, making it the standard summary statistic for binary classifiers.

### Calibration

A classifier is **calibrated** if its predicted probabilities match observed frequencies: among observations where $\hat{\pi} \approx 0.7$, roughly 70% should actually have $y = 1$. A model can have high AUC but poor calibration — it ranks observations correctly but assigns systematically biased probabilities. When the downstream decision depends on the actual probability (e.g., risk scores, medical diagnosis), calibration matters as much as discrimination.

---

## 9. Strengths and Weaknesses

### Strengths

**Probabilistic output**: Unlike hard classifiers, logistic regression outputs calibrated probabilities. These can be used for decision-making under uncertainty, not just classification.

**Interpretability**: Coefficients have a precise interpretation via log-odds and odds ratios. No other comparably predictive model offers this level of transparency.

**Efficiency**: The MLE achieves the Cramér-Rao lower bound asymptotically. For correctly specified models, no consistent estimator can do better.

**Well-understood inference**: Wald tests, LR tests, and confidence intervals are all available and well-understood. Statistical inference on individual predictors is straightforward.

### Weaknesses

**Linear decision boundary**: Logistic regression can only produce a linear boundary in the original feature space. Nonlinear relationships require manual feature engineering.

**Asymptotic guarantees only**: All inferential properties are asymptotic. In small samples, rare events settings, or near-separation, MLE is unreliable.

**Sensitive to separation**: Perfect or quasi-perfect separation causes MLE to fail or produce grossly inflated estimates — a problem that OLS does not share.

**No robust standard errors by default**: Unlike OLS with HC3, standard logistic regression assumes the model is correctly specified. If the link function is wrong, standard errors are invalid. Sandwich (robust) standard errors can be applied but are less commonly reported than in the OLS literature.

---

## 10. Summary

1. **The logistic model** estimates $P(y = 1 \mid X) = \sigma(X\beta)$. It is linear on the log-odds scale, nonlinear on the probability scale. Coefficients measure effects on the log-odds, not on probabilities directly.

2. **MLE maximizes** the Bernoulli log-likelihood. There is no closed-form solution; the problem is solved iteratively via IRLS. The log-likelihood is strictly concave, guaranteeing a unique global maximum when it exists.

3. **Interpretation** requires care. Raw coefficients are log-odds differences. Exponentiate to get odds ratios. Compute AME to get probability-scale effects that vary across observations.

4. **Inference** is asymptotic. Wald tests are standard but LR tests are preferred for their better finite-sample properties and invariance to reparameterization.

5. **Separation** invalidates the MLE. Large coefficients with enormous standard errors are the warning sign. The remedy is penalization or removal of the separating predictor.

6. **Evaluation** requires more than accuracy. AUC measures discrimination; calibration measures probability accuracy. Both matter, and the right trade-off between precision and recall depends on the problem's cost structure.

---

## References

- Greene, W. H. (2012). *Econometric Analysis* (7th ed.). Pearson.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
- King, G., & Zeng, L. (2001). Logistic regression in rare events data. *Political Analysis*, 9(2), 137–163.
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.

---

**Next Lecture**: Logistic Regression Diagnostics
