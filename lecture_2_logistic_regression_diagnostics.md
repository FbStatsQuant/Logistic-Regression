# Logistic Regression Diagnostics

## Introduction

Fitting a logistic regression model is only the beginning. Before drawing any inference from coefficients or using predicted probabilities for decisions, the model must be interrogated for structural failures, influential observations, assumption violations, and poor fit. This is the diagnostic phase.

The diagnostic framework for logistic regression is fundamentally different from that of OLS. The Gauss-Markov assumptions — linearity in $y$, homoscedasticity, normally distributed errors — are irrelevant here. Logistic regression has its own set of assumptions, its own residual definitions, and its own failure modes. Applying OLS intuitions uncritically is a methodological error.

This lecture covers the complete diagnostic toolkit: goodness of fit, residual analysis, influence measures, multicollinearity, linearity in the log-odds, separation, and the discrimination-calibration distinction. Each diagnostic addresses a specific question about where and how the model may be failing.

---

## 1. Why OLS Diagnostics Don't Transfer

In OLS, the residual $e_i = y_i - \hat{y}_i$ has a natural interpretation: it is the portion of $y_i$ unexplained by the model. Under correct specification, residuals are i.i.d. $\mathcal{N}(0, \sigma^2)$, which gives you a single homogeneous scale for identifying outliers and heteroscedasticity.

In logistic regression, the analogous quantity $y_i - \hat{\pi}_i$ has non-constant variance by construction: $\mathrm{Var}(y_i - \hat{\pi}_i) = \pi_i(1 - \pi_i)$. This variance is not a model failure — it is the correct Bernoulli variance. It means:

- There is no single residual scale; standardization requires case-specific weighting.
- Residuals near 0 and 1 are naturally compressed; observations near $\hat{\pi} = 0.5$ have the largest potential residuals.
- Standard normality-of-residuals plots are meaningless without appropriate transformation.

Estimation is also different. OLS diagnostics exploit the hat matrix $H = X(X^\top X)^{-1} X^\top$ derived from least squares. In logistic regression, the analogue is the weighted hat matrix:

$$H = W^{1/2} X (X^\top W X)^{-1} X^\top W^{1/2}$$

where $W = \mathrm{diag}(\hat{\pi}_i(1 - \hat{\pi}_i))$. Leverage depends on both covariate position and predicted probability.

---

## 2. Goodness of Fit

### 2.1 Likelihood Ratio (Deviance) Test

The primary test for overall model utility. Compare the fitted model against the **null model** (intercept only) using the likelihood ratio statistic:

$$G^2 = -2\left[\ell(\hat{\beta}_{\mathrm{null}}) - \ell(\hat{\beta}_{\mathrm{full}})\right] \sim \chi^2(p)$$

where $p$ is the number of added parameters. A significant $G^2$ means the model explains more than chance. This should always be reported — a pseudo-$R^2$ alone does not establish statistical significance of the model.

The **null deviance** is $D_\mathrm{null} = -2\ell(\hat{\beta}_{\mathrm{null}})$ and the **residual deviance** is $D_\mathrm{res} = -2\ell(\hat{\beta}_{\mathrm{full}})$. Their difference equals $G^2$.

### 2.2 Hosmer-Lemeshow Test

Groups observations into $g$ bins (typically deciles) of predicted probability, then compares observed and expected event counts within each bin:

$$H = \sum_{k=1}^{g} \frac{(O_k - E_k)^2}{E_k(1 - E_k/n_k)} \sim \chi^2(g - 2)$$

where $O_k$ is the observed count of events in bin $k$, $E_k = \sum_{i \in k} \hat{\pi}_i$ is the expected count, and $n_k$ is the bin size. A non-significant $p$-value is taken as evidence of adequate fit.

**Known limitations:**

| Limitation | Consequence |
|---|---|
| Low power in small samples | May fail to detect real misfit |
| Overpowered in large samples | May reject adequately fitting models |
| Sensitive to number of groups $g$ | Results can change with grouping choice |
| Non-significant ≠ good fit | Only rules out detectable misfit |

Use as one signal among several, not as a standalone verdict.

### 2.3 Pseudo-$R^2$ Statistics

No single $R^2$ analogue exists for logistic regression. The following are all in common use, and all measure different things:

| Statistic | Formula | Notes |
|---|---|---|
| McFadden's $R^2$ | $1 - \ell(\hat{\beta}_\mathrm{full}) / \ell(\hat{\beta}_\mathrm{null})$ | Proportional reduction in log-likelihood |
| Cox-Snell $R^2$ | $1 - (L_\mathrm{null} / L_\mathrm{full})^{2/n}$ | Cannot reach 1 for binary outcomes |
| Nagelkerke $R^2$ | Cox-Snell divided by its theoretical maximum | Rescaled to $[0, 1]$ |

McFadden's $R^2$ values of $0.2$–$0.4$ are considered excellent. Do not benchmark against OLS $R^2$ conventions. These statistics summarize fit but carry no causal meaning and should not substitute for residual diagnostics.

---

## 3. Residuals

The term "residual" in logistic regression is ambiguous — there are multiple definitions, each suited to a different diagnostic purpose. Conflating them produces incorrect conclusions.

### 3.1 Response Residuals

The most intuitive definition, on the probability scale:

$$e_i^{(r)} = y_i - \hat{\pi}_i$$

Useful for checking calibration but not for outlier detection, because variance is non-constant: $\mathrm{Var}(e_i^{(r)}) = \hat{\pi}_i(1 - \hat{\pi}_i)$. A large response residual near $\hat{\pi} = 0.5$ is less unusual than a small one near $\hat{\pi} = 0.01$.

### 3.2 Pearson Residuals

Standardize response residuals by their theoretical standard deviation:

$$e_i^{(P)} = \frac{y_i - \hat{\pi}_i}{\sqrt{\hat{\pi}_i(1 - \hat{\pi}_i)}}$$

Under correct specification, Pearson residuals have approximately unit variance. Their sum of squares is the **Pearson $\chi^2$ statistic**, which under the null of correct specification is approximately $\chi^2(n - p - 1)$ when there are grouped observations.

Flag $|e_i^{(P)}| > 2$ as potential outliers; $|e_i^{(P)}| > 3$ as strong candidates for investigation.

### 3.3 Deviance Residuals

The signed square root of each observation's contribution to total deviance:

$$d_i = \mathrm{sign}(y_i - \hat{\pi}_i) \cdot \sqrt{-2\left[y_i \log \hat{\pi}_i + (1 - y_i)\log(1 - \hat{\pi}_i)\right]}$$

The sum $\sum_i d_i^2$ equals the residual deviance. Deviance residuals are more symmetric and closer to normally distributed than Pearson residuals, particularly at extreme probabilities. **They are preferred for outlier detection and residual plots.**

### 3.4 Studentized Residuals

Adjust deviance residuals for leverage:

$$r_i = \frac{d_i}{\sqrt{1 - h_{ii}}}$$

where $h_{ii}$ is the diagonal element of the weighted hat matrix $H$. High-leverage observations have their residuals deflated — the model is pulled toward them, making their raw residuals artificially small. Studentizing corrects for this. Observations with $|r_i| > 2$ warrant examination.

### 3.5 Diagnostic Plots for Residuals

| Plot | What to look for |
|---|---|
| Deviance residuals vs. fitted log-odds | Any systematic pattern indicates misspecification |
| Deviance residuals vs. each continuous predictor | Curves or trends indicate missed nonlinearity |
| Index plot (residuals vs. observation number) | Clusters of misfit, temporal patterns |
| Q-Q plot of deviance residuals | Severe departures from normality (only approximate) |

A random, structureless scatter in residual vs. fitted plots is the target. Any discernible pattern is a diagnostic failure requiring investigation.

---

## 4. Influence Analysis

An observation can distort model estimates in two ways: by having an unusual response (outlier), by occupying an extreme position in covariate space (high leverage), or both (influential). Influence analysis separates and combines these.

### 4.1 Leverage

The hat values $h_{ii}$ from the weighted hat matrix measure how much each observation controls its own fitted value. In logistic regression:

$$h_{ii} = x_i^\top (X^\top W X)^{-1} x_i \cdot \hat{\pi}_i(1 - \hat{\pi}_i)$$

**Key distinction from OLS**: leverage depends not only on covariate position but also on predicted probability. Observations near $\hat{\pi} = 0.5$ receive the highest weight $\pi(1-\pi) = 0.25$ and therefore have the most potential for leverage. Observations with extreme predicted probabilities near 0 or 1 are downweighted.

**Rule of thumb**: $h_{ii} > 2(p+1)/n$ is high leverage.

### 4.2 Cook's Distance

Measures the aggregate shift in all fitted values when observation $i$ is deleted:

$$D_i = \frac{e_i^{(P)2} \cdot h_{ii}}{p(1 - h_{ii})^2}$$

This captures both outlier magnitude (via the Pearson residual) and leverage. Common thresholds:

| Threshold | Interpretation |
|---|---|
| $D_i > 1$ | Highly influential — warrants investigation |
| $D_i > 4/n$ | Conservative flag in large samples |

An influence plot showing $D_i$ against $h_{ii}$ simultaneously, with point size proportional to studentized residuals, is more informative than any single index.

### 4.3 DFBETAs

Measures the change in each individual coefficient when observation $i$ is deleted, expressed in standard error units:

$$\mathrm{DFBETA}_{ij} = \frac{\hat{\beta}_j - \hat{\beta}_j^{(-i)}}{\widehat{\mathrm{SE}}(\hat{\beta}_j)}$$

Flag $|\mathrm{DFBETA}_{ij}| > 2/\sqrt{n}$. Unlike Cook's $D$, DFBETAs tell you *which* coefficients are being distorted by *which* observations — Cook's $D$ only tells you that *some* coefficient is affected.

### 4.4 DFFITS

Measures the change in the fitted value for observation $i$ when it is excluded:

$$\mathrm{DFFITS}_i = \frac{\hat{\pi}_i - \hat{\pi}_i^{(-i)}}{\sqrt{\hat{\pi}_i(1 - \hat{\pi}_i) h_{ii}/n}}$$

Flag $|\mathrm{DFFITS}_i| > 2\sqrt{p/n}$. Provides a case-level summary analogous to Cook's $D$ but focused on the change in a single fitted value.

### 4.5 Workflow for Influence Analysis

High leverage alone is not a problem — it only matters when combined with a large residual. The recommended sequence is:

1. Plot Cook's $D$ against observation index. Flag observations above the threshold.
2. For flagged observations, examine DFBETAs across all coefficients to identify which parameters are most affected.
3. Investigate flagged cases substantively. Are they data entry errors? Unusual but genuine cases? Subgroups the model systematically misspecifies?
4. Refit without flagged observations and compare coefficients. Report sensitivity if findings change materially.

Deletion of influential observations requires substantive justification — statistical influence alone is not sufficient grounds for exclusion.

---

## 5. Multicollinearity

Collinearity among predictors inflates the variance of $\hat{\beta}$, makes estimates unstable, can reverse coefficient signs, and renders individual tests unreliable. Detection and quantification proceed identically to OLS.

### 5.1 Variance Inflation Factor

Regress each predictor $X_j$ on all remaining predictors using OLS (these auxiliary regressions use only the design matrix, not the binary response):

$$\mathrm{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the coefficient of determination from the auxiliary regression.

| VIF | Interpretation |
|---|---|
| $1$ | No collinearity |
| $1$–$5$ | Mild — generally acceptable |
| $5$–$10$ | Moderate — investigate |
| $> 10$ | Severe — estimates unreliable |

### 5.2 Condition Number

The condition number $\kappa$ of $X^\top X$ is the ratio of its largest to smallest eigenvalue. $\kappa > 30$ indicates serious collinearity; $\kappa > 100$ is severe. It captures system-level collinearity that VIF may miss when multiple predictors are jointly problematic.

### 5.3 Logistic-Specific Aggravation

Collinearity is more damaging in logistic regression when predicted probabilities are near 0 or 1. The Fisher information matrix is $X^\top W X$, and the weights $w_i = \hat{\pi}_i(1 - \hat{\pi}_i)$ shrink toward zero at the extremes, reducing the effective information available for estimation. An already-collinear design matrix becomes more numerically precarious in this regime.

### 5.4 Remedies

Ridge-penalized logistic regression adds an L2 penalty to the log-likelihood:

$$\hat{\beta}_\mathrm{ridge} = \underset{\beta}{\arg\max} \; \ell(\beta) - \lambda \|\beta\|^2$$

This regularizes $X^\top W X + 2\lambda I$, ensuring invertibility and shrinking coefficients toward zero. The trade-off is bias. Alternatively, remove or consolidate highly collinear predictors based on domain knowledge.

---

## 6. Linearity in the Log-Odds

Logistic regression assumes the log-odds of the outcome is a linear function of each continuous predictor:

$$\log\left(\frac{\pi_i}{1 - \pi_i}\right) = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}$$

If this linearity assumption is violated for a continuous predictor, the coefficient estimate for that predictor is biased and the residual pattern will reflect the missed curvature. Three methods are used to check this.

### 6.1 Box-Tidwell Test

For each continuous predictor $X_j$, augment the model with the interaction term $X_j \cdot \ln(X_j)$. If the coefficient on this term is statistically significant, the log-odds relationship with $X_j$ is not linear:

$$\mathrm{logit}(\pi) = \cdots + \beta_j X_j + \gamma_j (X_j \ln X_j) + \cdots$$

Test $H_0: \gamma_j = 0$. Requires $X_j > 0$; apply a shift constant if the predictor includes non-positive values.

### 6.2 Smoothed Residual Plots (LOWESS)

Plot deviance residuals against each continuous predictor and overlay a LOWESS (locally weighted scatterplot smoothing) curve. A flat curve centered at zero indicates linearity holds. A systematic curve indicates curvature that the linear term does not capture. This is a visual diagnostic but the most sensitive to subtle nonlinearity.

### 6.3 Empirical Logit Plot

Bin each continuous predictor into groups (e.g., deciles), compute the observed log-odds $\log(\bar{y}_k / (1 - \bar{y}_k))$ within each bin, and plot against the bin mean. Under linearity, these points should fall approximately on a straight line. If the relationship is curved, a transformation (e.g., $\log X_j$, $\sqrt{X_j}$) or a spline term is indicated.

---

## 7. Separation and Quasi-Complete Separation

Separation is a logistic-regression-specific pathology with no OLS equivalent. It occurs when a linear combination of predictors perfectly (or nearly perfectly) predicts the outcome. The MLE does not exist under complete separation.

### 7.1 Complete Separation

A predictor $X_j$ (or linear combination) perfectly predicts $y$: all observations with $y = 1$ have $X_j > c$ and all observations with $y = 0$ have $X_j < c$ for some threshold $c$. The likelihood is maximized as $\hat{\beta}_j \to \infty$. The MLE is undefined.

### 7.2 Quasi-Complete Separation

Perfect separation holds for a subset of observations. The MLE technically exists but is substantially biased and has severely inflated variance.

### 7.3 Symptoms

| Symptom | What it signals |
|---|---|
| $|\hat{\beta}_j| \gg 10$ | Possible separation on predictor $j$ |
| $\widehat{\mathrm{SE}}(\hat{\beta}_j)$ very large | MLE is wandering in flat likelihood region |
| Optimizer convergence warnings | Likelihood did not flatten — iterations diverged |
| A categorical predictor with zero events in one cell | Structural quasi-separation |

### 7.4 Remedies

| Remedy | When to use |
|---|---|
| **Firth's penalized likelihood** | Primary fix — adds Jeffrey's prior as penalty, reduces bias, MLE always exists |
| **Exact logistic regression** | Small samples where asymptotic approximation is unreliable |
| **Collapse sparse categories** | Categorical predictor has near-empty cells |
| **Remove the predictor** | If substantively peripheral and separation is structural |

Firth's method is the standard recommendation. It modifies the score equations to $U(\beta) - \frac{1}{2}\mathrm{tr}[I(\beta)^{-1} \partial I(\beta) / \partial \beta] = 0$, which corresponds to maximizing a penalized log-likelihood and produces finite estimates in all cases.

---

## 8. Discrimination and Calibration

These are distinct model properties that are frequently conflated. A model can excel at one while failing at the other.

### 8.1 Discrimination

**Discrimination** measures the model's ability to rank cases by their true outcome status. The standard summary is the **AUC-ROC**: the probability that a randomly selected event case receives a higher predicted probability than a randomly selected non-event case.

$$\mathrm{AUC} = P(\hat{\pi}_i > \hat{\pi}_j \mid y_i = 1, y_j = 0)$$

| AUC | Interpretation |
|---|---|
| $0.5$ | Random — no discrimination |
| $0.5$–$0.7$ | Poor |
| $0.7$–$0.8$ | Acceptable |
| $0.8$–$0.9$ | Excellent |
| $> 0.9$ | Outstanding |

For imbalanced outcomes, the **AUC-PR (Precision-Recall curve)** is more informative. AUC-ROC is insensitive to class imbalance because it treats false positives and false negatives symmetrically across all thresholds, inflating apparent performance when negative cases vastly outnumber positive cases.

### 8.2 Calibration

**Calibration** measures whether predicted probabilities match observed event rates. Among observations where $\hat{\pi} \approx 0.3$, approximately 30% should actually have $y = 1$.

**Calibration plot**: Bin observations by decile of $\hat{\pi}$, compute mean predicted probability and observed event rate within each bin, and plot observed against predicted. A perfectly calibrated model falls on the 45° diagonal.

**Calibration slope**: Regress observed events on $\mathrm{logit}(\hat{\pi})$:

$$\mathrm{logit}(\pi_i) = \alpha + \delta \cdot \mathrm{logit}(\hat{\pi}_i)$$

Under perfect calibration, $\alpha = 0$ and $\delta = 1$.

| Slope $\delta$ | Interpretation |
|---|---|
| $\delta < 1$ | Predictions too extreme (overfitting) |
| $\delta = 1$ | Perfect calibration |
| $\delta > 1$ | Predictions too compressed (underfitting) |

**Calibration-in-the-large**: Is the overall mean prediction close to the observed event rate? This checks systematic over- or under-prediction across all observations.

### 8.3 When Each Matters

| Use case | Priority |
|---|---|
| Ranking or prioritizing cases | Discrimination (AUC) |
| Predicted probabilities drive decisions | Calibration |
| Treatment eligibility thresholds | Both |
| Risk score communication to patients | Calibration |

A model that ranks cases well but assigns badly calibrated probabilities will mislead any downstream user who interprets the predicted probabilities as real event probabilities.

---

## 9. Overdispersion

Overdispersion occurs when the observed variance in the response exceeds the nominal Bernoulli variance. It is most relevant when the response is grouped (multiple binary trials per covariate pattern), less common with individual-level binary outcomes.

### 9.1 Detection

Estimate the dispersion parameter:

$$\hat{\phi} = \frac{\chi^2_\mathrm{Pearson}}{n - p - 1} = \frac{\sum_i e_i^{(P)2}}{n - p - 1}$$

Under correct specification, $\hat{\phi} \approx 1$. $\hat{\phi} > 1$ indicates overdispersion; $\hat{\phi} < 1$ indicates underdispersion (less common, can indicate model overfit).

### 9.2 Consequences of Ignoring Overdispersion

Standard logistic regression assumes $\mathrm{Var}(y_i) = \pi_i(1 - \pi_i)$. If actual variance is $\phi \cdot \pi_i(1 - \pi_i)$ with $\phi > 1$, then standard errors are underestimated by a factor of $\sqrt{\phi}$, and p-values are anti-conservative.

### 9.3 Remedies

**Quasi-likelihood**: Multiplies the variance function by $\hat{\phi}$, scaling all standard errors by $\sqrt{\hat{\phi}}$. Coefficients are unchanged; inference is corrected. This is the simplest fix when $\phi$ is moderately above 1.

**Beta-binomial model**: Explicitly models extra-binomial variability by allowing the success probability to vary according to a Beta distribution. More principled than quasi-likelihood but requires more assumptions.

---

## 10. Implementation in Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
import scipy.stats as stats

# ── Data preparation ──────────────────────────────────────────────────────────
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Subset to keep example tractable
cols = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
X_sub = X_scaled[cols].copy()
X_const = sm.add_constant(X_sub)

# ── 1. Fit model ──────────────────────────────────────────────────────────────
model = sm.Logit(y, X_const).fit()
print(model.summary())

# ── 2. Goodness of fit ────────────────────────────────────────────────────────
# Likelihood ratio test (G²)
G2 = model.llnull * (-2) - model.llf * (-2)
# Note: statsmodels gives llr and llr_pvalue directly
print(f"\nLikelihood Ratio G²: {model.llr:.3f}, p = {model.llr_pvalue:.4f}")
print(f"McFadden R²: {model.prsquared:.3f}")

# Hosmer-Lemeshow test
def hosmer_lemeshow(y_true, y_pred, g=10):
    df = pd.DataFrame({'y': y_true, 'yhat': y_pred})
    df['decile'] = pd.qcut(df['yhat'], q=g, duplicates='drop')
    grouped = df.groupby('decile')
    obs = grouped['y'].sum()
    exp = grouped['yhat'].sum()
    n   = grouped['y'].count()
    H = ((obs - exp)**2 / (exp * (1 - exp / n))).sum()
    p = 1 - stats.chi2.cdf(H, df=g - 2)
    return H, p

fitted = model.predict()
hl_stat, hl_p = hosmer_lemeshow(y, fitted)
print(f"\nHosmer-Lemeshow: H = {hl_stat:.3f}, p = {hl_p:.4f}")

# ── 3. Residuals ──────────────────────────────────────────────────────────────
response_resid  = y - fitted
pearson_resid   = response_resid / np.sqrt(fitted * (1 - fitted))

# Deviance residuals
deviance_resid  = np.sign(response_resid) * np.sqrt(
    -2 * (y * np.log(fitted.clip(1e-10)) + (1 - y) * np.log((1 - fitted).clip(1e-10)))
)

# Hat values (leverage) from the weighted hat matrix
W       = np.diag(fitted * (1 - fitted))
X_arr   = X_const.values
XtWX    = X_arr.T @ W @ X_arr
XtWX_inv = np.linalg.inv(XtWX)
H_diag  = np.array([X_arr[i] @ XtWX_inv @ X_arr[i] * fitted[i] * (1 - fitted[i])
                    for i in range(len(y))])

# Studentized deviance residuals
stud_resid = deviance_resid / np.sqrt(1 - H_diag)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(fitted, deviance_resid, alpha=0.4, s=20)
axes[0].axhline(0, color='red', lw=1)
axes[0].set(xlabel='Fitted probability', ylabel='Deviance residual',
            title='Deviance residuals vs. fitted')

axes[1].scatter(fitted, stud_resid, alpha=0.4, s=20)
axes[1].axhline([2, -2], color='orange', lw=1, linestyle='--')
axes[1].axhline(0, color='red', lw=1)
axes[1].set(xlabel='Fitted probability', ylabel='Studentized residual',
            title='Studentized residuals vs. fitted')

sm.qqplot(deviance_resid, line='s', ax=axes[2], alpha=0.4)
axes[2].set_title('Q-Q plot of deviance residuals')

plt.tight_layout()
plt.savefig('residual_plots.png', dpi=150)
plt.show()

# ── 4. Influence analysis ─────────────────────────────────────────────────────
p = X_const.shape[1]
cooks_d = (pearson_resid**2 * H_diag) / (p * (1 - H_diag)**2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].stem(np.arange(len(cooks_d)), cooks_d, markerfmt='C0o', basefmt='k-')
axes[0].axhline(4 / len(y), color='red', linestyle='--', label='4/n threshold')
axes[0].set(xlabel='Observation', ylabel="Cook's D", title="Cook's Distance")
axes[0].legend()

axes[1].scatter(H_diag, stud_resid, c=cooks_d, cmap='RdYlGn_r',
                s=40, alpha=0.7, edgecolors='k', linewidths=0.3)
axes[1].axhline([2, -2], color='orange', linestyle='--')
axes[1].set(xlabel='Leverage $h_{ii}$', ylabel='Studentized residual',
            title='Influence plot')
plt.colorbar(axes[1].collections[0], ax=axes[1], label="Cook's D")

plt.tight_layout()
plt.savefig('influence_plots.png', dpi=150)
plt.show()

# Identify highly influential observations
influential = np.where(cooks_d > 4 / len(y))[0]
print(f"\nInfluential observations (Cook's D > 4/n): {len(influential)} cases")

# ── 5. Multicollinearity ──────────────────────────────────────────────────────
vif_values = pd.Series(
    [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
    index=X_const.columns
)
print("\nVIF:\n", vif_values.round(2))

cond_number = np.linalg.cond(X_const.values)
print(f"Condition number: {cond_number:.1f}")

# ── 6. Linearity in the log-odds (Box-Tidwell) ────────────────────────────────
def box_tidwell_test(X_df, y, cols):
    X_bt = X_df.copy()
    for col in cols:
        shift = X_bt[col] - X_bt[col].min() + 0.01  # ensure positive
        X_bt[f'{col}_bt'] = shift * np.log(shift)
    X_bt_const = sm.add_constant(X_bt)
    model_bt = sm.Logit(y, X_bt_const).fit(disp=False)
    bt_results = model_bt.pvalues[[c for c in model_bt.pvalues.index if '_bt' in c]]
    return bt_results

bt_pvals = box_tidwell_test(X_sub, y, cols)
print("\nBox-Tidwell p-values (H₀: linear log-odds):")
print(bt_pvals.round(4))

# ── 7. Separation check ───────────────────────────────────────────────────────
large_coef = model.params[np.abs(model.params) > 10]
large_se   = model.bse[model.bse > 100]

if not large_coef.empty:
    print(f"\nWARNING: Possible separation — large coefficients:\n{large_coef}")
if not large_se.empty:
    print(f"WARNING: Possible separation — large standard errors:\n{large_se}")
else:
    print("\nNo separation symptoms detected.")

# ── 8. Discrimination and calibration ─────────────────────────────────────────
# AUC-ROC
auc = roc_auc_score(y, fitted)
fpr, tpr, _ = roc_curve(y, fitted)

# AUC-PR
prec, rec, _ = precision_recall_curve(y, fitted)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ROC
axes[0].plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
axes[0].legend()

# Precision-Recall
axes[1].plot(rec, prec, lw=2, color='darkorange')
axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')

# Calibration plot
n_bins = 10
bin_edges = np.percentile(fitted, np.linspace(0, 100, n_bins + 1))
bin_labels = np.digitize(fitted, bin_edges[1:-1])
cal_df = pd.DataFrame({'y': y, 'yhat': fitted, 'bin': bin_labels})
cal_grp = cal_df.groupby('bin').agg(obs=('y', 'mean'), pred=('yhat', 'mean'))

axes[2].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
axes[2].scatter(cal_grp['pred'], cal_grp['obs'], s=60, zorder=5)
axes[2].plot(cal_grp['pred'], cal_grp['obs'], lw=1.5, alpha=0.7)
axes[2].set(xlabel='Mean predicted probability', ylabel='Observed event rate',
            title='Calibration Plot')
axes[2].legend()

plt.tight_layout()
plt.savefig('discrimination_calibration.png', dpi=150)
plt.show()

print(f"\nAUC-ROC: {auc:.3f}")

# Calibration slope
cal_slope_model = sm.Logit(y, sm.add_constant(np.log(fitted / (1 - fitted).clip(1e-10)))).fit(disp=False)
print(f"Calibration slope: {cal_slope_model.params.iloc[1]:.3f} (ideal = 1.0)")

# ── 9. Overdispersion ─────────────────────────────────────────────────────────
phi_hat = (pearson_resid**2).sum() / (len(y) - p)
print(f"\nDispersion parameter φ̂ = {phi_hat:.3f} (ideal ≈ 1.0)")
if phi_hat > 1.5:
    print("WARNING: Possible overdispersion — consider quasi-likelihood.")
```

---

## 11. Summary of Diagnostics

| Diagnostic | Tool | Flags |
|---|---|---|
| Overall model utility | Likelihood ratio test $G^2$ | $p > 0.05$ — model explains nothing |
| Goodness of fit | Hosmer-Lemeshow, calibration plot | Poor observed-vs-expected agreement |
| Outliers | Deviance residuals, studentized residuals | $|r_i| > 2$ |
| Leverage | Hat values $h_{ii}$ | $h_{ii} > 2(p+1)/n$ |
| Influence | Cook's $D$, DFBETAs | $D_i > 4/n$, $|\mathrm{DFBETA}| > 2/\sqrt{n}$ |
| Multicollinearity | VIF, condition number | VIF $> 10$, $\kappa > 30$ |
| Linearity in log-odds | Box-Tidwell, LOWESS | Significant $\gamma_j$, curved smoother |
| Separation | Coefficient magnitude, SE magnitude | $|\hat{\beta}| \gg 10$, $\widehat{\mathrm{SE}} \gg 100$ |
| Discrimination | AUC-ROC, AUC-PR | AUC $< 0.7$ for most applications |
| Calibration | Calibration plot, slope | Slope $\neq 1$, systematic deviation from diagonal |
| Overdispersion | $\hat{\phi}$ | $\hat{\phi} > 1.5$ in grouped data |

The diagnostic process is iterative, not a checklist. Each finding may reveal a deeper issue that requires re-examination of earlier steps. A model that passes all diagnostics is not necessarily correct — it is a model for which the available diagnostics have not detected failure.

---

## References

- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
- Pregibon, D. (1981). Logistic regression diagnostics. *Annals of Statistics*, 9(4), 705–724.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.
- Harrell, F. E. (2015). *Regression Modeling Strategies* (2nd ed.). Springer.
- Steyerberg, E. W. (2009). *Clinical Prediction Models*. Springer.
- Box, G. E. P., & Tidwell, P. W. (1962). Transformation of the independent variables. *Technometrics*, 4(4), 531–550.

---

**Previous Lecture**: Logistic Regression
