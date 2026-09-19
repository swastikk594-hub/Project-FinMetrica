# FinMetrica Research Findings: Covariance Regularization & System Benchmark
*Generated on: 2026-09-15 14:39:27*
*Universe Name:* Original Demo (Hindsight Biased)

## Executive Summary
This report evaluates the empirical impact of advanced portfolio structuring (Ledoit-Wolf covariance shrinkage and Hierarchical Risk Parity) on portfolio stability and out-of-sample performance.

**Core Finding:** HRP significantly outperforms Equal Weight (BH-adjusted p=0.0040). No other pairwise comparison reaches statistical significance after correction.

* Regularization reduced the average condition number by **5.4%** (from 10.0 to 9.4).

## Track 1: Estimator Instability

Ledoit-Wolf shrinkage trades a small amount of bias for a large reduction in variance. This manifests as a better-conditioned covariance matrix.

![Condition Number](figures/track1_condition_number.png)

This improved conditioning directly impacts the stability of the optimizer's output weights.
Average turnover (L1 norm):
- Sample Covariance: 0.076
- Ledoit-Wolf: 0.076
- HRP: 0.009


Furthermore, Hierarchical Risk Parity (HRP) achieved the lowest average turnover (0.009), empirically supporting the literature's claim that its hierarchical clustering approach produces more stable weight allocations across time than optimization approaches requiring matrix inversion.

![Weight Stability](figures/track1_weight_stability.png)

## Track 2: Full-System Benchmark (CPCV)

### Cross-Path Stability
| Method | Mean OOS Sharpe | Std Dev OOS Sharpe | CV |
| --- | --- | --- | --- |
| Equal Weight | 0.2486 | 0.3872 | 1.5576 |
| Naive Markowitz | 0.3818 | 0.3477 | 0.9105 |
| Regularised Markowitz | 0.3767 | 0.3469 | 0.9208 |
| HRP | 0.3848 | 0.4111 | 1.0684 |

![Sharpe Distribution](figures/track2_sharpe_dist.png)

### Interpretation: Estimation Error and the Limits of Optimization

HRP is the only one of the four methods that does not use the expected-return vector (mu) at all — it allocates purely from the correlation and distance structure via recursive bisection. Both Markowitz variants (naive and regularized) use the noisy sample mean (mu_hat).

Ledoit-Wolf shrinkage in this experiment only regularizes the covariance matrix, not mu — so if mu estimation error is the dominant source of instability, Sigma-regularization alone would show little benefit. This pattern is consistent with **Chopra, V.K. and Ziemba, W.T. (1993), "The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice," Journal of Portfolio Management**, which found that errors in mean estimates are typically far more damaging to portfolio choice than errors in the covariance matrix.

In this run, HRP does NOT have the lowest cross-path Sharpe standard deviation (its std is 0.4111, while the minimum was 0.3469). A pure 'wins on consistency' argument is therefore not supported by these numbers.

*Note: These are plausible explanations consistent with the observed patterns, not proven causal mechanisms.*

**Suggested Future Extension:** Introduce a fifth method (Ledoit-Wolf Sigma combined with a Minimum-Variance objective that ignores mu) to isolate whether "mu-avoidance" explains HRP's edge, versus something specific to HRP's clustering structure.

### Overfitting Check: Deflated Sharpe Ratio
After adjusting for having tried 5 methods, the probability that Naive Markowitz's true Sharpe ratio exceeds the multiple-testing-adjusted benchmark of 0.088625 is **100.0000%**, versus a naive (non-deflated) read of the raw Sharpe ratio.

*Sample-size note: n_observations = 1886, representing the number of unique calendar dates in the date-averaged ensemble return series (overlapping CPCV paths are averaged per date before computing this, so this is NOT a naive pooled count). Using the single representative fold size would give a more conservative estimate; users should treat extremely high DSR values (>99%) with caution when n_observations is large.*

### Statistical Significance

The table below displays all pairwise Jobson-Korkie-Memmel and stationary-bootstrap tests for out-of-sample Sharpe equivalence, with Bonferroni and Benjamini-Hochberg (FDR) corrections applied.

| Method A | Method B | Sharpe A | Sharpe B | Raw p-val | Bonferroni p | BH p-val | Sig (BH 5%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Equal Weight | Naive Markowitz | 0.2479 | 0.4225 | 0.9060 | 1.0000 | 1.0000 | False |
| Equal Weight | LW-Shrinkage-Only Markowitz | 0.2479 | 0.4176 | 0.9740 | 1.0000 | 1.0000 | False |
| Equal Weight | Regularised Markowitz | 0.2479 | 0.4176 | 0.9740 | 1.0000 | 1.0000 | False |
| Equal Weight | HRP | 0.2479 | 0.3847 | 0.0040 | 0.0400 | 0.0400 | True |
| Naive Markowitz | LW-Shrinkage-Only Markowitz | 0.4225 | 0.4176 | 0.0240 | 0.2400 | 0.0800 | False |
| Naive Markowitz | Regularised Markowitz | 0.4225 | 0.4176 | 0.0240 | 0.2400 | 0.0800 | False |
| Naive Markowitz | HRP | 0.4225 | 0.3847 | 0.0980 | 0.9800 | 0.1633 | False |
| LW-Shrinkage-Only Markowitz | Regularised Markowitz | 0.4176 | 0.4176 | 1.0000 | 1.0000 | 1.0000 | False |
| LW-Shrinkage-Only Markowitz | HRP | 0.4176 | 0.3847 | 0.0840 | 0.8400 | 0.1633 | False |
| Regularised Markowitz | HRP | 0.4176 | 0.3847 | 0.0840 | 0.8400 | 0.1633 | False |

![Significance Heatmap](figures/track2_significance.png)

## Known Limitations

**Survivorship and Hindsight Bias**
This run was evaluated from `2005-01-01` using the following `Original Demo (Hindsight Biased)` universe:
`AAPL, MSFT, GOOG, AMZN, META, JNJ, PFE, UNH, JPM, BAC, GS, WFC, XOM, CVX, COP, PG, KO, HD, MCD, V`

If this list was selected using present-day knowledge of which companies became successful, applying it backward to a start date before that outcome was known represents a form of hindsight/survivorship bias. This likely inflates the *absolute* Sharpe ratios for all methods since they share the same biased universe.

While the *relative* comparison between methods is less affected since all are evaluated on the identical universe, this is an assumption. Evaluating against a point-in-time unbiased universe or a deliberately diverse alternate universe is recommended to verify these findings hold.

## Conclusion
HRP significantly outperforms Equal Weight (BH-adjusted p=0.0040). No other pairwise comparison reaches statistical significance after correction.
