# Cross-Universe Robustness Check

This document compares the findings between the Original Demo universe (highly concentrated, hindsight-biased large cap tech/financials) and the Diversified Alternate universe (point-in-time realistic, uncorrelated sectors).

## Robust Claims (Hold Across Both Universes)
- **Turnover & Stability:** HRP produces the lowest weight turnover and most stable allocations across time in both universes, corroborating claims that its distance-based hierarchical clustering avoids the extreme instability of covariance matrix inversion.
- **Covariance Conditioning:** Ledoit-Wolf shrinkage consistently improves the condition number of the covariance matrix over the sample covariance matrix.
- **The Power of 1/N:** In neither universe does any complex optimization method definitively and robustly outperform the naive Equal Weight (1/N) benchmark across the board. The DeMiguel (2009) assertion that 1/N is extremely difficult to beat due to estimation error holds strongly.

## Fragile Claims (Fail to Replicate in Alternate Universe)
- **HRP Outperformance:** In earlier unfixed runs, HRP appeared to outperform Equal Weight in the Original Demo universe. However, after fixing deterministic seeds and running strictly out-of-sample, this outperformance disappeared entirely in BOTH universes (p > 0.05). This suggests any previous HRP performance advantage was an artifact of specific noisy seeds rather than a universal property.

## 10-Pair Significance Comparison

| Pair | Original Demo Sig (BH 5%) | Alt Universe Sig (BH 5%) | Status |
|------|---------------------------|--------------------------|--------|
| Equal Weight vs Naive Markowitz | False | False | Replicates (Null) |
| Equal Weight vs LW-Shrinkage-Only Markowitz | False | False | Replicates (Null) |
| Equal Weight vs Regularised Markowitz | False | False | Replicates (Null) |
| Equal Weight vs HRP | False | False | Replicates (Null) |
| Naive Markowitz vs LW-Shrinkage-Only Markowitz | False | False | Replicates (Null) |
| Naive Markowitz vs Regularised Markowitz | False | False | Replicates (Null) |
| Naive Markowitz vs HRP | False | False | Replicates (Null) |
| LW-Shrinkage-Only Markowitz vs Regularised Markowitz | False | False | Replicates (Null) |
| LW-Shrinkage-Only Markowitz vs HRP | False | False | Replicates (Null) |
| Regularised Markowitz vs HRP | False | False | Replicates (Null) |

## Conclusion
The claims about estimator stability and turnover hold robustly across datasets. The claims about out-of-sample Sharpe ratio outperformance do not. All performance claims must be explicitly caveated in the final paper as being sensitive to universe selection.