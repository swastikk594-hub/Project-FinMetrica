import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.stats.multitest as multitest
from typing import Dict, List, Any, Tuple
import statsmodels.api as sm
import statsmodels.stats.sandwich_covariance as sw

class RobustnessDiagnostics:
    """
    Robustness diagnostics to compare results across normalization methods.
    """
    def __init__(self, baseline_normalization: str = 'cross_sectional_zscore'):
        self.baseline_normalization = baseline_normalization

    def compute_sign_stability(self, delta_r2_by_norm: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        For each factor, compute proportion of dates where delta_R2 has same sign across normalizations.
        Formula: \frac{1}{N} \\sum_{i=1}^N I(sign(x_{i,norm_a}) == sign(x_{i,norm_b}))
        
        Args:
            delta_r2_by_norm: Dict mapping normalization name to DataFrame of delta_R2 over time for each factor.
                              Assuming DataFrame has index=date, columns=factors.

        Returns:
            DataFrame with rows=factors, columns=normalizations, values=proportion of same sign vs baseline.
        """
        if self.baseline_normalization not in delta_r2_by_norm:
            raise ValueError(f"Baseline '{self.baseline_normalization}' not found in data.")
            
        baseline_df = delta_r2_by_norm[self.baseline_normalization]
        baseline_sign = np.sign(baseline_df)
        
        results = {}
        for factor in baseline_df.columns:
            factor_res = {}
            for norm, df in delta_r2_by_norm.items():
                if factor not in df.columns:
                    factor_res[norm] = np.nan
                    continue
                norm_sign = np.sign(df[factor])
                # Proportion of same sign
                aligned = pd.concat([baseline_sign[factor], norm_sign], axis=1, join='inner')
                if aligned.empty:
                    factor_res[norm] = np.nan
                else:
                    match = (aligned.iloc[:, 0] == aligned.iloc[:, 1]).mean()
                    factor_res[norm] = float(match)
            results[factor] = factor_res
            
        return pd.DataFrame.from_dict(results, orient='index')

    def compute_magnitude_change(self, delta_r2_by_norm: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute absolute and relative magnitude change of delta_R2 vs baseline normalization.
        abs_change = |mean(\\Delta R^2_{norm}) - mean(\\Delta R^2_{base})|
        rel_change = abs_change / |mean(\\Delta R^2_{base})|
        
        Args:
            delta_r2_by_norm: Dict mapping normalization name to DataFrame of delta_R2 (index=date, cols=factors)

        Returns:
            DataFrame with columns: factor, normalization, abs_change, rel_change.
        """
        if self.baseline_normalization not in delta_r2_by_norm:
            raise ValueError(f"Baseline '{self.baseline_normalization}' not found in data.")
            
        baseline_means = delta_r2_by_norm[self.baseline_normalization].mean()
        
        records = []
        for norm, df in delta_r2_by_norm.items():
            if norm == self.baseline_normalization:
                continue
            norm_means = df.mean()
            
            for factor in df.columns:
                if factor not in baseline_means.index:
                    continue
                
                b_mean = baseline_means[factor]
                n_mean = norm_means[factor]
                
                abs_change = abs(n_mean - b_mean)
                rel_change = abs_change / abs(b_mean) if b_mean != 0 else np.nan
                
                records.append({
                    'factor': factor,
                    'normalization': norm,
                    'abs_change': float(abs_change),
                    'rel_change': float(rel_change)
                })
                
        return pd.DataFrame(records)

    def compute_summary_statistics(self, delta_r2_by_norm: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute min, max, range, and standard deviation of mean delta_R2 across normalizations.
        """
        # First, compute mean delta_R2 for each factor, for each normalization
        means_by_norm = {}
        for norm, df in delta_r2_by_norm.items():
            means_by_norm[norm] = df.mean()
            
        # Combine into a single DataFrame: rows=factors, columns=normalizations
        combined = pd.DataFrame(means_by_norm)
        
        results = []
        for factor in combined.index:
            row = combined.loc[factor].dropna()
            if len(row) > 0:
                results.append({
                    'factor': factor,
                    'min': float(row.min()),
                    'max': float(row.max()),
                    'range': float(row.max() - row.min()),
                    'std': float(row.std()) if len(row) > 1 else 0.0
                })
                
        return pd.DataFrame(results)

    def compute_statistical_robustness(self, coeff_series_by_norm: Dict[str, Dict[str, pd.Series]]) -> pd.DataFrame:
        """
        For each factor and normalization, compute Newey-West 95% CI.
        Flag whether CI excludes zero.
        CI = \\bar{\\beta} \\pm 1.96 \\cdot SE_{NW}

        Args:
            coeff_series_by_norm: Dict mapping normalization name to Dict mapping factor to Series of coefficients.

        Returns:
            DataFrame with columns: factor, normalization, mean_coeff, nw_t_stat, ci_lower, ci_upper, significant.
        """
        records = []
        for norm, factors_dict in coeff_series_by_norm.items():
            for factor, series in factors_dict.items():
                ts = series.dropna()
                if len(ts) < 2:
                    continue
                
                max_lags = int(4 * (len(ts) / 100)**(2/9))
                mean_coef = float(ts.mean())
                
                model = sm.OLS(ts, np.ones(len(ts))).fit()
                cov_nw = sw.cov_hac(model, nlags=max_lags)
                
                se_nw = np.sqrt(cov_nw[0, 0])
                if se_nw == 0:
                    t_stat = np.nan
                    ci_lower, ci_upper = mean_coef, mean_coef
                else:
                    t_stat = mean_coef / se_nw
                    ci_lower = mean_coef - 1.96 * se_nw
                    ci_upper = mean_coef + 1.96 * se_nw
                    
                significant = (ci_lower > 0) or (ci_upper < 0)
                
                records.append({
                    'factor': factor,
                    'normalization': norm,
                    'mean_coeff': mean_coef,
                    'nw_t_stat': float(t_stat),
                    'ci_lower': float(ci_lower),
                    'ci_upper': float(ci_upper),
                    'significant': bool(significant)
                })
                
        return pd.DataFrame(records)

    def compute_rank_robustness(self, factor_scores_by_norm: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute Spearman rank correlation of security rankings across normalizations.
        \rho = 1 - \frac{6 \\sum d_i^2}{n(n^2 - 1)}
        
        Args:
            factor_scores_by_norm: Dict mapping norm to DataFrame of factor scores (index=assets, columns=factors).
                                   Assume single point in time or aggregated.
        
        Returns:
            Correlation matrix across normalizations (average across factors).
        """
        norms = list(factor_scores_by_norm.keys())
        matrix = pd.DataFrame(index=norms, columns=norms, dtype=float)
        
        for n1 in norms:
            for n2 in norms:
                df1 = factor_scores_by_norm[n1]
                df2 = factor_scores_by_norm[n2]
                
                common_factors = list(set(df1.columns) & set(df2.columns))
                common_assets = list(set(df1.index) & set(df2.index))
                
                corrs = []
                for f in common_factors:
                    s1 = df1.loc[common_assets, f]
                    s2 = df2.loc[common_assets, f]
                    # spearman corr
                    corr, _ = stats.spearmanr(s1, s2, nan_policy='omit')
                    if not np.isnan(corr):
                        corrs.append(corr)
                        
                matrix.loc[n1, n2] = np.mean(corrs) if corrs else np.nan
                
        return matrix

    def compute_predictive_robustness(self, r2_by_norm: Dict[str, float]) -> pd.DataFrame:
        """
        Compare out-of-sample R2 of full model across normalizations.
        
        Args:
            r2_by_norm: Dict mapping norm to R2.

        Returns:
            DataFrame with normalization, r2, delta_from_baseline.
        """
        records = []
        baseline_r2 = r2_by_norm.get(self.baseline_normalization, np.nan)
        
        for norm, r2 in r2_by_norm.items():
            records.append({
                'normalization': norm,
                'r2': r2,
                'delta_from_baseline': r2 - baseline_r2 if not np.isnan(baseline_r2) else np.nan
            })
            
        return pd.DataFrame(records)

    def generate_robustness_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate all robustness diagnostics into a single report dict.
        
        Args:
            results: dict containing outputs of various compute_* methods.
            
        Returns:
            Dict representing the report.
        """
        report = {}
        if 'summary_statistics' in results:
            report['Summary Statistics'] = results['summary_statistics'].to_dict('records')
        if 'sign_stability' in results:
            report['Sign Stability'] = results['sign_stability'].to_dict()
        if 'magnitude_change' in results:
            report['Magnitude Change'] = results['magnitude_change'].to_dict('records')
        if 'statistical_robustness' in results:
            report['Statistical Robustness'] = results['statistical_robustness'].to_dict('records')
        if 'rank_robustness' in results:
            report['Rank Robustness'] = results['rank_robustness'].to_dict()
        if 'predictive_robustness' in results:
            report['Predictive Robustness'] = results['predictive_robustness'].to_dict('records')
            
        return report

    def apply_multiple_testing_correction(self, p_values: pd.Series, method: str = 'holm') -> pd.Series:
        """
        Apply Holm-Bonferroni or BH correction to p-values.
        
        Args:
            p_values: Series of p-values.
            method: 'holm' or 'fdr_bh'
            
        Returns:
            Series of corrected p-values.
        """
        mask = p_values.notna()
        valid_p = p_values[mask]
        
        if len(valid_p) == 0:
            return p_values.copy()
            
        reject, pvals_corrected, _, _ = multitest.multipletests(valid_p, alpha=0.05, method=method)
        
        res = p_values.copy()
        res.loc[mask] = pvals_corrected
        return res
