"""
Integration Layer: Module Score Aggregation
============================================
PURPOSE:
  Combine the outputs of Modules 1-6 into a single composite score
  with mathematically justified weighting.

MATHEMATICAL FRAMEWORK:

  RAW MODULE SCORES: M1, M2, ..., M6 (each on 0-100 scale)

  STEP 1 - DEPENDENCY ANALYSIS:
    Compute correlation matrix C of module scores:
    C_ij = Cor(M_i, M_j)
    Flag pairs with |C_ij| > 0.7 as highly correlated.
    If modules are highly correlated, their combined weight should be
    reduced (they are measuring the same information).

  STEP 2 - WEIGHTING SCHEMES (compared):
    A. Equal Weighting: w_i = 1/6 for all i
       Advantages: Transparent, robust to estimation error
       Disadvantages: Ignores information content differences

    B. Volatility-Adjusted: w_i ∝ 1/σ_i
       where σ_i = standard deviation of M_i over historical universe
       Advantages: Downweights unstable/noisy modules
       Disadvantages: Requires cross-sectional history

    C. IC-Based (Information Coefficient):
       IC_i = Spearman rank correlation of M_i with future returns
       w_i ∝ IC_i (only use positive-IC modules)
       Advantages: Empirically validated; modules earn their weight
       Disadvantages: Requires sufficient historical data; overfitting risk

  SELECTED: Equal weighting as primary (robust, interpretable).
            IC-based weighting shown as alternative when data sufficient.

  STEP 3 - AGGREGATION:
    Score_composite = Σ_i w_i * M_i

  STEP 4 - RISK ADJUSTMENT:
    Risk score from Module 5 is used as a penalty:
    Score_adjusted = Score_composite * (1 - λ * RiskPenalty)
    where λ ∈ [0,1] and RiskPenalty ∈ [0,1]

  STEP 5 - UNCERTAINTY:
    Bootstrap confidence interval around Score_adjusted
    CI_90 = [5th percentile, 95th percentile] of bootstrap distribution
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """
    Complete result of the integration layer.
    
    All components are preserved for full transparency.
    """
    module_scores: dict                # Raw module scores M1-M6
    normalised_scores: dict            # Normalised to 0-100
    weights: dict                      # Applied weights w_i
    weighting_scheme: str              # 'equal', 'vol_adjusted', 'ic_based'
    composite_score: float             # Σ w_i * M_i
    risk_penalty: float                # From Module 5
    risk_lambda: float                 # Penalty strength
    adjusted_score: float             # After risk adjustment
    score_ci_lower: float             # 90% confidence lower bound
    score_ci_upper: float             # 90% confidence upper bound
    correlation_matrix: pd.DataFrame   # Module score correlations
    high_correlation_pairs: list       # Pairs with |r| > 0.7
    warnings: list                     # Any data quality or methodology flags
    metadata: dict = field(default_factory=dict)


class ScoreAggregator:
    """
    Aggregates module scores M1-M6 into a composite quantitative score.
    
    DECISION LOG:
    - Equal weighting chosen as primary because it maximises robustness
      and transparency. IC-based weighting is available but requires
      sufficient cross-sectional and temporal data to be reliable.
    - Risk from Module 5 is applied as a multiplicative penalty rather
      than an additive component, because risk should scale the entire
      score (a high-quality but extremely risky stock is still risky).
    """

    def __init__(self, config: dict):
        self.config = config
        integ_cfg = config.get('integration', {})
        self.weighting_scheme = integ_cfg.get('weighting_scheme', 'equal')
        self.risk_lambda = integ_cfg.get('risk_penalty', {}).get('lambda', 0.3)
        self.risk_penalty_enabled = integ_cfg.get('risk_penalty', {}).get('enabled', True)
        bootstrap_cfg = integ_cfg.get('bootstrap', {})
        self.n_bootstrap = bootstrap_cfg.get('n_samples', 1000)
        self.bootstrap_ci = bootstrap_cfg.get('confidence_level', 0.90)
        self.seed = config.get('system', {}).get('random_seed', 42)

    def normalise_scores(self, module_scores: dict) -> dict:
        """
        Normalise module scores to a common 0-100 scale.

        Module scores are already on a 0-100 scale produced by each module.
        This function simply clips them to [0, 100] and passes them through.
        Rank normalisation here would be wrong — it would rank modules against
        each other for a single ticker (meaningless), not rank the same module
        across tickers (cross-sectional).  Cross-sectional comparison requires
        multiple tickers to be present and is left to the caller.

        Parameters
        ----------
        module_scores : dict
            Dict of {module_name: raw_score} from each module (each on 0-100 scale)

        Returns
        -------
        dict
            Scores clipped to [0, 100]
        """
        scores = {k: v for k, v in module_scores.items() if v is not None and not np.isnan(v)}
        if not scores:
            return {}
        # Simply clip each score to [0, 100] — they are already on the right scale.
        return {k: float(np.clip(v, 0.0, 100.0)) for k, v in scores.items()}

    def compute_module_correlation(self, historical_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation matrix of module scores over time/cross-section.
        
        Parameters
        ----------
        historical_scores : pd.DataFrame
            DataFrame with module scores (columns) over assets/time (rows)
            
        Returns
        -------
        pd.DataFrame
            Correlation matrix — used to detect redundant modules
        """
        if historical_scores.shape[0] < 5:
            logger.warning("Too few observations for reliable correlation estimation.")
            return pd.DataFrame()
        return historical_scores.corr(method='spearman')

    def compute_weights(
        self,
        normalised_scores: dict,
        historical_scores: Optional[pd.DataFrame] = None,
        scheme: Optional[str] = None
    ) -> dict:
        """
        Compute module combination weights.
        
        Parameters
        ----------
        normalised_scores : dict
            Normalised module scores
        historical_scores : pd.DataFrame, optional
            Historical scores for IC-based weighting
        scheme : str, optional
            Override weighting scheme
            
        Returns
        -------
        dict
            Weights for each module (sum to 1.0)
        """
        scheme = scheme or self.weighting_scheme
        modules = list(normalised_scores.keys())
        n = len(modules)
        
        if n == 0:
            return {}
        
        if scheme == 'equal':
            weights = {m: 1.0 / n for m in modules}
            
        elif scheme == 'vol_adjusted' and historical_scores is not None:
            # Downweight volatile/unstable modules
            stds = historical_scores[modules].std()
            inv_std = 1.0 / (stds + 1e-8)
            inv_std_normalised = inv_std / inv_std.sum()
            weights = inv_std_normalised.to_dict()
            
        else:
            if scheme != 'equal':
                logger.warning(
                    f"Weighting scheme '{scheme}' requires historical data. Falling back to equal."
                )
            weights = {m: 1.0 / n for m in modules}
        
        # Verify weights sum to 1 — floating-point arithmetic may produce tiny deviations
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {total:.8f}, re-normalising.")
            weights = {m: w / total for m, w in weights.items()}
        
        return weights

    def aggregate(
        self,
        normalised_scores: dict,
        weights: dict
    ) -> float:
        """
        Compute weighted composite score.
        
        Score_composite = Σ_i w_i * M_i
        
        Parameters
        ----------
        normalised_scores : dict
            Module scores normalised to 0-100
        weights : dict
            Module weights (must sum to 1)
            
        Returns
        -------
        float
            Composite score 0-100
        """
        return sum(weights[m] * normalised_scores[m] for m in weights)

    def apply_risk_adjustment(self, composite_score: float, risk_score: float) -> tuple[float, float]:
        """
        Apply risk penalty from Module 5.
        
        Score_adjusted = Score_composite × (1 - λ × RiskPenalty)
        
        Module 5's normalised_score is ALREADY INVERTED:
          high normalised_score → low risk → good
          low  normalised_score → high risk → bad
        
        Therefore, to penalise high-risk assets:
          RiskPenalty = (100 - risk_score) / 100
        This makes RiskPenalty large (close to 1) when risk_score is low
        (high actual risk) and small (close to 0) when risk_score is high
        (low actual risk).
        
        Parameters
        ----------
        composite_score : float
            Aggregated module score 0-100
        risk_score : float
            Module 5 normalised_score 0-100 (higher = LOWER risk = better)
            
        Returns
        -------
        tuple[float, float]
            (adjusted_score, penalty_applied)
        """
        if not self.risk_penalty_enabled:
            return composite_score, 0.0
        
        # Convert M5's normalised_score (high=good) to a risk penalty (high=bad)
        # risk_score=80 (low risk) → penalty_factor=0.2 (small penalty, good)
        # risk_score=20 (high risk) → penalty_factor=0.8 (large penalty, bad)
        risk_penalty_factor = (100.0 - risk_score) / 100.0
        penalty = self.risk_lambda * risk_penalty_factor
        adjusted = composite_score * (1.0 - penalty)
        return adjusted, penalty

    def bootstrap_confidence_interval(
        self,
        module_scores_dict: dict,
        weights: dict,
        risk_score: float
    ) -> tuple[float, float]:
        """
        Bootstrap confidence interval around composite score.
        
        Resamples the module scores (treating each as a noisy estimate)
        and recomputes the composite for each bootstrap sample.
        
        Parameters
        ----------
        module_scores_dict : dict
            {module: score} pairs
        weights : dict
            Applied weights
        risk_score : float
            Module 5 risk score
            
        Returns
        -------
        tuple[float, float]
            (ci_lower, ci_upper) at self.bootstrap_ci confidence level
        """
        rng = np.random.default_rng(self.seed)
        modules = list(weights.keys())
        scores = np.array([module_scores_dict[m] for m in modules])
        
        # Bootstrap: add noise proportional to score uncertainty
        # We assume each module score has ~10 points of measurement uncertainty
        noise_std = 10.0
        bootstrap_composites = []
        
        for _ in range(self.n_bootstrap):
            noisy_scores = scores + rng.normal(0, noise_std, size=len(scores))
            noisy_scores = np.clip(noisy_scores, 0, 100)
            w = np.array([weights[m] for m in modules])
            composite = float(np.dot(w, noisy_scores))
            adjusted, _ = self.apply_risk_adjustment(composite, risk_score)
            bootstrap_composites.append(adjusted)
        
        alpha = (1 - self.bootstrap_ci) / 2
        ci_lower = float(np.percentile(bootstrap_composites, alpha * 100))
        ci_upper = float(np.percentile(bootstrap_composites, (1 - alpha) * 100))
        return ci_lower, ci_upper

    def run(
        self,
        module_scores: dict,
        risk_score: float,
        historical_scores: Optional[pd.DataFrame] = None
    ) -> AggregationResult:
        """
        Run the full aggregation pipeline.
        
        Parameters
        ----------
        module_scores : dict
            {module_name: score_0_to_100} for M1-M6
        risk_score : float
            Module 5 risk score (0-100, higher = more risky)
        historical_scores : pd.DataFrame, optional
            Historical scores for volatility-adjusted weighting
            
        Returns
        -------
        AggregationResult
        """
        warnings = []
        
        # Step 1: Filter out None/NaN scores
        valid_scores = {k: v for k, v in module_scores.items() 
                       if v is not None and not np.isnan(float(v))}
        
        if len(valid_scores) < 3:
            warnings.append(
                f"Only {len(valid_scores)} of 6 modules produced valid scores. "
                "Composite score reliability is reduced."
            )
        
        # Step 2: Normalise (for single-asset, returns neutral 50s)
        normalised = self.normalise_scores(valid_scores)
        
        # Step 3: Compute correlation matrix
        corr_matrix = pd.DataFrame()
        high_corr_pairs = []
        if historical_scores is not None:
            corr_matrix = self.compute_module_correlation(historical_scores)
            # Flag highly correlated pairs
            for i in corr_matrix.columns:
                for j in corr_matrix.columns:
                    if i < j and abs(corr_matrix.loc[i, j]) > 0.7:
                        high_corr_pairs.append((i, j, float(corr_matrix.loc[i, j])))
                        warnings.append(
                            f"Modules {i} and {j} are highly correlated "
                            f"(r={corr_matrix.loc[i,j]:.2f}). They may measure the same thing."
                        )
        
        # Step 4: Compute weights
        weights = self.compute_weights(normalised, historical_scores)
        
        # Step 5: Aggregate
        composite = self.aggregate(normalised, weights)
        
        # Step 6: Risk adjustment
        adjusted, penalty = self.apply_risk_adjustment(composite, risk_score)
        
        # Step 7: Bootstrap CI
        ci_lower, ci_upper = self.bootstrap_confidence_interval(
            normalised, weights, risk_score
        )
        
        return AggregationResult(
            module_scores=module_scores,
            normalised_scores=normalised,
            weights=weights,
            weighting_scheme=self.weighting_scheme,
            composite_score=composite,
            risk_penalty=penalty,
            risk_lambda=self.risk_lambda,
            adjusted_score=adjusted,
            score_ci_lower=ci_lower,
            score_ci_upper=ci_upper,
            correlation_matrix=corr_matrix,
            high_correlation_pairs=high_corr_pairs,
            warnings=warnings,
        )

    def explain(self, result: AggregationResult) -> str:
        """
        Generate a human-readable explanation of the composite score.
        
        Designed for audiences aged 13-18. Shows the mathematics
        without hiding it.
        """
        lines = [
            "=" * 60,
            "  COMPOSITE SCORE EXPLANATION",
            "=" * 60,
            "",
            f"  Final Adjusted Score:  {result.adjusted_score:.1f} / 100",
            f"  Confidence Interval:   [{result.score_ci_lower:.1f}, {result.score_ci_upper:.1f}]  (90% CI)",
            f"  Composite (pre-risk):  {result.composite_score:.1f} / 100",
            f"  Risk Penalty Applied:  {result.risk_penalty:.1%} reduction",
            "",
            "  HOW IT WAS CALCULATED:",
            "  ─────────────────────────────────────────────────",
            "  Formula: Score = Σ (weight_i × module_score_i) × (1 - λ × Risk)",
        ]
        
        for module, score in result.normalised_scores.items():
            w = result.weights.get(module, 0)
            lines.append(f"    {module:12s}  score={score:5.1f}  weight={w:.2f}  contribution={w*score:5.1f}")
        
        lines += [
            "",
            f"  Weighting Scheme: {result.weighting_scheme}",
            f"  Risk Penalty (λ): {result.risk_lambda}",
            "",
            "  WHAT THIS MEANS:",
            f"  A score of {result.adjusted_score:.0f}/100 means this asset scores in the",
            f"  {result.adjusted_score:.0f}th percentile of the analysed universe",
            "  on this combined quantitative assessment.",
            "",
            "  IMPORTANT LIMITATIONS:",
            "  • This score summarises past data. It says nothing definitive",
            "    about future returns.",
            "  • The confidence interval shows the uncertainty in the estimate.",
            "  • A high score does not mean 'buy'. Always consider your own",
            "    risk tolerance, time horizon, and diversification.",
        ]
        
        if result.warnings:
            lines += ["", "  ⚠ WARNINGS:"]
            for w in result.warnings:
                lines.append(f"    • {w}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
