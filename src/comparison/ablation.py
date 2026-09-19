"""
Ablation Study Module
======================
Tests what happens when each module is removed.
This validates that every module provides independent information.

METHODOLOGY:
  For each module M_k, compute composite score WITHOUT M_k:
  Score_ablated_k = Σ_{i ≠ k} w_i * M_i / Σ_{i ≠ k} w_i
  
  Compare ablated score vs full score:
  - High similarity → M_k provides little unique information
  - Low similarity → M_k provides unique, non-redundant information
  
  Also run walk-forward backtest without each module and compare
  Sharpe ratios and Information Ratios.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

class AblationStudy:
    """
    Class to conduct ablation studies on modular scores.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AblationStudy with a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration parameters.
        """
        self.config = config
        np.random.seed(config.get("random_seed", 42))

    def run_module_ablation(self, module_scores: Dict[str, pd.Series], risk_score: pd.Series, aggregator: Callable) -> pd.DataFrame:
        """
        For each module, remove it and compute composite score.

        Parameters
        ----------
        module_scores : Dict[str, pd.Series]
            Dictionary mapping module names to their respective score series.
        risk_score : pd.Series
            Risk adjustment score.
        aggregator : Callable
            Function that takes (module_scores: Dict, risk_score: pd.Series) and returns composite score pd.Series.

        Returns
        -------
        pd.DataFrame
            DataFrame showing: full_score, ablated_score, score_change, pct_change
            (using mean or aggregated metrics over the series).
        """
        logger.info("Running module ablation study.")
        try:
            full_score = aggregator(module_scores, risk_score)
            results = []
            
            for module_name in module_scores.keys():
                ablated_scores = {k: v for k, v in module_scores.items() if k != module_name}
                ablated_composite = aggregator(ablated_scores, risk_score)
                
                # Compare series (e.g., mean absolute difference)
                score_change = np.mean(np.abs(full_score - ablated_composite))
                pct_change = score_change / np.mean(np.abs(full_score)) if np.mean(np.abs(full_score)) != 0 else 0.0
                
                results.append({
                    "module": module_name,
                    "score_change": float(score_change),
                    "pct_change": float(pct_change)
                })
                
            return pd.DataFrame(results).set_index("module")
        except Exception as e:
            logger.error(f"Error during module ablation: {e}")
            raise

    def rank_module_importance(self, ablation_results: pd.DataFrame) -> pd.DataFrame:
        """
        Ranks modules by absolute score change.

        Parameters
        ----------
        ablation_results : pd.DataFrame
            DataFrame containing 'score_change' from run_module_ablation.

        Returns
        -------
        pd.DataFrame
            DataFrame with module, score_impact, rank, interpretation.
        """
        logger.info("Ranking module importance.")
        try:
            ranked = ablation_results.copy()
            ranked['score_impact'] = ranked['score_change'].abs()
            ranked['rank'] = ranked['score_impact'].rank(ascending=False, method='min')
            ranked = ranked.sort_values('rank')
            
            def interpret(impact):
                if impact > self.config.get("high_impact_threshold", 0.5):
                    return "Critical component, highly unique information."
                elif impact > self.config.get("medium_impact_threshold", 0.1):
                    return "Valuable component, adds some unique information."
                else:
                    return "Redundant component, provides little unique information."
                    
            ranked['interpretation'] = ranked['score_impact'].apply(interpret)
            return ranked
        except Exception as e:
            logger.error(f"Error ranking module importance: {e}")
            raise

    def explain_ablation(self, results: pd.DataFrame) -> str:
        """
        Generates a human-readable summary of the ablation study.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame containing ranked module importance.

        Returns
        -------
        str
            Summary explanation.
        """
        logger.info("Generating ablation study explanation.")
        try:
            explanation = "Ablation Study Results:\n"
            explanation += "Imagine building a sports team. If you remove your star player and the team still wins, "
            explanation += "that player wasn't as critical as you thought. Here, we remove each module to see how much the final score changes.\n\n"
            
            for idx, row in results.iterrows():
                explanation += f"Module '{idx}' (Rank {int(row['rank'])}):\n"
                explanation += f"  - Score Impact: {row['score_impact']:.4f}\n"
                explanation += f"  - Interpretation: {row['interpretation']}\n\n"
                
            return explanation
        except Exception as e:
            logger.error(f"Error generating ablation explanation: {e}")
            raise
