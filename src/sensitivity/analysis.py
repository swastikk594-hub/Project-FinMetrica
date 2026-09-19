"""
Sensitivity Analysis Module
============================
Systematically vary model parameters and measure impact on output.

MATHEMATICS:
  For each parameter θ with current value θ₀:
  Sensitivity = ΔOutput / Δθ  ≈ (f(θ₀+Δ) - f(θ₀-Δ)) / (2Δ)
  
  This is numerical differentiation (central differences).
  We compute this for each parameter in the config.

PARAMETERS TESTED:
  - momentum_window: [63, 126, 189, 252, 315] trading days
  - var_confidence: [0.90, 0.95, 0.99]
  - risk_penalty_lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
  - rolling_factor_window: [24, 36, 48, 60] months
  - max_position_size: [0.20, 0.30, 0.40, 0.50]
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, List

logger = logging.getLogger(__name__)

class SensitivityAnalysis:
    """
    Class to conduct sensitivity analysis on model parameters.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SensitivityAnalysis with a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration parameters.
        """
        self.config = config
        np.random.seed(config.get("random_seed", 42))

    def parameter_sweep(self, param_name: str, param_values: List[float], scoring_function: Callable, baseline_score: float) -> pd.DataFrame:
        """
        Sweep a parameter over a range of values and compute the sensitivity.

        Parameters
        ----------
        param_name : str
            Name of the parameter.
        param_values : List[float]
            List of values to test.
        scoring_function : Callable
            Function that takes (param_name, param_value) and returns a scalar score.
        baseline_score : float
            Score of the model at baseline parameter settings.

        Returns
        -------
        pd.DataFrame
            DataFrame containing: param_value, output_score, sensitivity, pct_change_from_baseline.
        """
        logger.info(f"Sweeping parameter: {param_name}")
        try:
            results = []
            
            for val in param_values:
                score = scoring_function(param_name, val)
                
                pct_change = (score - baseline_score) / baseline_score if baseline_score != 0 else 0.0
                
                results.append({
                    "param_value": val,
                    "output_score": score,
                    "pct_change_from_baseline": pct_change
                })
                
            df = pd.DataFrame(results)
            # Calculate numerical sensitivity (central difference logic simplified over sorted values)
            df = df.sort_values("param_value").reset_index(drop=True)
            df["sensitivity"] = df["output_score"].diff() / df["param_value"].diff()
            df["sensitivity"] = df["sensitivity"].fillna(0.0)
            return df
        except Exception as e:
            logger.error(f"Error during parameter sweep for {param_name}: {e}")
            raise

    def run_full_sensitivity(self, config: Dict[str, Any], scoring_function: Callable) -> Dict[str, pd.DataFrame]:
        """
        Sweeps all parameters defined in config['sensitivity']['parameters'].

        Parameters
        ----------
        config : Dict[str, Any]
            System configuration containing sensitivity parameters.
        scoring_function : Callable
            Function that evaluates a given parameter set.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Mapping of parameter names to their sweep results DataFrames.
        """
        logger.info("Running full sensitivity analysis.")
        try:
            parameters = config.get('sensitivity', {}).get('parameters', {})
            baseline_score = scoring_function("baseline", None)
            
            all_results = {}
            for param, values in parameters.items():
                df = self.parameter_sweep(param, values, scoring_function, baseline_score)
                all_results[param] = df
            return all_results
        except Exception as e:
            logger.error(f"Error during full sensitivity run: {e}")
            raise

    def rank_parameter_importance(self, all_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Ranks parameters by their total sensitivity.

        Parameters
        ----------
        all_results : Dict[str, pd.DataFrame]
            Results from run_full_sensitivity.

        Returns
        -------
        pd.DataFrame
            DataFrame with parameters ranked by sensitivity score.
        """
        logger.info("Ranking parameter importance.")
        try:
            ranking = []
            for param, df in all_results.items():
                scores = df["output_score"]
                mean_abs_score = scores.abs().mean()
                if mean_abs_score != 0:
                    sensitivity_score = scores.std() / mean_abs_score
                else:
                    sensitivity_score = 0.0
                
                ranking.append({
                    "Parameter": param,
                    "Sensitivity Score": sensitivity_score
                })
                
            rank_df = pd.DataFrame(ranking)
            rank_df['Rank'] = rank_df['Sensitivity Score'].rank(ascending=False, method='min')
            return rank_df.sort_values("Rank").reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error ranking parameter importance: {e}")
            raise

    def print_sensitivity_report(self, all_results: Dict[str, pd.DataFrame], ranked: pd.DataFrame) -> None:
        """
        Prints a formatted sensitivity report.

        Parameters
        ----------
        all_results : Dict[str, pd.DataFrame]
            Results from run_full_sensitivity.
        ranked : pd.DataFrame
            Results from rank_parameter_importance.
        """
        logger.info("Printing sensitivity report.")
        try:
            # Using rich layout format in string output
            print("\n=== Sensitivity Analysis Report ===")
            print("Parameter Importance Ranking:")
            print(ranked.to_string(index=False))
            print("\nDetailed Sweeps:")
            for param, df in all_results.items():
                print(f"\nParameter: {param}")
                print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print("===================================\n")
        except Exception as e:
            logger.error(f"Error printing sensitivity report: {e}")
            raise
