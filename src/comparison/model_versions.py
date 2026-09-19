"""
Model Comparison Framework
===========================
DECISION: Does mathematical complexity actually improve performance?

We test 4 model versions:

MODEL A (Baseline): Simple momentum + P/E ratio
  - Momentum: 12-month trailing return (from M3)
  - Valuation: P/E z-score (from M2)
  - Equal weights, no risk adjustment

MODEL B (Multi-Factor): Modules 1-4 only
  - Fundamentals, Valuation, Time-Series, Factor Model
  - Equal weighting, no risk adjustment

MODEL C (Risk-Adjusted): Models 1-5
  - All of Model B plus explicit risk management (M5)
  - Risk penalty applied

MODEL D (Full System): All 7 Modules
  - Complete system with macro and portfolio optimisation
  - Risk adjustment and uncertainty quantification

COMPARISON METRICS:
  - Sharpe ratio (out-of-sample)
  - Information ratio vs benchmark
  - Maximum drawdown
  - Stability (rolling Sharpe over time)
  - Complexity (number of parameters/inputs)
  
HYPOTHESIS: The full model should outperform on risk-adjusted basis.
BUT: We must test this empirically and report the actual result,
even if the simpler model performs equally well or better.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelVersionComparison:
    """
    Class for comparing different model complexities.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ModelVersionComparison with config.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration parameters.
        """
        self.config = config
        np.random.seed(config.get("random_seed", 42))

    def define_model_a(self, price_data: pd.DataFrame, fundamental_data: pd.DataFrame) -> Dict[str, float]:
        """
        Baseline model: Simple momentum + P/E ratio.

        Parameters
        ----------
        price_data : pd.DataFrame
            Historical price data.
        fundamental_data : pd.DataFrame
            Fundamental data containing P/E ratios.

        Returns
        -------
        Dict[str, float]
            Dictionary of ticker to score.
        """
        logger.info("Defining Model A.")
        try:
            scores = {}
            for ticker in price_data.columns:
                if len(price_data[ticker]) >= 252:
                    momentum = price_data[ticker].iloc[-1] / price_data[ticker].iloc[-252] - 1
                else:
                    momentum = 0.0
                pe_ratio = fundamental_data.get(ticker, {}).get('pe', np.nan)
                if pd.isna(pe_ratio):
                    pe_score = 0.0
                else:
                    pe_score = - (pe_ratio - 15) / 5.0  # Simple heuristic valuation
                scores[ticker] = 0.5 * momentum + 0.5 * pe_score
            return scores
        except Exception as e:
            logger.error(f"Error defining Model A: {e}")
            raise

    def define_model_b(self, all_module_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Multi-Factor model: Modules 1-4 only.

        Parameters
        ----------
        all_module_results : Dict[str, Dict[str, float]]
            Module results.

        Returns
        -------
        Dict[str, float]
            Composite scores.
        """
        logger.info("Defining Model B.")
        try:
            composite = {}
            modules = ['M1', 'M2', 'M3', 'M4']
            tickers = set()
            for m in modules:
                if m in all_module_results:
                    tickers.update(all_module_results[m].keys())
                    
            for ticker in tickers:
                score = 0.0
                count = 0
                for m in modules:
                    if m in all_module_results and ticker in all_module_results[m]:
                        score += all_module_results[m][ticker]
                        count += 1
                composite[ticker] = score / count if count > 0 else 0.0
            return composite
        except Exception as e:
            logger.error(f"Error defining Model B: {e}")
            raise

    def define_model_c(self, all_module_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Risk-Adjusted model: Modules 1-5.

        Parameters
        ----------
        all_module_results : Dict[str, Dict[str, float]]
            Module results.

        Returns
        -------
        Dict[str, float]
            Composite scores.
        """
        logger.info("Defining Model C.")
        try:
            base_scores = self.define_model_b(all_module_results)
            risk_module = all_module_results.get('M5', {})
            risk_penalty = self.config.get('risk_penalty_lambda', 0.5)
            
            final_scores = {}
            for ticker, score in base_scores.items():
                risk = risk_module.get(ticker, 0.0)
                final_scores[ticker] = score - risk_penalty * risk
            return final_scores
        except Exception as e:
            logger.error(f"Error defining Model C: {e}")
            raise

    def define_model_d(self, all_module_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Full System: All 7 Modules.

        Parameters
        ----------
        all_module_results : Dict[str, Dict[str, float]]
            Module results.

        Returns
        -------
        Dict[str, float]
            Composite scores.
        """
        logger.info("Defining Model D.")
        try:
            base_scores = self.define_model_c(all_module_results)
            macro_module = all_module_results.get('M6', {})
            opt_module = all_module_results.get('M7', {})
            
            macro_multiplier = self.config.get('macro_multiplier', 1.0)
            final_scores = {}
            for ticker, score in base_scores.items():
                macro = macro_module.get(ticker, 0.0)
                opt = opt_module.get(ticker, 0.0)
                final_scores[ticker] = score + macro_multiplier * macro + opt
            return final_scores
        except Exception as e:
            logger.error(f"Error defining Model D: {e}")
            raise

    def compare_models(self, models: Dict[str, pd.Series], returns_df: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
        """
        Compares models across performance metrics.

        Parameters
        ----------
        models : Dict[str, pd.Series]
            Dictionary mapping model name to their respective simulated returns pd.Series.
        returns_df : pd.DataFrame
            Underlying asset returns.
        benchmark_returns : pd.Series
            Benchmark returns series.

        Returns
        -------
        pd.DataFrame
            Comparison table: model × performance metrics.
        """
        logger.info("Comparing models.")
        try:
            results = []
            for model_name, returns in models.items():
                mean_ret = returns.mean() * 252
                volatility = returns.std() * np.sqrt(252)
                sharpe = mean_ret / volatility if volatility != 0 else 0.0
                
                active_returns = returns - benchmark_returns
                tracking_error = active_returns.std() * np.sqrt(252)
                info_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0.0
                
                cumulative = (1 + returns).cumprod()
                peak = cumulative.expanding(min_periods=1).max()
                drawdown = (cumulative - peak) / peak
                max_drawdown = drawdown.min()
                
                results.append({
                    "Model": model_name,
                    "Sharpe Ratio": sharpe,
                    "Information Ratio": info_ratio,
                    "Max Drawdown": max_drawdown,
                    "Annual Return": mean_ret,
                    "Annual Volatility": volatility
                })
            return pd.DataFrame(results).set_index("Model")
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise

    def plot_comparison(self, comparison_df: pd.DataFrame) -> None:
        """
        Prints formatted table to console (logging).

        Parameters
        ----------
        comparison_df : pd.DataFrame
            The comparison results from compare_models.
        """
        logger.info("Plotting comparison (console output).")
        try:
            print("\n=== Model Version Comparison ===")
            print(comparison_df.to_string(float_format=lambda x: f"{x:.4f}"))
            print("================================\n")
        except Exception as e:
            logger.error(f"Error formatting comparison table: {e}")
            raise
