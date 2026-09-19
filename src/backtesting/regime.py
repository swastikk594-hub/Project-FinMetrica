"""
Regime Analysis Module
=======================
Tests whether model performance is consistent across market regimes.

REGIME DEFINITIONS:
  BULL MARKET: Market return > +20% from trough
  BEAR MARKET: Market return < -20% from peak  
  HIGH VOLATILITY: VIX > 25 (or market vol > 25th percentile)
  LOW VOLATILITY: VIX < 15 (or market vol < 75th percentile)
  RECESSION: NBER recession dates (or term spread < 0)

HYPOTHESIS: A model that performs well only in bull markets
            is not robust. We want consistent performance.

METHODOLOGY:
  1. Classify each historical date into a regime
  2. Split portfolio returns into regime-specific periods
  3. Compute performance metrics for each regime
  4. Compare model vs benchmark within each regime
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RegimeAnalysis:
    """
    Class to classify market regimes and analyze performance across them.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RegimeAnalysis with a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration parameters.
        """
        self.config = config
        np.random.seed(config.get("random_seed", 42))

    def classify_bull_bear(self, benchmark_returns: pd.Series, threshold: float = 0.20) -> pd.Series:
        """
        Classifies dates into bull, bear, or neutral markets.

        Parameters
        ----------
        benchmark_returns : pd.Series
            Daily returns of the benchmark.
        threshold : float
            Threshold for bull/bear classification (default 0.20).

        Returns
        -------
        pd.Series
            Series of 'bull', 'bear', 'neutral' strings.
        """
        logger.info("Classifying bull/bear regimes.")
        try:
            cum_rets = (1 + benchmark_returns).cumprod()
            peak = cum_rets.expanding(min_periods=1).max()
            trough = cum_rets.expanding(min_periods=1).min()
            
            drawdown = (cum_rets - peak) / peak
            runup = (cum_rets - trough) / trough
            
            regimes = pd.Series('neutral', index=benchmark_returns.index)
            regimes[drawdown < -threshold] = 'bear'
            regimes[runup > threshold] = 'bull'
            return regimes
        except Exception as e:
            logger.error(f"Error classifying bull/bear regimes: {e}")
            raise

    def classify_volatility_regime(self, returns: pd.Series, low_pct: float = 25, high_pct: float = 75) -> pd.Series:
        """
        Classifies volatility into low, normal, or high.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the benchmark.
        low_pct : float
            Percentile threshold for low volatility.
        high_pct : float
            Percentile threshold for high volatility.

        Returns
        -------
        pd.Series
            Series of 'low_vol', 'normal_vol', 'high_vol' strings.
        """
        logger.info("Classifying volatility regimes.")
        try:
            vol_window = self.config.get("vol_window", 63)
            rolling_vol = returns.rolling(window=vol_window, min_periods=vol_window).std() * np.sqrt(252)
            
            low_thresh = np.nanpercentile(rolling_vol, low_pct)
            high_thresh = np.nanpercentile(rolling_vol, high_pct)
            
            regimes = pd.Series('normal_vol', index=returns.index)
            regimes[rolling_vol < low_thresh] = 'low_vol'
            regimes[rolling_vol > high_thresh] = 'high_vol'
            return regimes
        except Exception as e:
            logger.error(f"Error classifying volatility regimes: {e}")
            raise

    def classify_macro_regime(self, term_spread: pd.Series) -> pd.Series:
        """
        Classifies macro regime based on term spread.

        Parameters
        ----------
        term_spread : pd.Series
            Term spread data (e.g., 10Y - 2Y yield).

        Returns
        -------
        pd.Series
            Series of 'expansion', 'contraction', 'neutral'.
        """
        logger.info("Classifying macro regimes.")
        try:
            regimes = pd.Series('neutral', index=term_spread.index)
            regimes[term_spread > 0.0] = 'expansion'
            regimes[term_spread < 0.0] = 'contraction'
            return regimes
        except Exception as e:
            logger.error(f"Error classifying macro regimes: {e}")
            raise

    def regime_performance(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, regime_series: pd.Series) -> pd.DataFrame:
        """
        Computes performance metrics for each regime.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily returns of the portfolio.
        benchmark_returns : pd.Series
            Daily returns of the benchmark.
        regime_series : pd.Series
            Regime classifications aligned with returns.

        Returns
        -------
        pd.DataFrame
            DataFrame with regime × metrics.
        """
        logger.info("Computing regime-specific performance.")
        try:
            unique_regimes = regime_series.dropna().unique()
            results = []
            
            for regime in unique_regimes:
                mask = (regime_series == regime)
                port_rets = portfolio_returns[mask]
                bench_rets = benchmark_returns[mask]
                
                if len(port_rets) < 2:
                    continue
                    
                mean_ret = port_rets.mean() * 252
                volatility = port_rets.std() * np.sqrt(252)
                sharpe = mean_ret / volatility if volatility != 0 else 0.0
                
                cum_rets = (1 + port_rets).cumprod()
                peak = cum_rets.expanding(min_periods=1).max()
                drawdown = (cum_rets - peak) / peak
                max_drawdown = drawdown.min()
                
                bench_mean = bench_rets.mean() * 252
                bench_vol = bench_rets.std() * np.sqrt(252)
                bench_sharpe = bench_mean / bench_vol if bench_vol != 0 else 0.0
                
                results.append({
                    "Regime": regime,
                    "Port Return": mean_ret,
                    "Port Vol": volatility,
                    "Port Sharpe": sharpe,
                    "Port MaxDD": max_drawdown,
                    "Bench Return": bench_mean,
                    "Bench Sharpe": bench_sharpe
                })
                
            return pd.DataFrame(results).set_index("Regime")
        except Exception as e:
            logger.error(f"Error computing regime performance: {e}")
            raise

    def run_full_regime_analysis(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, macro_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Runs all classifications and computes performance in each.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Portfolio returns.
        benchmark_returns : pd.Series
            Benchmark returns.
        macro_data : pd.DataFrame
            Macro data containing at least a 'term_spread' column.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing regime analysis DataFrames.
        """
        logger.info("Running full regime analysis.")
        try:
            bull_bear_series = self.classify_bull_bear(benchmark_returns)
            vol_series = self.classify_volatility_regime(benchmark_returns)
            term_spread = macro_data.get('term_spread', pd.Series(0, index=benchmark_returns.index))
            macro_series = self.classify_macro_regime(term_spread)
            
            return {
                "Bull_Bear": self.regime_performance(portfolio_returns, benchmark_returns, bull_bear_series),
                "Volatility": self.regime_performance(portfolio_returns, benchmark_returns, vol_series),
                "Macro": self.regime_performance(portfolio_returns, benchmark_returns, macro_series)
            }
        except Exception as e:
            logger.error(f"Error in full regime analysis: {e}")
            raise
