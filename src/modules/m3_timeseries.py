import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.modules.base import ModuleResult

logger = logging.getLogger(__name__)

"""
Module 3: Price / Time-Series Model
====================================
PURPOSE:
  Characterise the statistical properties of the price time series.
  This module does NOT predict future prices — it describes what kind
  of process the price series appears to follow and measures momentum.

MATHEMATICAL FRAMEWORK:

  1. MOMENTUM FACTORS:
     Momentum measures the tendency of recent returns to persist.
     Standard academic factor (Jegadeesh & Titman 1993):
     MOM = r(t-252, t-21)  — 12-month return, skipping last month
     (The 1-month skip removes short-term reversal contamination)
     
     Multi-period momentum: compute for 1M, 3M, 6M, 12M windows
     Primary momentum signal = 12-1 month
     
  2. HURST EXPONENT:
     H ∈ (0, 1) measures long-range dependence:
     H < 0.5 → mean-reverting process (anti-persistent)
     H = 0.5 → random walk (Brownian motion)
     H > 0.5 → trending process (persistent)
     
     Estimated via Rescaled Range (R/S) analysis:
     RS_n = (max(Xₜ) - min(Xₜ)) / std(rₜ)
     H ≈ slope of log(RS) vs log(n)
     
     DECISION NOTE: R/S vs DFA (Detrended Fluctuation Analysis):
     R/S is simpler and more established in finance; DFA is more robust
     to non-stationarity. We use R/S but document the limitation.
     
  3. AUTOCORRELATION ANALYSIS:
     ρ_k = Cov(rₜ, rₜ₋ₖ) / Var(rₜ)
     Ljung-Box test: H₀ = no serial correlation
     Q = n(n+2)·Σₖ₌₁ᵐ [ρₖ² / (n-k)] ~ χ²(m)
     
  4. MOVING AVERAGE REGIME INDICATOR:
     Not used as a trading signal, but as a technical regime indicator:
     Price > MA200 → uptrend regime
     Price < MA200 → downtrend regime
     (This is used to condition Module 6 regime analysis)

METHOD COMPARISON — 12-month vs 6-month momentum:
  12-month: Classic academic factor; best documented cross-sectionally
  6-month: Faster; may capture more recent sentiment
  SELECTED: 12-1 month primary, 6-1 month secondary, both reported

ASSUMPTIONS:
  - Log returns are approximately stationary (tested by ADF)
  - Momentum persistence is a real empirical phenomenon
  
LIMITATIONS:
  - Momentum crashes in sudden reversals (e.g. March 2020)
  - Hurst exponent estimates have high variance for short series
  - Past momentum provides no guarantee of future continuation
"""

class TimeSeriesModule:
    """
    Time Series Module to characterise the statistical properties
    of the price time series.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    def compute_momentum(self, returns: pd.Series, periods: List[int] = [21, 63, 126, 252], skip: int = 21) -> Dict[int, Dict[str, float]]:
        """
        Compute multi-period momentum, skipping the most recent period to avoid short-term reversal.

        Parameters
        ----------
        returns : pd.Series
            Daily log returns.
        periods : List[int], optional
            Lookback periods in days, by default [21, 63, 126, 252].
        skip : int, optional
            Number of recent days to skip, by default 21 (1 month).

        Returns
        -------
        Dict[int, Dict[str, float]]
            Dictionary mapping period to momentum metrics.
        """
        momentum_metrics = {}
        if len(returns) < skip:
            return momentum_metrics
            
        for period in periods:
            if len(returns) < period:
                continue
                
            # Historical cumulative returns matching the window length over the entire series to compute percentiles
            # Roll sum of log returns is equivalent to cumulative log returns
            rolling_cum_returns = returns.rolling(window=period - skip).sum().shift(skip)
            
            # Current momentum value
            current_mom = rolling_cum_returns.iloc[-1]
            
            if pd.isna(current_mom):
                continue
            
            # Annualised return for the period
            years = (period - skip) / 252.0
            annualised = current_mom / years if years > 0 else np.nan
            
            # Percentile
            historical_values = rolling_cum_returns.dropna()
            if len(historical_values) > 0:
                percentile = (historical_values < current_mom).mean() * 100
            else:
                percentile = np.nan
                
            momentum_metrics[period] = {
                'return': float(current_mom),
                'annualised': float(annualised),
                'percentile': float(percentile)
            }
            
        return momentum_metrics

    def compute_hurst_exponent(self, prices: pd.Series, min_window: int = 100) -> Dict[str, Any]:
        """
        Compute Hurst exponent using Rescaled Range (R/S) analysis.

        Parameters
        ----------
        prices : pd.Series
            Price series.
        min_window : int, optional
            Minimum window size for calculation, by default 100.

        Returns
        -------
        Dict[str, Any]
            Hurst exponent metrics and interpretation.
        """
        if len(prices) < min_window:
            return {'h_exponent': np.nan, 'interpretation': 'Insufficient data', 'r_squared': np.nan, 'warning': 'Not enough data points'}
            
        lags = range(2, min_window)
        
        # Log returns
        returns = np.log(prices).diff().dropna().values
        
        tau = []
        lagvec = []
        
        # Step through different lag sizes
        for lag in lags:
            if len(returns) < lag:
                break
            
            # Divide returns into non-overlapping chunks
            chunks = np.array_split(returns, len(returns) // lag)
            rs_vals = []
            for chunk in chunks:
                if len(chunk) < lag:
                    continue
                # Mean centered
                mean_adj = chunk - np.mean(chunk)
                # Cumulative sum of mean adjusted
                cum_sum = np.cumsum(mean_adj)
                # Range
                r = np.max(cum_sum) - np.min(cum_sum)
                # Standard deviation
                s = np.std(chunk)
                if s > 0:
                    rs_vals.append(r / s)
            
            if rs_vals:
                tau.append(np.mean(rs_vals))
                lagvec.append(lag)
                
        if not tau or len(tau) < 10:
            return {'h_exponent': np.nan, 'interpretation': 'Insufficient valid lags', 'r_squared': np.nan, 'warning': 'Poor R/S fit'}
            
        # Linear regression on log(RS) vs log(lags)
        log_lags = np.log(lagvec)
        log_tau = np.log(tau)
        
        A = np.vstack([log_lags, np.ones(len(log_lags))]).T
        m, c = np.linalg.lstsq(A, log_tau, rcond=None)[0]
        
        # R-squared
        residuals = log_tau - (m * log_lags + c)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_tau - np.mean(log_tau))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        h_exponent = m
        warning = 'Poor fit (R² < 0.9)' if r_squared < 0.9 else ''
        
        if h_exponent < 0.45:
            interp = 'Mean-reverting'
        elif h_exponent > 0.55:
            interp = 'Trending'
        else:
            interp = 'Random walk'
            
        return {
            'h_exponent': float(h_exponent),
            'interpretation': interp,
            'r_squared': float(r_squared),
            'warning': warning
        }

    def compute_autocorrelation(self, returns: pd.Series, lags: List[int] = [1, 5, 10, 21]) -> Dict[str, Any]:
        """
        Compute Autocorrelation Function (ACF) and perform Ljung-Box test.

        Parameters
        ----------
        returns : pd.Series
            Daily log returns.
        lags : List[int], optional
            Lags to compute ACF for, by default [1, 5, 10, 21].

        Returns
        -------
        Dict[str, Any]
            ACF values, Ljung-Box stats and significance.
        """
        clean_returns = returns.dropna()
        if len(clean_returns) < max(lags) * 2:
            return {'error': 'Insufficient data for ACF lags'}
            
        max_lag = max(lags)
        # Compute ACF
        acf_vals = acf(clean_returns, nlags=max_lag, fft=True)
        
        lag_acf = {lag: float(acf_vals[lag]) for lag in lags if lag < len(acf_vals)}
        
        # Ljung-Box test
        lb_res = acorr_ljungbox(clean_returns, lags=[max_lag], return_df=True)
        lb_stat = float(lb_res.iloc[0]['lb_stat'])
        lb_pval = float(lb_res.iloc[0]['lb_pvalue'])
        
        return {
            'acf_values': lag_acf,
            'lb_statistic': lb_stat,
            'lb_pvalue': lb_pval,
            'significant': bool(lb_pval < 0.05)
        }

    def compute_moving_average_regime(self, prices: pd.Series, short_window: int = 50, long_window: int = 200) -> Dict[str, Any]:
        """
        Compute Moving Average regime indicator.

        Parameters
        ----------
        prices : pd.Series
            Price series.
        short_window : int, optional
            Short MA window, by default 50.
        long_window : int, optional
            Long MA window, by default 200.

        Returns
        -------
        Dict[str, Any]
            Current regime, MA values, and metrics.
        """
        if len(prices) < long_window:
            return {'current_regime': 'Unknown', 'days_in_current_regime': 0}
            
        ma_short = prices.rolling(window=short_window).mean()
        ma_long = prices.rolling(window=long_window).mean()
        
        curr_price = prices.iloc[-1]
        curr_ma_short = ma_short.iloc[-1]
        curr_ma_long = ma_long.iloc[-1]
        
        # Calculate historical regime to find days in current regime
        regimes = (prices > ma_long).astype(int)
        # Find when regime changed
        changes = regimes.diff()
        last_change_idx = changes[changes != 0].last_valid_index()
        
        if last_change_idx:
            # Assuming business days
            days_in_regime = len(prices.loc[last_change_idx:])
        else:
            days_in_regime = len(prices)
            
        current_regime = 'uptrend' if curr_price > curr_ma_long else 'downtrend'
        ratio = curr_price / curr_ma_long if curr_ma_long > 0 else np.nan
        
        return {
            'current_regime': current_regime,
            'ma_short': float(curr_ma_short),
            'ma_long': float(curr_ma_long),
            'days_in_current_regime': int(days_in_regime),
            'price_to_ma200_ratio': float(ratio)
        }

    def compute_volatility_regime(self, returns: pd.Series, window: int = 21) -> Dict[str, Any]:
        """
        Compute Volatility regime.

        Parameters
        ----------
        returns : pd.Series
            Daily log returns.
        window : int, optional
            Rolling volatility window, by default 21.

        Returns
        -------
        Dict[str, Any]
            Volatility metrics and regime.
        """
        if len(returns) < window:
            return {'vol_regime': 'Unknown'}
            
        rolling_vol = returns.rolling(window=window).std()
        current_vol = rolling_vol.iloc[-1]
        
        hist_vol = rolling_vol.dropna()
        if len(hist_vol) > 0:
            percentile = (hist_vol < current_vol).mean() * 100
        else:
            percentile = np.nan
            
        if percentile < 25:
            regime = 'low'
        elif percentile > 75:
            regime = 'high'
        else:
            regime = 'normal'
            
        return {
            'current_vol': float(current_vol),
            'current_vol_annualised': float(current_vol * np.sqrt(252)),
            'vol_regime': regime,
            'vol_percentile': float(percentile)
        }

    def run(self, ticker: str, price_data: pd.DataFrame) -> ModuleResult:
        """
        Execute Time Series module calculations.

        Parameters
        ----------
        ticker : str
            Asset ticker.
        price_data : pd.DataFrame
            OHLCV DataFrame.

        Returns
        -------
        ModuleResult
            Results of the time series analysis.
        """
        logger.info(f"Running Time Series Module for {ticker}")
        
        close_col = 'Adj Close' if 'Adj Close' in price_data.columns else ('Close' if 'Close' in price_data.columns else None)
        if price_data.empty or close_col is None:
            return ModuleResult(
                module_name="TimeSeries",
                ticker=ticker,
                normalised_score=50.0,
                warnings=["Missing close prices"],
            )
            
        prices = price_data[close_col]
        returns = np.log(prices / prices.shift(1)).dropna()
        
        momentum = self.compute_momentum(returns)
        hurst = self.compute_hurst_exponent(prices)
        acf_res = self.compute_autocorrelation(returns)
        ma_regime = self.compute_moving_average_regime(prices)
        vol_regime = self.compute_volatility_regime(returns)
        
        metrics = {
            'momentum': momentum,
            'hurst': hurst,
            'acf': acf_res,
            'ma_regime': ma_regime,
            'vol_regime': vol_regime
        }
        
        # Scoring based on time-series properties
        score = 50.0  # Base score
        
        # Adjust for momentum (12-1 month = 252 days)
        if 252 in momentum:
            perc = momentum[252].get('percentile', 50)
            # High momentum -> higher score (trend following perspective)
            score += (perc - 50) / 2
            
        # Adjust for regime
        if ma_regime.get('current_regime') == 'uptrend':
            score += 10
        elif ma_regime.get('current_regime') == 'downtrend':
            score -= 10
            
        score = max(0.0, min(100.0, score))
        
        return ModuleResult(
            module_name="TimeSeries",
            ticker=ticker,
            normalised_score=score,
            components=metrics,
            metadata={"description": "Statistical time series properties characterisation"},
        )

    def explain(self, result: ModuleResult) -> str:
        """
        Provide human-readable explanation of the Time Series results.

        Parameters
        ----------
        result : ModuleResult
            Result from run().

        Returns
        -------
        str
            Explanation text.
        """
        lines = [f"--- Time Series Analysis for {result.ticker} ---",
                 f"Score: {result.score:.2f}/100", ""]
        
        metrics = result.metrics
        
        if 'error' in result.details:
            lines.append(f"Error: {result.details['error']}")
            return "\n".join(lines)
            
        mom = metrics.get('momentum', {})
        if 252 in mom:
            m = mom[252]
            lines.append(f"12M Momentum (skip 1M): {m['return']*100:.2f}% (Annualised: {m['annualised']*100:.2f}%)")
            lines.append(f"Momentum Percentile: {m['percentile']:.1f}%")
        
        ma = metrics.get('ma_regime', {})
        reg = ma.get('current_regime', 'Unknown')
        lines.append(f"MA Regime: {reg.capitalize()} (Days in regime: {ma.get('days_in_current_regime', 0)})")
        
        hurst = metrics.get('hurst', {})
        if 'h_exponent' in hurst and not pd.isna(hurst['h_exponent']):
            lines.append(f"Hurst Exponent: {hurst['h_exponent']:.3f} - {hurst.get('interpretation', '')}")
            if hurst.get('warning'):
                lines.append(f"Warning: {hurst['warning']}")
                
        vol = metrics.get('vol_regime', {})
        lines.append(f"Volatility Regime: {vol.get('vol_regime', 'Unknown').capitalize()} "
                     f"(Annualised Vol: {vol.get('current_vol_annualised', 0)*100:.2f}%)")
                     
        acf_res = metrics.get('acf', {})
        if 'significant' in acf_res:
            sig = "Yes" if acf_res['significant'] else "No"
            lines.append(f"Significant Autocorrelation (Ljung-Box): {sig}")
            
        return "\n".join(lines)
