"""
Module 6: Macroeconomic / Systematic Factor Model
===================================================
PURPOSE:
  Measure the asset's sensitivity to macroeconomic risk factors
  that are external to the company. Unlike Module 4 (which uses
  statistical factors derived from equity returns), this module
  uses ECONOMIC variables as the factors.

MATHEMATICAL FRAMEWORK:
  Macro factor regression:
  rᵢ,ₜ = α + β₁·ΔTermSpread_t + β₂·ΔCreditSpread_t + 
          β₃·ΔCPI_t + β₄·ΔUnemployment_t + εᵢ,ₜ
  
  where all macro variables are lagged by 1 month to prevent
  look-ahead bias (macro data released with delay).
  
  Term Spread = 10Y Treasury Yield - 2Y Treasury Yield
    Positive → normal yield curve
    Negative → inverted yield curve (recession signal)
    
  Credit Spread = Baa Corporate - 10Y Treasury
    Widening → increasing default risk perception
    
  Business Cycle Regime:
    Uses term spread to classify: Expansion / Contraction
    Term Spread > 0 → Expansion
    Term Spread < 0 → Contraction signal (but not certain)
    
METHOD COMPARISON — Number of Macro Variables:
  Parsimony principle: More variables → overfitting risk
  We test models with 2, 3, 4 variables and compare R²
  SELECTED: 4 variables (term spread, credit spread, CPI, unemployment)
            These are the least correlated major macro variables

ASSUMPTIONS:
  - Linear relationship between macro variables and returns
  - Macro variables have been lagged appropriately
  - Monthly frequency macro data aligns with monthly asset returns
  
LIMITATIONS:
  - Macro data released with significant delay (GDP: 30 days)
  - Relationship between macro and returns is not stable
  - Low R² is typical; macro explains little of individual stock returns
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import logging
from typing import Dict, Any, Optional
from src.modules.base import ModuleResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroModule:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def prepare_macro_features(self, macro_data: pd.DataFrame, lag_months: int = 1) -> pd.DataFrame:
        """
        Computes YoY changes, momentum, and regime indicators from macro data.
        Applies lag to prevent look-ahead.

        The fetcher returns FRED series under their series IDs.  This method
        normalises common series IDs to the canonical names used internally
        ('term_spread', 'credit_spread', 'cpi', 'unemployment', 'fed_funds').
        """
        # --- Normalise column names from FRED series IDs ---
        # Map any known FRED aliases to canonical internal names
        FRED_ALIAS = {
            'T10Y2Y':         'term_spread',
            'BAMLH0A0HYM2':   'credit_spread',
            'CPIAUCSL':       'cpi',
            'CPILFESL':       'cpi',  # core CPI alternative
            'UNRATE':         'unemployment',
            'FEDFUNDS':       'fed_funds',
            'GS10':           'treasury_10y',
            'GS2':            'treasury_2y',
            # Also accept human-readable column names passed in directly
            'Fed Funds Rate': 'fed_funds',
            'Term Spread':    'term_spread',
            'HY Spread':      'credit_spread',
        }
        renamed = {}
        for col in macro_data.columns:
            renamed[col] = FRED_ALIAS.get(col, col)
        macro_data = macro_data.rename(columns=renamed)

        features = pd.DataFrame(index=macro_data.index)
        
        # For rate/spread series: first-difference captures the change in level
        for col in ['term_spread', 'credit_spread', 'fed_funds']:
            if col in macro_data.columns:
                features[f'delta_{col}'] = macro_data[col].diff()
                
        for col in ['cpi', 'unemployment']:
            if col in macro_data.columns:
                # YoY percentage change or diff
                features[f'yoy_{col}'] = macro_data[col].pct_change(12)

        # If term_spread is missing but 10Y and 2Y are available, compute it
        if 'term_spread' not in macro_data.columns:
            if 'treasury_10y' in macro_data.columns and 'treasury_2y' in macro_data.columns:
                ts = macro_data['treasury_10y'] - macro_data['treasury_2y']
                features['delta_term_spread'] = ts.diff()
                macro_data['term_spread'] = ts  # store for regime classification

        # Lag features to avoid look-ahead bias
        features = features.shift(lag_months).dropna()
        # Store the processed macro_data for regime classification
        self._last_macro_data = macro_data
        return features

    def classify_regime(self, term_spread: pd.Series) -> pd.Series:
        """
        Returns 'expansion' or 'contraction' based on term spread.
        """
        # positive term spread -> expansion, negative -> contraction
        regimes = pd.Series('expansion', index=term_spread.index)
        regimes[term_spread < 0] = 'contraction'
        return regimes

    def estimate_macro_betas(self, monthly_returns: pd.Series, macro_features: pd.DataFrame, window: int = 36) -> Dict[str, Any]:
        """
        OLS regression of monthly returns on macro features.
        """
        aligned = pd.concat([monthly_returns, macro_features], axis=1).dropna()
        if aligned.empty or len(aligned) < 10:
            return {'betas': {}, 'r_squared': 0.0, 't_stats': {}, 'p_values': {}}
            
        y = aligned.iloc[:, 0]
        X = aligned.iloc[:, 1:]
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit()
        
        # Rolling regression
        rolling_betas = {}
        if len(aligned) >= window:
            for col in X.columns:
                rolling_betas[col] = []
            
            # Simple manual rolling regression
            for i in range(window, len(aligned) + 1):
                y_win = y.iloc[i-window:i]
                X_win = X.iloc[i-window:i]
                mod = sm.OLS(y_win, X_win).fit()
                for col in X.columns:
                    rolling_betas[col].append(mod.params[col])
                    
            rolling_df = pd.DataFrame(rolling_betas, index=aligned.index[window-1:])
        else:
            rolling_df = None

        return {
            'betas': model.params.to_dict(),
            't_stats': model.tvalues.to_dict(),
            'p_values': model.pvalues.to_dict(),
            'r_squared': model.rsquared,
            'residuals': model.resid,
            'rolling_betas': rolling_df
        }

    def compute_macro_score(self, macro_features: pd.DataFrame, betas: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute predicted macro contribution to return.
        """
        if macro_features.empty or not betas:
            return {'predicted_return': 0.0, 'tailwind': 0}
            
        latest_features = macro_features.iloc[-1]
        
        pred_return = betas.get('const', 0.0)
        for col, val in latest_features.items():
            pred_return += val * betas.get(col, 0.0)
            
        sign = 1 if pred_return > 0 else -1
        
        return {
            'predicted_macro_return': float(pred_return),
            'macro_tailwind': sign,
            'confidence': "Low (typical for macro models)"
        }

    def stress_test(self, macro_features: pd.DataFrame, betas: Dict[str, float], scenarios: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Test impact of macro shocks.
        """
        if scenarios is None:
            scenarios = {
                '2008_Crisis': {
                    'delta_term_spread': -0.02,
                    'delta_credit_spread': 0.04,
                    'yoy_cpi': -0.01,
                    'yoy_unemployment': 0.05
                },
                '2020_Covid': {
                    'delta_term_spread': 0.01,
                    'delta_credit_spread': 0.03,
                    'yoy_cpi': -0.02,
                    'yoy_unemployment': 0.10
                }
            }
            
        impacts = {}
        for name, shocks in scenarios.items():
            impact = betas.get('const', 0.0)
            for col, val in shocks.items():
                impact += val * betas.get(col, 0.0)
            impacts[name] = float(impact)
            
        return {'scenario_impacts': impacts}

    def run(self, ticker: str, price_data: pd.DataFrame, macro_data: pd.DataFrame) -> ModuleResult:
        """
        Execute full macro factor analysis.
        """
        logger.info(f"Running Macro Module for {ticker}")
        
        # Monthly returns
        close_col = 'Adj Close' if 'Adj Close' in price_data.columns else 'Close'
        monthly_prices = price_data[close_col].resample('ME').last()
        monthly_returns = monthly_prices.pct_change().dropna()

        if macro_data is None or macro_data.empty:
            return ModuleResult(
                ticker=ticker,
                module_name='Macro',
                normalised_score=50.0,
                warnings=['No macro data available — returning neutral score.'],
            )
        
        features = self.prepare_macro_features(macro_data)
        # _last_macro_data has FRED IDs replaced with canonical names
        macro_renamed = getattr(self, '_last_macro_data', macro_data)
        
        term_spread = macro_renamed.get('term_spread', pd.Series(0, index=macro_renamed.index)) \
            if hasattr(macro_renamed, 'get') else \
            (macro_renamed['term_spread'] if 'term_spread' in macro_renamed.columns else pd.Series(0, index=macro_renamed.index))
        regimes = self.classify_regime(term_spread)
        
        regression_results = self.estimate_macro_betas(monthly_returns, features)
        betas = regression_results.get('betas', {})
        
        macro_score_info = self.compute_macro_score(features, betas)
        stress_info = self.stress_test(features, betas)
        
        # Calculate a score 0-100 based on macro tailwind
        score = 50.0
        predicted_return = macro_score_info.get('predicted_macro_return', 0.0)
        if predicted_return > 0.02:
            score = 80.0
        elif predicted_return < -0.02:
            score = 20.0
            
        return ModuleResult(
            ticker=ticker,
            module_name='Macro',
            normalised_score=score,
            components={'regime': regimes.iloc[-1] if not regimes.empty else 'unknown',
                        'betas': betas, 'r_squared': regression_results.get('r_squared', 0.0),
                        'prediction': macro_score_info, 'stress_tests': stress_info},
        )

    def explain(self, result: ModuleResult) -> str:
        """
        Return educational explanation of the macro results.
        """
        score = result.score
        r2 = result.details.get('r_squared', 0.0)
        tailwind = result.details['prediction'].get('macro_tailwind', 0)
        
        explanation = f"Macro Module Analysis:\n"
        explanation += f"Macro Support Score: {score:.1f}/100\n\n"
        explanation += f"Model R²: {r2*100:.1f}% (Percentage of variance explained by macro factors)\n"
        
        if tailwind > 0:
            explanation += "Current macroeconomic conditions are providing a TAILWIND (positive expected contribution) for this asset.\n"
        else:
            explanation += "Current macroeconomic conditions are providing a HEADWIND (negative expected contribution) for this asset.\n"
            
        return explanation
