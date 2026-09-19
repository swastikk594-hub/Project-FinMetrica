import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class FamaMacBeth:
    r"""
    Implements Fama-MacBeth cross-sectional regression framework, augmented with
    proper out-of-sample (OOS) walk-forward testing for incremental R^2.
    """
    
    def __init__(self, horizon_days: int = 21, training_window: int = 252, rolling_step: int = 21):
        self.horizon_days = horizon_days
        self.training_window = training_window
        self.rolling_step = rolling_step

    def run_cross_sectional_regression(self, factor_scores: pd.DataFrame, forward_returns: pd.Series) -> pd.Series:
        r"""
        Single-date OLS cross-sectional regression.
        Clearly labeled as IN-SAMPLE.
        
        Args:
            factor_scores: DataFrame of factor scores for a single date (index = assets).
            forward_returns: Series of forward returns for the same date (index = assets).
            
        Returns:
            pd.Series containing the fitted coefficients.
        """
        if factor_scores.empty or forward_returns.empty:
            return pd.Series(dtype=float)
            
        common_idx = factor_scores.index.intersection(forward_returns.index)
        if len(common_idx) < 3:
            return pd.Series(dtype=float)
            
        X = factor_scores.loc[common_idx]
        y = forward_returns.loc[common_idx]
        
        # Drop NaNs
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        if len(valid_idx) < X.shape[1] + 1:
            return pd.Series(dtype=float)
            
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        X = sm.add_constant(X)
        try:
            model = sm.OLS(y, X).fit()
            return model.params
        except Exception as e:
            logger.warning(f"Cross-sectional regression failed: {e}")
            return pd.Series(dtype=float)

    def walk_forward_fama_macbeth(self, dates: List[pd.Timestamp], factor_scores_by_date: Dict[pd.Timestamp, pd.DataFrame], returns_by_date: Dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
        r"""
        Standard FM: date-by-date cross-sectional regressions.
        Returns coefficient time series.
        This is the INFERENTIAL component (t-stats, Newey-West).
        """
        coefs = []
        valid_dates = []
        for d in dates:
            if d not in factor_scores_by_date or d not in returns_by_date:
                continue
            X = factor_scores_by_date[d]
            y = returns_by_date[d]
            c = self.run_cross_sectional_regression(X, y)
            if not c.empty:
                coefs.append(c)
                valid_dates.append(d)
                
        if not coefs:
            return pd.DataFrame()
            
        res = pd.DataFrame(coefs, index=valid_dates)
        return res

    def walk_forward_oos_prediction(self, dates: List[pd.Timestamp], factor_scores_by_date: Dict[pd.Timestamp, pd.DataFrame], returns_by_date: Dict[pd.Timestamp, pd.Series], target_factor: Optional[str] = None) -> Dict[str, Union[pd.Series, float, int]]:
        r"""
        CRITICAL: Walk-forward OOS prediction loop.
        
        For each test date t:
          1. Collect training data from dates in [t - training_window, t)
          2. Fit baseline model (all factors except target) on training data
          3. Fit full model (all factors) on training data
          4. Generate predictions for test date t using fitted coefficients
          5. Record test-period squared errors
        
        Returns dict with per-date predictions and actual returns,
        or aggregated OOS stats if target_factor is passed.
        """
        # Sort dates to ensure temporal order
        sorted_dates = sorted(dates)
        
        all_y_test = []
        all_y_hat_base = []
        all_y_hat_full = []
        all_y_bar_train = []
        
        factors_available = None
        for d in sorted_dates:
            if d in factor_scores_by_date:
                factors_available = list(factor_scores_by_date[d].columns)
                break
                
        if not factors_available:
            return {}
            
        full_factors = factors_available
        if target_factor is not None:
            base_factors = [f for f in full_factors if f != target_factor]
        else:
            base_factors = full_factors # No baseline
            
        for i, test_date in enumerate(sorted_dates):
            if test_date not in factor_scores_by_date or test_date not in returns_by_date:
                continue
                
            # Define training window
            train_start = test_date - pd.Timedelta(days=self.training_window)
            train_dates = [d for d in sorted_dates if train_start <= d < test_date]
            
            if len(train_dates) < 3: # Need minimum training history
                continue
                
            X_train_list = []
            y_train_list = []
            for td in train_dates:
                if td in factor_scores_by_date and td in returns_by_date:
                    X_td = factor_scores_by_date[td]
                    y_td = returns_by_date[td]
                    common_idx = X_td.index.intersection(y_td.index)
                    if len(common_idx) > 0:
                        df_td = pd.concat([X_td.loc[common_idx], y_td.loc[common_idx].rename('y')], axis=1)
                        df_td = df_td.dropna()
                        if not df_td.empty:
                            X_train_list.append(df_td[full_factors])
                            y_train_list.append(df_td['y'])
                            
            if not X_train_list:
                continue
                
            X_train = pd.concat(X_train_list, axis=0)
            y_train = pd.concat(y_train_list, axis=0)
            
            if len(X_train) < len(full_factors) + 2:
                continue
                
            # Test data
            X_test_raw = factor_scores_by_date[test_date]
            y_test_raw = returns_by_date[test_date]
            common_idx_test = X_test_raw.index.intersection(y_test_raw.index)
            if len(common_idx_test) == 0:
                continue
                
            df_test = pd.concat([X_test_raw.loc[common_idx_test], y_test_raw.loc[common_idx_test].rename('y')], axis=1).dropna()
            if df_test.empty:
                continue
                
            X_test = df_test[full_factors]
            y_test = df_test['y']
            
            y_bar_train = y_train.mean()
            
            # Fit full model
            X_train_full_sm = sm.add_constant(X_train)
            try:
                model_full = sm.OLS(y_train, X_train_full_sm).fit()
            except Exception:
                continue
                
            X_test_full_sm = sm.add_constant(X_test, has_constant='add')
            
            # Make sure test data columns match train data columns in add_constant
            # statsmodels add_constant adds 'const' column.
            # Just in case, manual prediction:
            const_col = 'const' if 'const' in model_full.params else 'Intercept'
            
            try:
                y_hat_full = model_full.predict(X_test_full_sm)
            except Exception:
                # Manual predict if column mismatch
                y_hat_full = pd.Series(model_full.params.get('const', 0), index=X_test.index)
                for f in full_factors:
                    if f in model_full.params:
                        y_hat_full += model_full.params[f] * X_test[f]
            
            # Fit base model
            if target_factor is not None and len(base_factors) > 0:
                X_train_base_sm = sm.add_constant(X_train[base_factors])
                try:
                    model_base = sm.OLS(y_train, X_train_base_sm).fit()
                    X_test_base_sm = sm.add_constant(X_test[base_factors], has_constant='add')
                    try:
                        y_hat_base = model_base.predict(X_test_base_sm)
                    except Exception:
                        y_hat_base = pd.Series(model_base.params.get('const', 0), index=X_test.index)
                        for f in base_factors:
                            if f in model_base.params:
                                y_hat_base += model_base.params[f] * X_test[f]
                except Exception:
                    y_hat_base = pd.Series(y_bar_train, index=y_test.index)
            else:
                y_hat_base = pd.Series(y_bar_train, index=y_test.index)

            all_y_test.extend(y_test.values)
            all_y_hat_base.extend(y_hat_base.values)
            all_y_hat_full.extend(y_hat_full.values)
            all_y_bar_train.extend([y_bar_train] * len(y_test))
            
        return {
            'y_test': np.array(all_y_test),
            'y_hat_base': np.array(all_y_hat_base),
            'y_hat_full': np.array(all_y_hat_full),
            'y_bar_train': np.array(all_y_bar_train)
        }

    def compute_oos_incremental_r2(self, dates: List[pd.Timestamp], factor_scores_by_date: Dict[pd.Timestamp, pd.DataFrame], returns_by_date: Dict[pd.Timestamp, pd.Series], target_factor: str) -> Dict[str, Union[float, int, np.ndarray]]:
        r"""
        PRIMARY METRIC: Out-of-sample incremental R^2.
        
        Mathematical definition:
          SSE_full = \sum_{test} (Y_i - \hat{Y}_{full,i})^2
          SSE_base = \sum_{test} (Y_i - \hat{Y}_{base,i})^2  
          SST = \sum_{test} (Y_i - \bar{Y}_{train})^2
          
          R^2_{oos,full} = 1 - SSE_full / SST
          R^2_{oos,base} = 1 - SSE_base / SST
          \Delta R^2 = R^2_{oos,full} - R^2_{oos,base}
                     = (SSE_base - SSE_full) / SST
        
        CRITICAL: Y_bar_train uses ONLY training data mean.
        CRITICAL: Predictions use ONLY training-fitted coefficients.
        CRITICAL: Baseline and full models use IDENTICAL training/test observations.
        
        Returns dict with: r2_oos_full, r2_oos_base, delta_r2, 
                          sse_full, sse_base, sst, n_test_obs,
                          per_obs_loss_diff
        """
        res = self.walk_forward_oos_prediction(dates, factor_scores_by_date, returns_by_date, target_factor)
        
        if not res or len(res.get('y_test', [])) == 0:
            return {
                'r2_oos_full': np.nan, 'r2_oos_base': np.nan, 'delta_r2': np.nan,
                'sse_full': np.nan, 'sse_base': np.nan, 'sst': np.nan,
                'n_test_obs': 0, 'per_obs_loss_diff': np.array([])
            }
            
        y_test = res['y_test']
        y_hat_base = res['y_hat_base']
        y_hat_full = res['y_hat_full']
        y_bar_train = res['y_bar_train']
        
        se_full = (y_test - y_hat_full)**2
        se_base = (y_test - y_hat_base)**2
        st = (y_test - y_bar_train)**2
        
        sse_full = np.sum(se_full)
        sse_base = np.sum(se_base)
        sst = np.sum(st)
        
        r2_oos_full = 1.0 - (sse_full / sst) if sst > 1e-12 else np.nan
        r2_oos_base = 1.0 - (sse_base / sst) if sst > 1e-12 else np.nan
        delta_r2 = r2_oos_full - r2_oos_base if not np.isnan(r2_oos_full) and not np.isnan(r2_oos_base) else np.nan
        
        per_obs_loss_diff = se_base - se_full
        
        return {
            'r2_oos_full': r2_oos_full,
            'r2_oos_base': r2_oos_base,
            'delta_r2': delta_r2,
            'sse_full': sse_full,
            'sse_base': sse_base,
            'sst': sst,
            'n_test_obs': len(y_test),
            'per_obs_loss_diff': per_obs_loss_diff
        }

    def compute_all_oos_incremental(self, dates: List[pd.Timestamp], factor_scores_by_date: Dict[pd.Timestamp, pd.DataFrame], returns_by_date: Dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
        r"""
        Compute OOS incremental R^2 for ALL factors.
        """
        factors = []
        for d in dates:
            if d in factor_scores_by_date:
                factors = list(factor_scores_by_date[d].columns)
                break
                
        results = []
        for f in factors:
            res = self.compute_oos_incremental_r2(dates, factor_scores_by_date, returns_by_date, f)
            results.append({
                'factor': f,
                'delta_r2': res['delta_r2'],
                'r2_oos_full': res['r2_oos_full'],
                'r2_oos_base': res['r2_oos_base'],
                'n_test_obs': res['n_test_obs']
            })
            
        return pd.DataFrame(results).set_index('factor')

    def compute_newey_west_se(self, coefficients_series: pd.Series, max_lags: Optional[int] = None) -> float:
        r"""
        Newey-West HAC standard errors. Same as before.
        Default max_lags: 4 * (T/100)^(2/9)
        """
        if coefficients_series.empty or len(coefficients_series.dropna()) < 3:
            return np.nan
            
        T = len(coefficients_series)
        if max_lags is None:
            max_lags = int(4 * (T / 100.0) ** (2/9))
            
        # Fit regression on constant to get HAC SE of the mean
        y = coefficients_series.dropna()
        X = np.ones(len(y))
        
        try:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
            return model.bse[0]
        except Exception as e:
            logger.warning(f"Newey-West calculation failed: {e}")
            return np.nan
