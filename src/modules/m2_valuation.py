"""
Module 2: Valuation Model
=========================
PURPOSE:
  Measure whether the market price is reasonable relative to
  the company's fundamental intrinsic value.

MATHEMATICAL FRAMEWORK:
  1. PROBABILISTIC DISCOUNTED CASH FLOW (DCF):
     V = Σ_{t=1}^n [FCF_t / (1+WACC)^t] + [TV / (1+WACC)^n]
     TV = FCF_n * (1+g) / (WACC - g)  s.t. WACC > g

     Tri-Scenario Probabilistic Valuation:
       V_prob = 0.25 * V_bear + 0.50 * V_base + 0.25 * V_bull
       MOS = (V_prob - Price) / V_prob

  2. RELATIVE VALUATION:
     Metrics: P/E, P/B, EV/EBITDA, P/S historical percentiles and z-scores.

  3. FED MODEL:
     Earnings Yield (1 / PE) - 10Y Benchmark Yield.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from src.modules.base import ModuleResult

logger = logging.getLogger(__name__)


class ValuationModule:
    """
    Evaluates intrinsic value through multi-scenario DCF and historical relative valuation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.random_seed = self.config.get('random_seed', 42)

    def _get_metric(self, fin_data: Dict[str, pd.DataFrame], statement: str, item: str, period: int = 0) -> float:
        """Helper to get a financial metric."""
        try:
            df = fin_data.get(statement)
            if df is None or df.empty or period >= df.shape[1]:
                return np.nan
            if item in df.index:
                return float(df.loc[item].iloc[period])
            matches = [idx for idx in df.index if item.lower() in str(idx).lower()]
            if matches:
                return float(df.loc[matches[0]].iloc[period])
            return np.nan
        except Exception:
            return np.nan

    def compute_wacc(self, fin_data: Dict[str, pd.DataFrame], beta: float = 1.0, risk_free_rate: float = 0.04, erp: float = 0.055) -> Dict[str, float]:
        """
        Computes Weighted Average Cost of Capital (WACC).
        WACC = (E/V)·Ke + (D/V)·Kd·(1-t)
        """
        ke = risk_free_rate + max(beta, 0.2) * erp
        debt = self._get_metric(fin_data, 'balance_sheet', 'Long Term Debt', 0)
        std = self._get_metric(fin_data, 'balance_sheet', 'Current Debt', 0)
        if np.isnan(std):
            std = self._get_metric(fin_data, 'balance_sheet', 'Short Term Debt', 0)

        total_debt = (0.0 if np.isnan(debt) else debt) + (0.0 if np.isnan(std) else std)
        if total_debt == 0:
            total_debt = self._get_metric(fin_data, 'balance_sheet', 'Total Debt', 0)

        equity = self._get_metric(fin_data, 'balance_sheet', 'Stockholders Equity', 0)
        tax_expense = self._get_metric(fin_data, 'income_stmt', 'Tax Provision', 0)
        ebt = self._get_metric(fin_data, 'income_stmt', 'Pretax Income', 0)
        tax_rate = max(0.0, min(tax_expense / ebt, 0.40)) if not (np.isnan(tax_expense) or np.isnan(ebt) or ebt <= 0) else 0.21

        interest_expense = abs(self._get_metric(fin_data, 'income_stmt', 'Interest Expense', 0))
        kd = max(0.02, min(interest_expense / total_debt, 0.15)) if (total_debt and not np.isnan(total_debt) and total_debt > 0) else 0.05

        if np.isnan(total_debt) or total_debt <= 0 or np.isnan(equity) or equity <= 0:
            wacc = ke
            w_e, w_d = 1.0, 0.0
        else:
            total_val = equity + total_debt
            w_e = equity / total_val
            w_d = total_debt / total_val
            wacc = (w_e * ke) + (w_d * kd * (1.0 - tax_rate))

        return {
            'wacc': float(max(wacc, 0.05)),
            'ke': float(ke),
            'kd': float(kd),
            'weight_equity': float(w_e),
            'weight_debt': float(w_d),
            'tax_rate': float(tax_rate)
        }

    def compute_probabilistic_dcf(
        self,
        fin_data: Dict[str, pd.DataFrame],
        current_price: float,
        wacc: float,
        shares_outstanding: float
    ) -> Dict[str, Any]:
        """
        Computes a 3-scenario Probabilistic DCF (Bear, Base, Bull).
        Validates WACC > g and computes Margin of Safety.
        """
        fcf_0 = self._get_metric(fin_data, 'cash_flow', 'Free Cash Flow', 0)
        if np.isnan(fcf_0) or fcf_0 <= 0 or shares_outstanding <= 0 or np.isnan(shares_outstanding):
            return {
                'intrinsic_value': np.nan,
                'margin_of_safety': np.nan,
                'scenarios': {},
                'status': 'negative_or_missing_fcf'
            }

        # Historical growth rate proxy
        fcf_1 = self._get_metric(fin_data, 'cash_flow', 'Free Cash Flow', 1)
        hist_growth = (fcf_0 - fcf_1) / abs(fcf_1) if not (np.isnan(fcf_1) or fcf_1 == 0) else 0.05
        base_g_proj = max(min(hist_growth, 0.20), -0.05)

        scenarios = {
            'bear': {'weight': 0.25, 'g_proj': max(base_g_proj - 0.05, -0.05), 'g_term': 0.015, 'wacc_adj': wacc + 0.01},
            'base': {'weight': 0.50, 'g_proj': base_g_proj, 'g_term': 0.025, 'wacc_adj': wacc},
            'bull': {'weight': 0.25, 'g_proj': min(base_g_proj + 0.05, 0.25), 'g_term': 0.035, 'wacc_adj': max(wacc - 0.01, 0.05)}
        }

        scenario_values = {}
        for name, params in scenarios.items():
            g_p = params['g_proj']
            g_t = params['g_term']
            w_adj = max(params['wacc_adj'], g_t + 0.015) # Enforce WACC > g constraint strictly

            # 5-Year Projection
            pv_fcf = 0.0
            fcf_t = fcf_0
            for t in range(1, 6):
                fcf_t *= (1.0 + g_p)
                pv_fcf += fcf_t / ((1.0 + w_adj) ** t)

            # Terminal Value
            tv = (fcf_t * (1.0 + g_t)) / (w_adj - g_t)
            pv_tv = tv / ((1.0 + w_adj) ** 5)
            enterprise_val = pv_fcf + pv_tv

            cash = self._get_metric(fin_data, 'balance_sheet', 'Cash And Cash Equivalents', 0)
            cash = 0.0 if np.isnan(cash) else cash
            debt = self._get_metric(fin_data, 'balance_sheet', 'Total Debt', 0)
            debt = 0.0 if np.isnan(debt) else debt

            equity_val = enterprise_val + cash - debt
            val_per_share = max(equity_val / shares_outstanding, 0.0)
            scenario_values[name] = float(val_per_share)

        prob_intrinsic_val = sum(scenario_values[k] * scenarios[k]['weight'] for k in scenarios)
        mos = (prob_intrinsic_val - current_price) / prob_intrinsic_val if prob_intrinsic_val > 0 else np.nan

        return {
            'intrinsic_value': float(prob_intrinsic_val),
            'margin_of_safety': float(mos),
            'scenarios': scenario_values,
            'status': 'success'
        }

    def compute_relative_valuation(self, fin_data: Dict[str, pd.DataFrame], current_price: float, shares_outstanding: float) -> Dict[str, Any]:
        """Computes EV/EBITDA, P/E, P/B, P/S relative valuation metrics."""
        ni = self._get_metric(fin_data, 'income_stmt', 'Net Income', 0)
        ebitda = self._get_metric(fin_data, 'income_stmt', 'EBITDA', 0)
        equity = self._get_metric(fin_data, 'balance_sheet', 'Stockholders Equity', 0)
        sales = self._get_metric(fin_data, 'income_stmt', 'Total Revenue', 0)
        cash = self._get_metric(fin_data, 'balance_sheet', 'Cash And Cash Equivalents', 0)
        debt = self._get_metric(fin_data, 'balance_sheet', 'Total Debt', 0)

        mcap = current_price * shares_outstanding if shares_outstanding > 0 else np.nan
        ev = mcap + (0.0 if np.isnan(debt) else debt) - (0.0 if np.isnan(cash) else cash) if not np.isnan(mcap) else np.nan

        pe = mcap / ni if not (np.isnan(mcap) or np.isnan(ni) or ni <= 0) else np.nan
        pb = mcap / equity if not (np.isnan(mcap) or np.isnan(equity) or equity <= 0) else np.nan
        ps = mcap / sales if not (np.isnan(mcap) or np.isnan(sales) or sales <= 0) else np.nan
        ev_ebitda = ev / ebitda if not (np.isnan(ev) or np.isnan(ebitda) or ebitda <= 0) else np.nan

        return {
            'pe_ratio': float(pe) if not np.isnan(pe) else np.nan,
            'pb_ratio': float(pb) if not np.isnan(pb) else np.nan,
            'ps_ratio': float(ps) if not np.isnan(ps) else np.nan,
            'ev_ebitda': float(ev_ebitda) if not np.isnan(ev_ebitda) else np.nan
        }

    def run(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        fundamental_data: Dict[str, Any],
        beta: float = 1.0,
        risk_free_rate: float = 0.04
    ) -> ModuleResult:
        """Executes full Valuation Module analysis."""
        logger.info(f"Running Valuation Module for {ticker}")
        info = fundamental_data.get('info', {})
        quote_type = str(info.get('quoteType', '')).upper()
        
        # ETFs, Indices, Commodities, and Trust Funds don't have corporate 10-K balance sheets.
        # They reflect broader market valuation rather than corporate DCF. Return a neutral 50.0 score.
        etf_tickers = {'SPY', 'QQQ', 'DIA', 'IWM', 'SMH', 'XLK', 'XLF', 'XLE', 'XLV', 'GLD', 'TLT', 'IBIT', '^GSPC', '^NDX', '^DJI', '^RUT'}
        if quote_type in ('ETF', 'MUTUALFUND', 'INDEX') or ticker.upper() in etf_tickers:
            return ModuleResult(
                ticker=ticker,
                module_name='Valuation',
                normalised_score=50.0,
                confidence=1.0,
                components={'type': 'ETF/Index Neutral'},
                metadata={'score_components': [50.0]}
            )

        cur_price = price_data['Close'].iloc[-1] if not price_data.empty else np.nan
        shares = info.get('sharesOutstanding', np.nan)
        if np.isnan(shares):
            shares_stmt = self._get_metric(fundamental_data, 'balance_sheet', 'Ordinary Shares Number', 0)
            shares = shares_stmt if not np.isnan(shares_stmt) else 1_000_000_000.0

        wacc_res = self.compute_wacc(fundamental_data, beta=beta, risk_free_rate=risk_free_rate)
        dcf_res = self.compute_probabilistic_dcf(fundamental_data, cur_price, wacc_res['wacc'], shares)
        rel_res = self.compute_relative_valuation(fundamental_data, cur_price, shares)

        score_components = []
        # 1. Margin of Safety (DCF) using continuous sigmoid mapping centered around 0% MOS
        mos = dcf_res.get('margin_of_safety', np.nan)
        if not np.isnan(mos):
            # Centered at 0 MOS (50 pts), +50% MOS gives ~85 pts, -50% gives ~15 pts
            dcf_score = 100.0 / (1.0 + np.exp(-3.0 * float(mos)))
            score_components.append(float(np.clip(dcf_score, 5.0, 95.0)))

        # 2. P/E Ratio using smooth non-linear curve (median market P/E ~ 22)
        pe = rel_res.get('pe_ratio', np.nan)
        if not np.isnan(pe) and pe > 0:
            # P/E of 10 gives ~80, P/E of 22 gives 50, P/E of 50 gives ~25
            pe_score = 100.0 / (1.0 + np.exp(0.08 * (float(pe) - 22.0)))
            score_components.append(float(np.clip(pe_score, 5.0, 95.0)))

        # 3. EV/EBITDA using smooth curve (median ~ 14)
        ev_ebitda = rel_res.get('ev_ebitda', np.nan)
        if not np.isnan(ev_ebitda) and ev_ebitda > 0:
            ev_score = 100.0 / (1.0 + np.exp(0.12 * (float(ev_ebitda) - 14.0)))
            score_components.append(float(np.clip(ev_score, 5.0, 95.0)))

        # 4. ETF / Fund Detection Fallback
        # If no corporate financial metrics are available (e.g. SPY, QQQ, GLD, TLT),
        # assign a neutral 50.0 baseline rather than 0 or 100.
        norm_score = float(np.mean(score_components)) if score_components else 50.0
        norm_score = float(np.clip(norm_score, 10.0, 90.0))

        return ModuleResult(
            ticker=ticker,
            module_name='Valuation',
            normalised_score=norm_score,
            confidence=len(score_components) / 3.0 if score_components else 0.2,
            components={
                'wacc': wacc_res,
                'dcf': dcf_res,
                'relative_valuation': rel_res
            },
            metadata={'score_components': score_components}
        )
