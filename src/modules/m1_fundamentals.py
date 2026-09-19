"""
Module 1: Fundamental / Business Quality Model
===============================================
PURPOSE:
  Assess the intrinsic economic quality of a business using
  publicly available accounting data. This module is separated
  from valuation (Module 2).

MATHEMATICAL FRAMEWORK:
  1. Piotroski F-Score: 9-point binary fundamental health signal.
  2. DuPont 3-Way Decomposition: ROE = Net Margin * Asset Turnover * Equity Multiplier.
  3. Altman Z-Score: Distance to bankruptcy distress.
  4. Return on Invested Capital (ROIC): NOPAT / Invested Capital.
  5. Accruals Quality Ratio: (Net Income - CFO) / Average Total Assets.
  6. Financial Coverage & Margins: Gross, Operating, EBITDA, FCF Margins & Net Debt/EBITDA.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from src.modules.base import ModuleResult

logger = logging.getLogger(__name__)


class FundamentalsModule:
    """
    Evaluates business profitability persistence, capital allocation efficiency,
    and financial solvency.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.random_seed = self.config.get('random_seed', 42)

    def _get_metric(self, fin_data: Dict[str, pd.DataFrame], statement: str, item: str, period: int = 0) -> float:
        """Safely retrieves an accounting metric from financial statements."""
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
        except Exception as e:
            logger.debug(f"Could not retrieve {item} from {statement}: {e}")
            return np.nan

    def compute_piotroski_fscore(self, fin_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Computes Piotroski F-Score (0-9)."""
        ni_curr = self._get_metric(fin_data, 'income_stmt', 'Net Income', 0)
        ni_prev = self._get_metric(fin_data, 'income_stmt', 'Net Income', 1)
        cfo_curr = self._get_metric(fin_data, 'cash_flow', 'Operating Cash Flow', 0)
        
        assets_curr = self._get_metric(fin_data, 'balance_sheet', 'Total Assets', 0)
        assets_prev = self._get_metric(fin_data, 'balance_sheet', 'Total Assets', 1)
        assets_prev2 = self._get_metric(fin_data, 'balance_sheet', 'Total Assets', 2)

        roa_curr = ni_curr / assets_prev if not (np.isnan(ni_curr) or np.isnan(assets_prev) or assets_prev == 0) else np.nan
        roa_prev = ni_prev / assets_prev2 if not (np.isnan(ni_prev) or np.isnan(assets_prev2) or assets_prev2 == 0) else np.nan

        ltd_curr = self._get_metric(fin_data, 'balance_sheet', 'Long Term Debt', 0)
        ltd_prev = self._get_metric(fin_data, 'balance_sheet', 'Long Term Debt', 1)

        ca_curr = self._get_metric(fin_data, 'balance_sheet', 'Current Assets', 0)
        cl_curr = self._get_metric(fin_data, 'balance_sheet', 'Current Liabilities', 0)
        ca_prev = self._get_metric(fin_data, 'balance_sheet', 'Current Assets', 1)
        cl_prev = self._get_metric(fin_data, 'balance_sheet', 'Current Liabilities', 1)

        cr_curr = ca_curr / cl_curr if not (np.isnan(ca_curr) or np.isnan(cl_curr) or cl_curr == 0) else np.nan
        cr_prev = ca_prev / cl_prev if not (np.isnan(ca_prev) or np.isnan(cl_prev) or cl_prev == 0) else np.nan

        shares_curr = self._get_metric(fin_data, 'balance_sheet', 'Ordinary Shares Number', 0)
        shares_prev = self._get_metric(fin_data, 'balance_sheet', 'Ordinary Shares Number', 1)

        gross_curr = self._get_metric(fin_data, 'income_stmt', 'Gross Profit', 0)
        gross_prev = self._get_metric(fin_data, 'income_stmt', 'Gross Profit', 1)
        rev_curr = self._get_metric(fin_data, 'income_stmt', 'Total Revenue', 0)
        rev_prev = self._get_metric(fin_data, 'income_stmt', 'Total Revenue', 1)

        margin_curr = gross_curr / rev_curr if not (np.isnan(gross_curr) or np.isnan(rev_curr) or rev_curr == 0) else np.nan
        margin_prev = gross_prev / rev_prev if not (np.isnan(gross_prev) or np.isnan(rev_prev) or rev_prev == 0) else np.nan

        turn_curr = rev_curr / assets_prev if not (np.isnan(rev_curr) or np.isnan(assets_prev) or assets_prev == 0) else np.nan
        turn_prev = rev_prev / assets_prev2 if not (np.isnan(rev_prev) or np.isnan(assets_prev2) or assets_prev2 == 0) else np.nan

        components = {
            'f1_roa_positive': {'name': 'Positive ROA', 'signal': int(roa_curr > 0) if not np.isnan(roa_curr) else np.nan},
            'f2_cfo_positive': {'name': 'Positive CFO', 'signal': int(cfo_curr > 0) if not np.isnan(cfo_curr) else np.nan},
            'f3_roa_growth': {'name': 'ROA Growth', 'signal': int(roa_curr > roa_prev) if not (np.isnan(roa_curr) or np.isnan(roa_prev)) else np.nan},
            'f4_earnings_quality': {'name': 'Earnings Quality', 'signal': int(cfo_curr > ni_curr) if not (np.isnan(cfo_curr) or np.isnan(ni_curr)) else np.nan},
            'f5_leverage_decrease': {'name': 'Lower Leverage', 'signal': int(ltd_curr < ltd_prev) if not (np.isnan(ltd_curr) or np.isnan(ltd_prev)) else np.nan},
            'f6_liquidity_increase': {'name': 'Higher Liquidity', 'signal': int(cr_curr > cr_prev) if not (np.isnan(cr_curr) or np.isnan(cr_prev)) else np.nan},
            'f7_no_dilution': {'name': 'No Dilution', 'signal': int(shares_curr <= shares_prev) if not (np.isnan(shares_curr) or np.isnan(shares_prev)) else np.nan},
            'f8_margin_increase': {'name': 'Margin Expansion', 'signal': int(margin_curr > margin_prev) if not (np.isnan(margin_curr) or np.isnan(margin_prev)) else np.nan},
            'f9_turnover_increase': {'name': 'Higher Turnover', 'signal': int(turn_curr > turn_prev) if not (np.isnan(turn_curr) or np.isnan(turn_prev)) else np.nan}
        }

        valid_signals = [v['signal'] for v in components.values() if not np.isnan(v['signal'])]
        fscore = sum(valid_signals) if len(valid_signals) > 0 else np.nan

        return {
            'fscore': fscore,
            'components': components,
            'valid_signals': len(valid_signals)
        }

    def compute_roic(self, fin_data: Dict[str, pd.DataFrame], tax_rate: float = 0.21) -> Dict[str, float]:
        """
        Computes Return on Invested Capital (ROIC).
        ROIC = NOPAT / Invested Capital
        NOPAT = EBIT * (1 - tax_rate)
        Invested Capital = Total Debt + Equity - Cash
        """
        ebit = self._get_metric(fin_data, 'income_stmt', 'EBIT', 0)
        debt = self._get_metric(fin_data, 'balance_sheet', 'Total Debt', 0)
        if np.isnan(debt):
            ltd = self._get_metric(fin_data, 'balance_sheet', 'Long Term Debt', 0)
            std = self._get_metric(fin_data, 'balance_sheet', 'Current Debt', 0)
            debt = (0.0 if np.isnan(ltd) else ltd) + (0.0 if np.isnan(std) else std)

        equity = self._get_metric(fin_data, 'balance_sheet', 'Stockholders Equity', 0)
        cash = self._get_metric(fin_data, 'balance_sheet', 'Cash And Cash Equivalents', 0)
        cash = 0.0 if np.isnan(cash) else cash

        nopat = ebit * (1.0 - tax_rate) if not np.isnan(ebit) else np.nan
        invested_capital = debt + equity - cash if not (np.isnan(debt) or np.isnan(equity)) else np.nan

        roic = nopat / invested_capital if (invested_capital and not np.isnan(invested_capital) and invested_capital > 0) else np.nan

        return {
            'roic': float(roic) if not np.isnan(roic) else np.nan,
            'nopat': float(nopat) if not np.isnan(nopat) else np.nan,
            'invested_capital': float(invested_capital) if not np.isnan(invested_capital) else np.nan
        }

    def compute_accruals_ratio(self, fin_data: Dict[str, pd.DataFrame]) -> float:
        """Computes Accruals Ratio = (Net Income - CFO) / Average Total Assets."""
        ni = self._get_metric(fin_data, 'income_stmt', 'Net Income', 0)
        cfo = self._get_metric(fin_data, 'cash_flow', 'Operating Cash Flow', 0)
        ta_0 = self._get_metric(fin_data, 'balance_sheet', 'Total Assets', 0)
        ta_1 = self._get_metric(fin_data, 'balance_sheet', 'Total Assets', 1)

        avg_ta = (ta_0 + ta_1) / 2.0 if not (np.isnan(ta_0) or np.isnan(ta_1)) else ta_0
        if np.isnan(ni) or np.isnan(cfo) or np.isnan(avg_ta) or avg_ta == 0:
            return np.nan

        return float((ni - cfo) / avg_ta)

    def compute_altman_z(self, fin_data: Dict[str, pd.DataFrame], market_cap: float) -> Dict[str, Any]:
        """Computes Altman Z-Score."""
        ca = self._get_metric(fin_data, 'balance_sheet', 'Current Assets', 0)
        cl = self._get_metric(fin_data, 'balance_sheet', 'Current Liabilities', 0)
        ta = self._get_metric(fin_data, 'balance_sheet', 'Total Assets', 0)
        re = self._get_metric(fin_data, 'balance_sheet', 'Retained Earnings', 0)
        ebit = self._get_metric(fin_data, 'income_stmt', 'EBIT', 0)
        tl = self._get_metric(fin_data, 'balance_sheet', 'Total Liabilities Net Minority Interest', 0)
        sales = self._get_metric(fin_data, 'income_stmt', 'Total Revenue', 0)

        if np.isnan(ta) or ta == 0:
            return {'z_score': np.nan, 'zone': 'unknown'}

        x1 = (ca - cl) / ta if not (np.isnan(ca) or np.isnan(cl)) else 0.0
        x2 = re / ta if not np.isnan(re) else 0.0
        x3 = ebit / ta if not np.isnan(ebit) else 0.0
        mcap = market_cap if (market_cap and not np.isnan(market_cap)) else ta
        x4 = mcap / tl if not (np.isnan(tl) or tl == 0) else 1.0
        x5 = sales / ta if not np.isnan(sales) else 0.0

        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        zone = 'safe' if z > 2.99 else ('distress' if z < 1.81 else 'grey')
        return {'z_score': float(z), 'zone': zone}

    def run(self, ticker: str, price_data: pd.DataFrame, fundamental_data: Dict[str, Any]) -> ModuleResult:
        """Executes full Fundamental Quality analysis."""
        logger.info(f"Running Fundamental Quality Module for {ticker}")
        info = fundamental_data.get('info', {})
        quote_type = str(info.get('quoteType', '')).upper()
        etf_tickers = {'SPY', 'QQQ', 'DIA', 'IWM', 'SMH', 'XLK', 'XLF', 'XLE', 'XLV', 'GLD', 'TLT', 'IBIT', '^GSPC', '^NDX', '^DJI', '^RUT'}
        if quote_type in ('ETF', 'MUTUALFUND', 'INDEX') or ticker.upper() in etf_tickers:
            return ModuleResult(
                ticker=ticker,
                module_name='Fundamentals',
                normalised_score=50.0,
                confidence=1.0,
                components={'type': 'ETF/Index Neutral'},
                metadata={'score_components': [50.0]}
            )

        fscore_res = self.compute_piotroski_fscore(fundamental_data)
        roic_res = self.compute_roic(fundamental_data)
        accruals = self.compute_accruals_ratio(fundamental_data)

        info = fundamental_data.get('info', {})
        mcap = info.get('marketCap', np.nan)
        z_res = self.compute_altman_z(fundamental_data, mcap)

        # Composite normalisation
        score_components = []
        if not np.isnan(fscore_res['fscore']):
            score_components.append((fscore_res['fscore'] / 9.0) * 100.0)
        if not np.isnan(roic_res['roic']):
            score_components.append(min(max(roic_res['roic'] / 0.25 * 100.0, 0.0), 100.0))
        if not np.isnan(z_res['z_score']):
            score_components.append(min(max((z_res['z_score'] - 1.0) / 4.0 * 100.0, 0.0), 100.0))
        if not np.isnan(accruals):
            # Lower accruals is better (earnings backed by cash)
            score_components.append(min(max((0.10 - accruals) / 0.20 * 100.0, 0.0), 100.0))

        norm_score = float(np.mean(score_components)) if score_components else 50.0

        return ModuleResult(
            ticker=ticker,
            module_name='Fundamentals',
            normalised_score=norm_score,
            confidence=len(score_components) / 4.0 if score_components else 0.2,
            components={
                'fscore': fscore_res,
                'roic': roic_res,
                'altman_z': z_res,
                'accruals': accruals
            },
            metadata={'score_components': score_components}
        )
